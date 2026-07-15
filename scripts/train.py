import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from minitrain.data.dataloader import build_training_dataloader
from minitrain.runtime.config import experiment_config_from_dict, load_yaml_dict
from minitrain.runtime.device import get_default_device
from minitrain.runtime.factory import build_model, build_ops_backend, build_parallel_strategy
from minitrain.runtime.logger import build_event_logger, get_tensorboard_log_dir
from minitrain.train.optim import build_optimizer
from minitrain.train.checkpoint import (
    resolve_resume_checkpoint,
    restore_training_checkpoint,
)
from minitrain.train.lr_scheduler import LearningRateScheduler, resolve_total_steps
from minitrain.train.precision import resolve_precision_policy
from minitrain.train.runner import TrainingRunner
from minitrain.train.trainer import Trainer
from minitrain.utils.seed import seed_everything


def resolve_device(name: str) -> torch.device:
    """Choose the device requested by the CLI.

    `auto` keeps the friendly default: use CUDA when it exists, otherwise CPU.
    Explicit `cpu` is useful for smoke tests on any machine.
    """

    if name == "auto":
        return get_default_device()
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available")
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {name}")


def distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config", default="configs/model_default.yaml")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="PATH|latest",
        help="resume from a checkpoint; omit the value to use the latest checkpoint",
    )
    args = parser.parse_args()

    # The run config and model config are separate on purpose: it lets one model
    # size be reused across many backend/distributed experiments.
    run_payload = load_yaml_dict(args.config)
    model_payload = load_yaml_dict(args.model_config)
    run_payload["model"] = model_payload.get("model", {})
    cfg = experiment_config_from_dict(run_payload)

    seed_everything(cfg.run.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    precision = resolve_precision_policy(cfg.train.precision, device)
    ops = build_ops_backend(cfg.backend)
    model = build_model(
        cfg.model, ops, activation_dtype=precision.activation_dtype
    ).to(device)
    model_cfg = model.cfg
    dataloader = build_training_dataloader(
        cfg.data,
        seq_len=model_cfg.seq_len,
        batch_size=cfg.train.batch_size,
        vocab_size=model_cfg.vocab_size,
        seed=cfg.run.seed,
    )
    strategy = build_parallel_strategy(cfg)
    param_count = sum(p.numel() for p in model.parameters())
    world_size = distributed_world_size()
    tensorboard_dir = get_tensorboard_log_dir(cfg.logging, run_name=cfg.run.name)
    logger = build_event_logger(
        cfg.logging,
        run_name=cfg.run.name,
        tensorboard_log_dir=tensorboard_dir,
    )

    logger.log_event(
        {
            "event": "init",
            "run": cfg.run.name,
            "device": str(device),
            "ops": ops.name,
            "parallel": strategy.name,
            "world_size": world_size,
            "params": param_count,
            "ffn_type": model_cfg.ffn_type,
            "data_source": cfg.data.source,
            "batch_size": cfg.train.batch_size,
            "seq_len": model_cfg.seq_len,
            "precision": precision.name,
            "activation_dtype": str(precision.activation_dtype),
            "grad_scaler": precision.grad_scaling_enabled,
            "optimizer": cfg.optimizer.name,
            "learning_rate": cfg.optimizer.lr,
            "lr_schedule": cfg.lr_scheduler.schedule,
            "warmup_steps": cfg.lr_scheduler.warmup_steps,
            "tensorboard_dir": str(tensorboard_dir) if tensorboard_dir is not None else "disabled",
        }
    )

    try:
        strategy.setup()
        model = strategy.wrap_model(model)
        max_steps = args.smoke_steps if args.smoke_steps > 0 else cfg.train.max_steps
        total_steps = resolve_total_steps(
            max_steps=max_steps,
            epochs=cfg.train.epochs,
            steps_per_epoch=len(dataloader),
        )
        optimizer = build_optimizer(model, cfg=cfg.optimizer)
        lr_scheduler = LearningRateScheduler(
            optimizer,
            cfg.lr_scheduler,
            total_steps=total_steps,
        )
        trainer = Trainer(
            model,
            optimizer,
            device=device,
            use_fused_loss=cfg.train.use_fused_loss,
            precision=precision.name,
            grad_clip_norm=cfg.train.grad_clip_norm,
            lr_scheduler=lr_scheduler,
        )

        resume_value = args.resume if args.resume is not None else cfg.train.resume_from
        resume_path = None
        if resume_value is not None:
            resume_path = resolve_resume_checkpoint(
                resume_value,
                checkpoint_dir=cfg.train.checkpoint_dir,
                run_name=cfg.run.name,
            )
            restored = restore_training_checkpoint(
                resume_path,
                model,
                optimizer,
                grad_scaler=trainer.grad_scaler,
                lr_scheduler=lr_scheduler,
                restore_rng=True,
            )
            trainer.state.step = restored["step"]
            trainer.state.epoch = restored["epoch"]
            trainer.state.tokens_seen = restored["tokens_seen"]
            lr_scheduler.step(trainer.state.step)
            logger.log_event(
                {
                    "event": "resume",
                    "path": str(resume_path),
                    **restored,
                }
            )

        TrainingRunner(
            cfg=cfg,
            trainer=trainer,
            dataloader=dataloader,
            strategy=strategy,
            logger=logger,
            device=device,
            world_size=world_size,
        ).run(max_steps=max_steps, resume_path=resume_path)
    finally:
        logger.close()
        strategy.teardown()


if __name__ == "__main__":
    main()
