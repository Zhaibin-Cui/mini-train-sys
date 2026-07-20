"""Assemble configured data, model, precision, distribution, and training."""

import argparse
import os
import sys
from pathlib import Path

# ruff: noqa: E402 -- direct script execution needs the repository root on sys.path.

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from minitrain.data.dataloader import build_training_dataloader
from minitrain.runtime.config import experiment_config_from_dict, load_yaml_dict
from minitrain.runtime.device import get_default_device
from minitrain.runtime.factory import build_model, build_ops_backend, build_parallel_strategy
from minitrain.runtime.logger import build_event_logger, get_run_log_dir
from minitrain.runtime.provenance import collect_provenance
from minitrain.runtime.scaling import resolve_batch_scale
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
        metavar="PATH|latest|safety",
        help=(
            "resume from a checkpoint; omit the value to use latest, or pass safety "
            "to use the retained fallback anchor"
        ),
    )
    args = parser.parse_args()

    # ---- Configuration -----------------------------------------------------
    # The run config and model config are separate on purpose: it lets one model
    # size be reused across many backend/distributed experiments.
    run_payload = load_yaml_dict(args.config)
    model_payload = load_yaml_dict(args.model_config)
    run_payload["model"] = model_payload.get("model", {})
    cfg = experiment_config_from_dict(run_payload)

    # ---- Runtime/data/model assembly --------------------------------------
    # Seed before model construction so every DDP rank starts from identical
    # parameters.  The DataLoader derives deterministic sampler plans from it.
    seed_everything(cfg.run.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    precision = resolve_precision_policy(cfg.train.precision, device)
    ops = build_ops_backend(cfg.backend)
    model = build_model(cfg.model, ops, activation_dtype=precision.activation_dtype).to(device)
    model_cfg = model.cfg
    world_size = distributed_world_size()
    if (
        cfg.parallel.expected_world_size is not None
        and world_size != cfg.parallel.expected_world_size
    ):
        raise RuntimeError(
            f"Config expects WORLD_SIZE={cfg.parallel.expected_world_size}, got {world_size}"
        )
    batch_scale = resolve_batch_scale(
        cfg.train,
        cfg.optimizer,
        cfg.lr_scheduler,
        world_size=world_size,
    )
    dataloader = build_training_dataloader(
        cfg.data,
        seq_len=model_cfg.seq_len,
        batch_size=cfg.train.batch_size,
        vocab_size=model_cfg.vocab_size,
        seed=cfg.run.seed,
    )
    strategy = build_parallel_strategy(cfg, resolved_precision=precision.name)
    param_count = sum(p.numel() for p in model.parameters())
    run_log_dir = get_run_log_dir(cfg.logging, run_name=cfg.run.name)
    logger = build_event_logger(
        cfg.logging,
        run_name=cfg.run.name,
        tensorboard_log_dir=run_log_dir,
    )

    # Record the fully resolved runtime contract once, including provenance and
    # dtype decisions that may have originated from an "auto" setting.
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
            "data_shuffle": cfg.data.shuffle,
            "data_shuffle_window": cfg.data.shuffle_window,
            "data_max_open_shards": cfg.data.max_open_shards,
            "data_packing": cfg.data.packing,
            "data_num_workers": dataloader.num_workers,
            "data_workers_per_node": getattr(dataloader, "minitrain_worker_budget", 0),
            "data_prefetch_factor": cfg.data.prefetch_factor,
            "data_pin_memory": cfg.data.pin_memory,
            "data_persistent_workers": cfg.data.persistent_workers,
            "data_drop_last": cfg.data.drop_last,
            "local_batch_size": batch_scale.local_batch_size,
            "global_batch_size": batch_scale.global_batch_size,
            "reference_global_batch_size": batch_scale.reference_global_batch_size,
            "batch_size_scale": batch_scale.scale,
            "seq_len": model_cfg.seq_len,
            "max_steps": cfg.train.max_steps,
            "epochs": cfg.train.epochs,
            "checkpoint_every_epochs": cfg.checkpoint.every_epochs,
            "checkpoint_keep_last": cfg.checkpoint.keep_last,
            "precision": precision.name,
            "activation_dtype": str(precision.activation_dtype),
            "grad_scaler": precision.grad_scaling_enabled,
            "optimizer": cfg.optimizer.name,
            "reference_learning_rate": cfg.optimizer.lr,
            "learning_rate": batch_scale.optimizer.lr,
            "lr_schedule": batch_scale.lr_scheduler.schedule,
            "warmup_steps": batch_scale.lr_scheduler.warmup_steps,
            "run_log_dir": str(run_log_dir) if run_log_dir is not None else "disabled",
            **collect_provenance(ROOT),
        }
    )

    try:
        # ---- Parallel wrapping and optimization ---------------------------
        # Process-group setup precedes wrapping; the optimizer must see the
        # wrapped model parameters (especially for sharded strategies).
        strategy.setup()
        model = strategy.wrap_model(model)
        if strategy.name == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel

            fsdp_units = sum(
                isinstance(module, FullyShardedDataParallel) for module in model.modules()
            )
            expected_units = (
                model_cfg.n_layers + 1
                if cfg.parallel.fsdp.auto_wrap_policy == "transformer_block"
                else 1
            )
            if fsdp_units != expected_units:
                raise RuntimeError(
                    f"FSDP wrap verification failed: found {fsdp_units} units, "
                    f"expected {expected_units}"
                )
            logger.log_event(
                {
                    "event": "parallel_ready",
                    "strategy": "fsdp",
                    "fsdp_units": fsdp_units,
                    "auto_wrap_policy": cfg.parallel.fsdp.auto_wrap_policy,
                    "sharding_strategy": cfg.parallel.fsdp.sharding_strategy,
                }
            )
        max_steps = args.smoke_steps if args.smoke_steps > 0 else cfg.train.max_steps
        total_steps = resolve_total_steps(
            max_steps=max_steps,
            epochs=cfg.train.epochs,
            steps_per_epoch=len(dataloader),
        )
        optimizer = build_optimizer(model, cfg=batch_scale.optimizer)
        lr_scheduler = LearningRateScheduler(
            optimizer,
            batch_scale.lr_scheduler,
            total_steps=total_steps,
        )
        trainer = Trainer(
            model,
            optimizer,
            device=device,
            use_fused_loss=cfg.train.use_fused_loss,
            precision=precision.name,
            grad_clip_norm=cfg.train.grad_clip_norm,
            check_finite=cfg.train.check_finite,
            lr_scheduler=lr_scheduler,
        )

        # ---- Optional state restoration -----------------------------------
        # Restore model/optimizer/scaler/scheduler/RNG as one checkpoint unit,
        # then mirror scalar counters into the in-memory Trainer state.
        resume_value = args.resume if args.resume is not None else cfg.checkpoint.resume_from
        resume_path = None
        if resume_value is not None:
            resume_path = resolve_resume_checkpoint(
                resume_value,
                checkpoint_dir=cfg.checkpoint.directory,
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
            trainer.state.lr_step = restored["lr_step"]
            if restored.get("saved_world_size", world_size) != world_size:
                # Epoch checkpoints are the durable contract. Re-express LR
                # progress in the new rank layout while Adam keeps its true
                # per-parameter update counters from DCP.
                trainer.state.lr_step = trainer.state.epoch * len(dataloader)
            if not restored["rng_restored"]:
                # New ranks after an elastic/world-size change have no matching
                # RNG sidecar. Model/Adam are exact; seed their future streams
                # deterministically without claiming bitwise continuation.
                seed_everything(cfg.run.seed + strategy.rank + trainer.state.step)
            lr_scheduler.step(trainer.state.lr_step)
            logger.log_event(
                {
                    "event": "resume",
                    "path": str(resume_path),
                    **restored,
                }
            )

        # ---- Epoch/step execution -----------------------------------------
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
