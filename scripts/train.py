import argparse
import os
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from minitrain.data.dataloader import build_training_dataloader
from minitrain.model import MiniTransformer, ModelConfig
from minitrain.runtime.config import experiment_config_from_dict, load_yaml_dict
from minitrain.runtime.device import get_default_device
from minitrain.runtime.factory import build_ops_backend, build_parallel_strategy
from minitrain.runtime.logger import build_event_logger, get_tensorboard_log_dir
from minitrain.train.optim import build_optimizer
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
        return torch.device("cuda", int(torch.cuda.current_device()))
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {name}")


def memory_metrics_mb(device: torch.device) -> dict[str, float]:
    """Return memory metrics with device-specific names for TensorBoard."""

    if device.type == "cuda":
        return {
            "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / 1024**2, 2),
            "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved(device) / 1024**2, 2),
            "gpu_peak_memory_allocated_mb": round(
                torch.cuda.max_memory_allocated(device) / 1024**2,
                2,
            ),
        }
    _, peak_bytes = tracemalloc.get_traced_memory()
    return {"host_peak_memory_mb": round(peak_bytes / 1024**2, 2)}


def maybe_reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def repeat_dataloader(dataloader):
    """Yield batches forever without caching them in Python memory."""

    while True:
        yield from dataloader


def build_train_log_event(
    *,
    step: int,
    loss: float,
    tokens_seen: int,
    interval_tokens: int,
    interval_seconds: float,
    total_seconds: float,
    device: torch.device,
    world_size: int,
) -> dict[str, object]:
    global_tokens_seen = tokens_seen * world_size
    global_interval_tokens = interval_tokens * world_size
    payload: dict[str, object] = {
        "event": "train",
        "step": step,
        "loss": round(float(loss), 6),
        "tokens_seen": global_tokens_seen,
        "tokens_per_sec": round(global_interval_tokens / max(interval_seconds, 1e-12), 2),
        "avg_tokens_per_sec": round(global_tokens_seen / max(total_seconds, 1e-12), 2),
    }
    payload.update(memory_metrics_mb(device))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config", default="configs/model_15m.yaml")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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
    ops = build_ops_backend(cfg.backend)
    model_cfg = ModelConfig(**cfg.model)
    model = MiniTransformer(model_cfg, ops).to(device)
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
            "data_source": cfg.data.source,
            "batch_size": cfg.train.batch_size,
            "seq_len": model_cfg.seq_len,
            "tensorboard_dir": str(tensorboard_dir) if tensorboard_dir is not None else "disabled",
        }
    )

    try:
        strategy.setup()
        model = strategy.wrap_model(model)
        optimizer = build_optimizer(model, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        trainer = Trainer(
            model,
            optimizer,
            device=device,
            use_fused_loss=cfg.train.use_fused_loss,
        )

        max_steps = args.smoke_steps if args.smoke_steps > 0 else cfg.train.max_steps
        if device.type != "cuda":
            tracemalloc.start()
        maybe_reset_peak_memory(device)
        strategy.barrier()
        start_time = time.perf_counter()
        last_log_time = start_time
        last_log_tokens = 0

        # Repeat the dataloader so tiny smoke datasets can run for any requested
        # number of steps without the training script needing special cases.
        for batch in repeat_dataloader(dataloader):
            loss = trainer.train_step(batch)

            should_log = trainer.state.step == 1 or trainer.state.step % cfg.train.log_interval == 0
            should_stop = trainer.state.step >= max_steps
            if should_log or should_stop:
                strategy.barrier()
                now = time.perf_counter()
                interval_tokens = trainer.state.tokens_seen - last_log_tokens
                logger.log_event(
                    build_train_log_event(
                        step=trainer.state.step,
                        loss=float(loss),
                        tokens_seen=trainer.state.tokens_seen,
                        interval_tokens=interval_tokens,
                        interval_seconds=now - last_log_time,
                        total_seconds=now - start_time,
                        device=device,
                        world_size=world_size,
                    )
                )
                last_log_time = now
                last_log_tokens = trainer.state.tokens_seen
            if should_stop:
                break
    finally:
        logger.close()
        strategy.teardown()
        if device.type != "cuda" and tracemalloc.is_tracing():
            tracemalloc.stop()


if __name__ == "__main__":
    main()
