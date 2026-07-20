"""Resolve no-accumulation batch-size scaling from a reference experiment."""

from __future__ import annotations

from dataclasses import dataclass, replace

from minitrain.runtime.config import LRSchedulerConfig, OptimizerConfig, TrainConfig


@dataclass(frozen=True)
class ResolvedBatchScale:
    local_batch_size: int
    global_batch_size: int
    reference_global_batch_size: int
    scale: float
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig


def resolve_batch_scale(
    train: TrainConfig,
    optimizer: OptimizerConfig,
    lr_scheduler: LRSchedulerConfig,
    *,
    world_size: int,
) -> ResolvedBatchScale:
    """Apply the linear-scaling rule while keeping epochs/person exposure fixed."""

    global_batch = train.batch_size * world_size
    reference_batch = train.reference_global_batch_size or global_batch
    scale = global_batch / reference_batch if train.batch_size_scaling == "linear" else 1.0

    def scale_steps(value: int | None) -> int | None:
        if value is None:
            return None
        if value == 0:
            return 0
        return max(1, round(value / scale))

    return ResolvedBatchScale(
        local_batch_size=train.batch_size,
        global_batch_size=global_batch,
        reference_global_batch_size=reference_batch,
        scale=scale,
        optimizer=replace(optimizer, lr=optimizer.lr * scale),
        lr_scheduler=replace(
            lr_scheduler,
            warmup_steps=scale_steps(lr_scheduler.warmup_steps) or 0,
            decay_steps=scale_steps(lr_scheduler.decay_steps),
        ),
    )
