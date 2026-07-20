from __future__ import annotations

import torch

from minitrain.distributed.ddp import DDPStrategy
from minitrain.distributed.fsdp import FSDPStrategy
from minitrain.distributed.single import SingleDeviceStrategy
from minitrain.distributed.strategy import ParallelStrategy
from minitrain.model.ops import OpsBackend, get_ops_backend
from minitrain.model import MiniTransformer, ModelConfig
from minitrain.runtime.config import BackendConfig, ExperimentConfig


def build_ops_backend(cfg: BackendConfig) -> OpsBackend:
    """Build the operator backend selected by the experiment config."""

    return get_ops_backend(cfg.ops)


def build_model(
    model_payload: dict,
    ops: OpsBackend,
    *,
    activation_dtype: torch.dtype = torch.float32,
) -> MiniTransformer:
    """Build either dense or MoE blocks from the same model config schema."""

    model_cfg = ModelConfig(**model_payload)
    return MiniTransformer(model_cfg, ops, activation_dtype=activation_dtype)


def build_parallel_strategy(
    cfg: ExperimentConfig, *, resolved_precision: str | None = None
) -> ParallelStrategy:
    """Build the distributed strategy selected by the experiment config."""

    name = cfg.parallel.strategy
    if name == "single":
        return SingleDeviceStrategy()
    if name == "ddp":
        return DDPStrategy(
            process_group_backend=cfg.parallel.process_group_backend,
            timeout_minutes=cfg.parallel.timeout_minutes,
            **vars(cfg.parallel.ddp),
        )
    if name == "fsdp":
        precision = cfg.train.precision if resolved_precision is None else resolved_precision
        return FSDPStrategy(
            process_group_backend=cfg.parallel.process_group_backend,
            timeout_minutes=cfg.parallel.timeout_minutes,
            precision=precision,
            **vars(cfg.parallel.fsdp),
        )
    raise ValueError(f"Unknown parallel strategy: {name}")
