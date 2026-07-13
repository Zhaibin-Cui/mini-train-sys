from minitrain.distributed.ddp import DDPStrategy
from minitrain.distributed.fsdp import FSDPStrategy
from minitrain.distributed.single import SingleDeviceStrategy
from minitrain.distributed.strategy import ParallelStrategy
from minitrain.model.ops import OpsBackend, get_ops_backend
from minitrain.runtime.config import BackendConfig, ExperimentConfig


def build_ops_backend(cfg: BackendConfig) -> OpsBackend:
    """Build the operator backend selected by the experiment config."""

    return get_ops_backend(cfg.ops)


def build_parallel_strategy(cfg: ExperimentConfig) -> ParallelStrategy:
    """Build the distributed strategy selected by the experiment config."""

    name = cfg.backend.parallel
    if name == "single":
        return SingleDeviceStrategy()
    if name == "ddp":
        return DDPStrategy(**cfg.distributed)
    if name == "fsdp":
        return FSDPStrategy(precision=cfg.train.precision, **cfg.distributed)
    raise ValueError(f"Unknown parallel strategy: {name}")
