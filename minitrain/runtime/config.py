from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    name: str = "debug"
    seed: int = 1337

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("run.name must not be empty")


@dataclass(frozen=True)
class BackendConfig:
    ops: str = "torch"
    parallel: str = "single"

    def __post_init__(self) -> None:
        if self.ops not in {"torch", "triton", "cuda"}:
            raise ValueError("backend.ops must be one of: torch, triton, cuda")
        if self.parallel not in {"single", "ddp", "fsdp"}:
            raise ValueError("backend.parallel must be one of: single, ddp, fsdp")


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    fused: bool | None = None

    def __post_init__(self) -> None:
        if self.name != "adamw":
            raise ValueError("optimizer.name must be 'adamw'")
        if self.lr <= 0 or self.weight_decay < 0 or self.eps <= 0:
            raise ValueError("optimizer lr/eps must be positive and weight_decay non-negative")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("optimizer betas must be in [0, 1)")


@dataclass(frozen=True)
class LRSchedulerConfig:
    schedule: str = "cosine"
    warmup_steps: int = 100
    decay_steps: int | None = None
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.schedule not in {"constant", "cosine"}:
            raise ValueError("lr_scheduler.schedule must be 'constant' or 'cosine'")
        if self.warmup_steps < 0:
            raise ValueError("lr_scheduler.warmup_steps must be non-negative")
        if self.decay_steps is not None and self.decay_steps <= 0:
            raise ValueError("lr_scheduler.decay_steps must be positive or null")
        if (
            self.schedule == "cosine"
            and self.decay_steps is not None
            and self.decay_steps <= self.warmup_steps
        ):
            raise ValueError("cosine decay_steps must be greater than warmup_steps")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("lr_scheduler.min_lr_ratio must be in [0, 1]")


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 8
    max_steps: int | None = 1000
    epochs: int | None = None
    log_interval: int = 10
    use_fused_loss: bool = False
    precision: str = "fp32"
    grad_clip_norm: float | None = 1.0
    checkpoint_every_epochs: int | None = None
    checkpoint_dir: str = "checkpoints"
    save_final_checkpoint: bool = False
    resume_from: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.log_interval <= 0:
            raise ValueError("train.batch_size and train.log_interval must be positive")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("train.precision must be one of: fp32, bf16, fp16")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("train.grad_clip_norm must be positive or null")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("train.max_steps must be positive or null")
        if self.epochs is not None and self.epochs <= 0:
            raise ValueError("train.epochs must be positive or null")
        if self.max_steps is None and self.epochs is None:
            raise ValueError("train.max_steps and train.epochs cannot both be null")
        if self.checkpoint_every_epochs is not None and self.checkpoint_every_epochs <= 0:
            raise ValueError("train.checkpoint_every_epochs must be positive or null")


@dataclass(frozen=True)
class DataConfig:
    """Where training tokens come from.

    `source=random` is intentionally useful: it lets us test the whole training
    stack before a tokenizer or dataset preprocessing pipeline exists.
    """

    source: str = "random"
    path: str | None = None
    num_tokens: int = 100_000
    shuffle: bool = True

    def __post_init__(self) -> None:
        if self.source not in {"random", "tokens"}:
            raise ValueError("data.source must be 'random' or 'tokens'")
        if self.source == "tokens" and not self.path:
            raise ValueError("data.path is required when data.source='tokens'")
        if self.num_tokens <= 0:
            raise ValueError("data.num_tokens must be positive")


@dataclass(frozen=True)
class LoggingConfig:
    """Training observability outputs.

    Console logging is the cheap always-on path. TensorBoard is optional at
    runtime, but enabled in the default configs because it is useful for watching
    loss, throughput, and memory move during longer runs.
    """

    console: bool = True
    tensorboard: bool = True
    log_dir: str = "runs"
    flush_secs: int = 10

    def __post_init__(self) -> None:
        if self.flush_secs <= 0:
            raise ValueError("logging.flush_secs must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    """Typed view over a YAML experiment file.

    Keep this small while the project is young. When configs grow, mirror
    TorchTitan's pattern: typed sections with explicit defaults and validation.
    """

    run: RunConfig = field(default_factory=RunConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    distributed: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)


def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    """Load a YAML file if PyYAML is installed.

    PyYAML is kept optional so the scaffold can import in minimal environments.
    """

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install pyyaml to load config files.") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config at {path}")
    return payload


def experiment_config_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        run=RunConfig(**payload.get("run", {})),
        backend=BackendConfig(**payload.get("backend", {})),
        optimizer=OptimizerConfig(**payload.get("optimizer", {})),
        lr_scheduler=LRSchedulerConfig(**payload.get("lr_scheduler", {})),
        train=TrainConfig(**payload.get("train", {})),
        data=DataConfig(**payload.get("data", {})),
        logging=LoggingConfig(**payload.get("logging", {})),
        distributed=dict(payload.get("distributed", {})),
        model=dict(payload.get("model", {})),
    )
