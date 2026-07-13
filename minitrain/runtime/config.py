from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    name: str = "debug"
    seed: int = 1337


@dataclass(frozen=True)
class BackendConfig:
    ops: str = "torch"
    parallel: str = "single"


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 1000
    log_interval: int = 10
    use_fused_loss: bool = False
    precision: str = "fp32"
    grad_clip_norm: float | None = 1.0

    def __post_init__(self) -> None:
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("train.precision must be one of: fp32, bf16, fp16")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("train.grad_clip_norm must be positive or null")


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


@dataclass(frozen=True)
class ExperimentConfig:
    """Typed view over a YAML experiment file.

    Keep this small while the project is young. When configs grow, mirror
    TorchTitan's pattern: typed sections with explicit defaults and validation.
    """

    run: RunConfig = field(default_factory=RunConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
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
        train=TrainConfig(**payload.get("train", {})),
        data=DataConfig(**payload.get("data", {})),
        logging=LoggingConfig(**payload.get("logging", {})),
        distributed=dict(payload.get("distributed", {})),
        model=dict(payload.get("model", {})),
    )
