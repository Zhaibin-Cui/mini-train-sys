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

    def __post_init__(self) -> None:
        if self.ops not in {"torch", "triton", "cuda"}:
            raise ValueError("backend.ops must be one of: torch, triton, cuda")


@dataclass(frozen=True)
class DDPConfig:
    """Knobs that apply only to DistributedDataParallel."""

    broadcast_buffers: bool = False
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False

    def __post_init__(self) -> None:
        if self.bucket_cap_mb <= 0:
            raise ValueError("parallel.ddp.bucket_cap_mb must be positive")


@dataclass(frozen=True)
class FSDPConfig:
    """FSDP1 policy for block sharding, prefetch, precision, and memory tradeoffs."""

    sharding_strategy: str = "full_shard"
    auto_wrap_policy: str = "transformer_block"
    backward_prefetch: str = "backward_pre"
    forward_prefetch: bool = False
    limit_all_gathers: bool = True
    use_orig_params: bool = True
    sync_module_states: bool = True
    cpu_offload: bool = False
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.sharding_strategy not in {"full_shard", "shard_grad_op"}:
            raise ValueError(
                "parallel.fsdp.sharding_strategy must be 'full_shard' or 'shard_grad_op'"
            )
        if self.auto_wrap_policy not in {"transformer_block", "none"}:
            raise ValueError(
                "parallel.fsdp.auto_wrap_policy must be 'transformer_block' or 'none'"
            )
        if self.backward_prefetch not in {"backward_pre", "backward_post", "none"}:
            raise ValueError(
                "parallel.fsdp.backward_prefetch must be backward_pre, backward_post, or none"
            )


@dataclass(frozen=True)
class ParallelConfig:
    """Common process-group settings plus isolated DDP and FSDP sub-configs."""

    strategy: str = "single"
    process_group_backend: str = "auto"
    timeout_minutes: int = 30
    # Optional topology guard for fixed single-node server presets.
    expected_world_size: int | None = None
    ddp: DDPConfig = field(default_factory=DDPConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)

    def __post_init__(self) -> None:
        if self.strategy not in {"single", "ddp", "fsdp"}:
            raise ValueError("parallel.strategy must be one of: single, ddp, fsdp")
        if self.process_group_backend not in {"auto", "nccl", "gloo"}:
            raise ValueError("parallel.process_group_backend must be auto, nccl, or gloo")
        if self.timeout_minutes <= 0:
            raise ValueError("parallel.timeout_minutes must be positive")
        if self.expected_world_size is not None and self.expected_world_size <= 0:
            raise ValueError("parallel.expected_world_size must be positive or null")


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
    precision: str = "auto"
    grad_clip_norm: float | None = 5.0
    check_finite: bool = True
    reference_global_batch_size: int | None = None
    batch_size_scaling: str = "none"

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.log_interval <= 0:
            raise ValueError("train.batch_size and train.log_interval must be positive")
        if self.precision not in {"auto", "fp32", "bf16", "fp16"}:
            raise ValueError("train.precision must be one of: auto, fp32, bf16, fp16")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("train.grad_clip_norm must be positive or null")
        if self.reference_global_batch_size is not None and self.reference_global_batch_size <= 0:
            raise ValueError("train.reference_global_batch_size must be positive or null")
        if self.batch_size_scaling not in {"none", "linear"}:
            raise ValueError("train.batch_size_scaling must be 'none' or 'linear'")
        if self.batch_size_scaling != "none" and self.reference_global_batch_size is None:
            raise ValueError("linear batch scaling requires train.reference_global_batch_size")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("train.max_steps must be positive or null")
        if self.epochs is not None and self.epochs <= 0:
            raise ValueError("train.epochs must be positive or null")
        if self.max_steps is None and self.epochs is None:
            raise ValueError("train.max_steps and train.epochs cannot both be null")


@dataclass(frozen=True)
class CheckpointConfig:
    """Durable checkpoint cadence, retention, resume, and probe-export policy."""

    directory: str = "checkpoints"
    every_epochs: int | None = 1
    keep_last: int | None = None
    keep_safety: int = 0
    safety_every_epochs: int | None = None
    keep_model_exports: int | None = None
    save_final: bool = False
    resume_from: str | None = None
    export_model: bool = False
    cpu_offload: bool = True

    def __post_init__(self) -> None:
        if self.every_epochs is not None and self.every_epochs <= 0:
            raise ValueError("checkpoint.every_epochs must be positive or null")
        if self.keep_last is not None and self.keep_last <= 0:
            raise ValueError("checkpoint.keep_last must be positive or null")
        if self.keep_safety < 0:
            raise ValueError("checkpoint.keep_safety must be non-negative")
        if self.keep_safety and self.keep_last is None:
            raise ValueError("checkpoint.keep_safety requires checkpoint.keep_last")
        if self.keep_safety and (
            self.safety_every_epochs is None or self.safety_every_epochs <= 0
        ):
            raise ValueError(
                "checkpoint.safety_every_epochs must be positive when keep_safety is enabled"
            )
        if self.keep_model_exports is not None and self.keep_model_exports <= 0:
            raise ValueError("checkpoint.keep_model_exports must be positive or null")
        if not self.directory.strip():
            raise ValueError("checkpoint.directory must not be empty")


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
    packing: str = "contiguous"
    tokenizer_fingerprint: str | None = None
    shuffle_window: int = 1024
    max_open_shards: int = 8
    # null means topology-aware auto allocation. An integer is an explicit
    # per-rank override; use 0 for synchronous CPU/debug loading.
    num_workers: int | None = None
    worker_budget: int | None = 32
    max_workers_per_rank: int = 4
    worker_cpu_affinity: bool = True
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = True

    def __post_init__(self) -> None:
        if self.source not in {"random", "tokens", "token_shards"}:
            raise ValueError("data.source must be 'random', 'tokens', or 'token_shards'")
        if self.source in {"tokens", "token_shards"} and not self.path:
            raise ValueError(f"data.path is required when data.source={self.source!r}")
        if self.packing not in {"contiguous", "randomized_documents"}:
            raise ValueError("data.packing must be 'contiguous' or 'randomized_documents'")
        if self.packing != "contiguous" and self.source != "token_shards":
            raise ValueError(
                "data.packing='randomized_documents' requires data.source='token_shards'"
            )
        if self.num_tokens <= 0:
            raise ValueError("data.num_tokens must be positive")
        if self.shuffle_window <= 0:
            raise ValueError("data.shuffle_window must be positive")
        if self.max_open_shards <= 0:
            raise ValueError("data.max_open_shards must be positive")
        if self.num_workers is not None and self.num_workers < 0:
            raise ValueError("data.num_workers must be non-negative or null")
        if self.worker_budget is not None and self.worker_budget <= 0:
            raise ValueError("data.worker_budget must be positive or null")
        if self.max_workers_per_rank <= 0:
            raise ValueError("data.max_workers_per_rank must be positive")
        if self.prefetch_factor <= 0:
            raise ValueError("data.prefetch_factor must be positive")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("data.persistent_workers requires data.num_workers > 0")


@dataclass(frozen=True)
class LoggingConfig:
    """Training observability outputs.

    Console logging is the cheap live path, JSONL is the durable audit trail,
    and TensorBoard is useful for watching longer runs.
    """

    console: bool = True
    tensorboard: bool = True
    jsonl: bool = True
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
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: dict[str, Any] = field(default_factory=dict)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge configuration mappings without mutating either input."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_dict(path: str | Path, *, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load YAML plus optional relative ``extends`` files in deterministic order."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install pyyaml to load config files.") from exc
    path = Path(path).resolve()
    if path in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, path))
        raise ValueError(f"Configuration extends cycle: {chain}")
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config at {path}")
    parents = payload.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(item, str) for item in parents):
        raise ValueError(f"extends must be a path or list of paths in {path}")
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_payload = load_yaml_dict(path.parent / parent, _stack=(*_stack, path))
        merged = _deep_merge(merged, parent_payload)
    return _deep_merge(merged, payload)


def experiment_config_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    parallel_payload = dict(payload.get("parallel", {}))
    ddp = DDPConfig(**parallel_payload.pop("ddp", {}))
    fsdp = FSDPConfig(**parallel_payload.pop("fsdp", {}))
    return ExperimentConfig(
        run=RunConfig(**payload.get("run", {})),
        backend=BackendConfig(**payload.get("backend", {})),
        parallel=ParallelConfig(**parallel_payload, ddp=ddp, fsdp=fsdp),
        optimizer=OptimizerConfig(**payload.get("optimizer", {})),
        lr_scheduler=LRSchedulerConfig(**payload.get("lr_scheduler", {})),
        train=TrainConfig(**payload.get("train", {})),
        checkpoint=CheckpointConfig(**payload.get("checkpoint", {})),
        data=DataConfig(**payload.get("data", {})),
        logging=LoggingConfig(**payload.get("logging", {})),
        model=dict(payload.get("model", {})),
    )
