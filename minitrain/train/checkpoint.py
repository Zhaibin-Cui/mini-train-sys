"""Portable single-process, DDP, and FSDP training checkpoints.

Every rank participates in Distributed Checkpoint (DCP).  Model and Adam state
therefore remain sharded on save/load and can be resharded when world size
changes.  Small runtime and RNG files sit beside the DCP payload.
"""

from __future__ import annotations

import random
import re
import shutil
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)

# PyTorch 2.5 emits this deprecation from its internal FSDP checkpoint path on
# every save/load. It is not actionable in application code until PyTorch's DCP
# implementation switches to DTensor. Keep every other warning visible.
warnings.filterwarnings(
    "ignore",
    message=r"Please use DTensor instead.*",
    category=FutureWarning,
    module=r"torch\.distributed\..*",
)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _broadcast_rank_zero_status(status: int) -> int:
    if not dist.is_initialized():
        return status
    values = [status if _rank() == 0 else 0]
    dist.broadcast_object_list(values, src=0)
    return int(values[0])


def checkpoint_path(root: str | Path, run_name: str, *, epoch: int, step: int) -> Path:
    return Path(root) / run_name / f"epoch_{epoch:06d}_step_{step:09d}"


def _is_committed(path: Path) -> bool:
    return path.is_dir() and (path / "COMMITTED").is_file()


def resolve_resume_checkpoint(
    resume_from: str | Path,
    *,
    checkpoint_dir: str | Path,
    run_name: str,
) -> Path:
    selector = str(resume_from).lower()
    if selector not in {"latest", "safety"}:
        path = Path(resume_from)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        if path.is_dir() and not _is_committed(path):
            raise ValueError(f"Checkpoint is incomplete (no COMMITTED marker): {path}")
        return path

    run_dir = Path(checkpoint_dir) / run_name
    directories = [path for path in run_dir.glob("epoch_*_step_*") if _is_committed(path)]
    if selector == "safety":
        candidates = sorted(path for path in directories if (path / "SAFETY").is_file())
        if not candidates:
            raise FileNotFoundError(f"No safety checkpoint found in: {run_dir}")
        return candidates[-1]
    legacy_files = list(run_dir.glob("epoch_*_step_*.pt"))
    candidates = sorted([*directories, *legacy_files])
    if not candidates:
        raise FileNotFoundError(f"No committed checkpoints found in: {run_dir}")
    return candidates[-1]


def prune_checkpoints(
    checkpoint_dir: str | Path,
    run_name: str,
    *,
    keep_last: int,
    keep_safety: int = 0,
    safety_every_epochs: int | None = None,
    keep_model_exports: int | None = None,
) -> list[Path]:
    """Bound storage while retaining recent checkpoints and older safety anchors."""

    if keep_last <= 0:
        raise ValueError("keep_last must be positive")
    if keep_safety < 0:
        raise ValueError("keep_safety must be non-negative")
    if keep_safety and (safety_every_epochs is None or safety_every_epochs <= 0):
        raise ValueError("safety_every_epochs must be positive when keep_safety is enabled")
    if keep_model_exports is not None and keep_model_exports <= 0:
        raise ValueError("keep_model_exports must be positive or null")
    run_dir = Path(checkpoint_dir) / run_name
    directories = [path for path in run_dir.glob("epoch_*_step_*") if _is_committed(path)]
    legacy_files = list(run_dir.glob("epoch_*_step_*.pt"))
    checkpoints = sorted([*directories, *legacy_files])
    recent = checkpoints[-keep_last:]
    older = checkpoints[:-keep_last]
    safety: list[Path] = []
    if keep_safety and older:
        pattern = re.compile(r"^epoch_(\d+)_step_")

        def epoch_number(path: Path) -> int | None:
            match = pattern.match(path.name)
            return int(match.group(1)) if match else None

        milestones = [
            path
            for path in older
            if (epoch := epoch_number(path)) is not None
            and epoch % int(safety_every_epochs) == 0
        ]
        safety = milestones[-keep_safety:]
        if len(safety) < keep_safety:
            # Before the first milestone ages out of the recent window, retain
            # the oldest committed checkpoint as an immediate fallback.
            fillers = [path for path in older if path not in safety]
            safety = fillers[: keep_safety - len(safety)] + safety
    kept = set(recent) | set(safety)
    removed = [path for path in checkpoints if path not in kept]
    for path in removed:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for path in directories:
        marker = path / "SAFETY"
        if path in safety:
            marker.write_text("resume fallback\n", encoding="utf-8")
        elif marker.exists():
            marker.unlink()
    if keep_model_exports is not None:
        exported = [path for path in sorted(kept) if path.is_dir() and (path / "model.pt").is_file()]
        for path in exported[:-keep_model_exports]:
            (path / "model.pt").unlink()
    return removed


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"])


def _fill_missing_optimizer_group_options(
    optimizer: torch.optim.Optimizer,
    original_groups: list[dict[str, Any]],
) -> None:
    """Retain constructor options that DCP may omit from FSDP param groups.

    PyTorch 2.5's FSDP/DCP state-dict path can restore tensor state and the
    parameter mapping while dropping non-tensor AdamW options such as
    ``betas``.  Keep values that were present in the checkpoint, and fill only
    missing keys from the freshly constructed optimizer's matching group.
    """

    if len(optimizer.param_groups) != len(original_groups):
        raise ValueError(
            "Checkpoint changed the optimizer parameter-group count: "
            f"expected {len(original_groups)}, got {len(optimizer.param_groups)}"
        )
    for restored, original in zip(optimizer.param_groups, original_groups, strict=True):
        for key, value in original.items():
            if key != "params":
                restored.setdefault(key, value)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    epoch: int = 0,
    lr_step: int | None = None,
    tokens_seen: int = 0,
    grad_scaler: torch.amp.GradScaler | None = None,
    lr_scheduler: Any | None = None,
    precision: str | None = None,
    config: dict[str, Any] | None = None,
    export_model: bool = False,
    cpu_offload: bool = True,
    write: bool | None = None,
) -> Path:
    """Collectively save a durable checkpoint.

    ``write`` is retained only for API compatibility.  DCP requires every rank;
    publication of metadata and the final directory remains rank-zero-only.
    """

    del write
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    status = 0
    if _rank() == 0:
        if _is_committed(path):
            status = 1
        elif path.exists():
            status = 2
    status = _broadcast_rank_zero_status(status)
    if status == 1:
        return path
    if status == 2:
        raise FileExistsError(f"Refusing to overwrite incomplete checkpoint path: {path}")
    if _rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
    _barrier()

    options = StateDictOptions(cpu_offload=cpu_offload)
    model_state, optim_state = get_state_dict(model, optimizer, options=options)
    dcp.save(
        {"model": model_state, "optimizer": optim_state},
        checkpoint_id=temporary / "distributed",
    )
    torch.save(_rng_state(), temporary / f"rng_rank_{_rank():05d}.pt")

    exported_state = None
    if export_model:
        exported_state = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    _barrier()

    if _rank() == 0:
        runtime: dict[str, Any] = {
            "format_version": 4,
            "trainer": {
                "step": step,
                "lr_step": step if lr_step is None else lr_step,
                "epoch": epoch,
                "tokens_seen": tokens_seen,
            },
            "saved_world_size": _world_size(),
            "precision": precision,
            "config": config,
        }
        if grad_scaler is not None:
            runtime["grad_scaler"] = grad_scaler.state_dict()
        if lr_scheduler is not None:
            runtime["lr_scheduler"] = lr_scheduler.state_dict()
        torch.save(runtime, temporary / "runtime.pt")
        if exported_state is not None:
            torch.save(exported_state, temporary / "model.pt")
        (temporary / "COMMITTED").write_text("ok\n", encoding="utf-8")
        temporary.replace(path)
    _barrier()
    return path


def _restore_legacy_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    grad_scaler: torch.amp.GradScaler | None,
    lr_scheduler: Any | None,
    restore_rng: bool,
) -> dict[str, int | bool]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if grad_scaler is not None and "grad_scaler" in payload:
        grad_scaler.load_state_dict(payload["grad_scaler"])
    if lr_scheduler is not None and "lr_scheduler" in payload:
        lr_scheduler.load_state_dict(payload["lr_scheduler"])
    if restore_rng and "rng" in payload:
        _restore_rng(payload["rng"])
    trainer = payload.get("trainer", {})
    return {
        "step": int(trainer.get("step", payload.get("step", 0))),
        "lr_step": int(trainer.get("lr_step", trainer.get("step", payload.get("step", 0)))),
        "epoch": int(trainer.get("epoch", 0)),
        "tokens_seen": int(trainer.get("tokens_seen", 0)),
        "rng_restored": bool(restore_rng and "rng" in payload),
    }


def restore_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    grad_scaler: torch.amp.GradScaler | None = None,
    lr_scheduler: Any | None = None,
    restore_rng: bool = True,
) -> dict[str, int | bool]:
    """Collectively restore model/Adam state and reshard for the current world size."""

    path = Path(path)
    if path.is_file():
        return _restore_legacy_checkpoint(
            path, model, optimizer, grad_scaler, lr_scheduler, restore_rng
        )
    if not _is_committed(path):
        raise ValueError(f"Checkpoint is incomplete: {path}")
    if optimizer is None:
        raise ValueError("Training restore requires an optimizer")

    options = StateDictOptions(cpu_offload=True)
    original_optimizer_groups = [dict(group) for group in optimizer.param_groups]
    model_state, optim_state = get_state_dict(model, optimizer, options=options)
    state = {"model": model_state, "optimizer": optim_state}
    dcp.load(state, checkpoint_id=path / "distributed")
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
        options=options,
    )
    _fill_missing_optimizer_group_options(optimizer, original_optimizer_groups)
    runtime = torch.load(path / "runtime.pt", map_location="cpu", weights_only=False)
    if grad_scaler is not None and "grad_scaler" in runtime:
        grad_scaler.load_state_dict(runtime["grad_scaler"])
    if lr_scheduler is not None and "lr_scheduler" in runtime:
        lr_scheduler.load_state_dict(runtime["lr_scheduler"])

    rng_path = path / f"rng_rank_{_rank():05d}.pt"
    rng_restored = restore_rng and rng_path.is_file()
    if rng_restored:
        _restore_rng(torch.load(rng_path, map_location="cpu", weights_only=False))
    trainer = runtime.get("trainer", {})
    saved_world_size = int(runtime.get("saved_world_size", 1))
    saved_local_tokens = int(trainer.get("tokens_seen", 0))
    local_tokens = saved_local_tokens * saved_world_size // _world_size()
    return {
        "step": int(trainer.get("step", 0)),
        "lr_step": int(trainer.get("lr_step", trainer.get("step", 0))),
        "epoch": int(trainer.get("epoch", 0)),
        # Trainer tracks local tokens. Rescale that counter on elastic resume so
        # local_tokens * new_world_size preserves prior global token accounting.
        "tokens_seen": local_tokens,
        "rng_restored": rng_restored,
        "saved_world_size": saved_world_size,
    }


def load_model_state_dict_from_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    """Load only consolidated model weights for evaluation/probes, never Adam state."""

    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
        if not path.is_file():
            raise FileNotFoundError(
                "This distributed checkpoint has no probe export. Set "
                "checkpoint.export_model: true while training."
            )
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        state = payload["model"] if "model" in payload else payload
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint does not contain a model state dict: {path}")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    **kwargs: Any,
) -> int:
    return int(restore_training_checkpoint(path, model, optimizer, **kwargs)["step"])


def checkpoint_trainer_state(path: str | Path) -> dict[str, int]:
    path = Path(path)
    payload_path = path / "runtime.pt" if path.is_dir() else path
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    trainer = payload.get("trainer", {})
    return {
        "step": int(trainer.get("step", payload.get("step", 0))),
        "lr_step": int(trainer.get("lr_step", trainer.get("step", payload.get("step", 0)))),
        "epoch": int(trainer.get("epoch", 0)),
        "tokens_seen": int(trainer.get("tokens_seen", 0)),
    }
