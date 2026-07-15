from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch


def checkpoint_path(
    root: str | Path, run_name: str, *, epoch: int, step: int
) -> Path:
    return Path(root) / run_name / f"epoch_{epoch:06d}_step_{step:09d}.pt"


def resolve_resume_checkpoint(
    resume_from: str | Path,
    *,
    checkpoint_dir: str | Path,
    run_name: str,
) -> Path:
    if str(resume_from).lower() != "latest":
        path = Path(resume_from)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path

    run_dir = Path(checkpoint_dir) / run_name
    candidates = sorted(run_dir.glob("epoch_*_step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in: {run_dir}")
    return candidates[-1]


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    epoch: int = 0,
    tokens_seen: int = 0,
    grad_scaler: torch.amp.GradScaler | None = None,
    lr_scheduler: Any | None = None,
    precision: str | None = None,
    config: dict[str, Any] | None = None,
    write: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 3,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "trainer": {"step": step, "epoch": epoch, "tokens_seen": tokens_seen},
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if grad_scaler is not None:
        payload["grad_scaler"] = grad_scaler.state_dict()
    if lr_scheduler is not None:
        payload["lr_scheduler"] = lr_scheduler.state_dict()
    if precision is not None:
        payload["precision"] = precision
    if config is not None:
        payload["config"] = config
    if write:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
    return path


def restore_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    grad_scaler: torch.amp.GradScaler | None = None,
    lr_scheduler: Any | None = None,
    restore_rng: bool = True,
) -> dict[str, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if grad_scaler is not None and "grad_scaler" in payload:
        grad_scaler.load_state_dict(payload["grad_scaler"])
    if lr_scheduler is not None and "lr_scheduler" in payload:
        lr_scheduler.load_state_dict(payload["lr_scheduler"])
    if restore_rng and "rng" in payload:
        random.setstate(payload["rng"]["python"])
        torch.set_rng_state(payload["rng"]["torch"])
        if torch.cuda.is_available() and payload["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    trainer = payload.get("trainer", {})
    return {
        "step": int(trainer.get("step", payload.get("step", 0))),
        "epoch": int(trainer.get("epoch", 0)),
        "tokens_seen": int(trainer.get("tokens_seen", 0)),
    }


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    grad_scaler: torch.amp.GradScaler | None = None,
    lr_scheduler: Any | None = None,
    restore_rng: bool = False,
) -> int:
    state = restore_training_checkpoint(
        path,
        model,
        optimizer,
        grad_scaler=grad_scaler,
        lr_scheduler=lr_scheduler,
        restore_rng=restore_rng,
    )
    return state["step"]


def checkpoint_trainer_state(path: str | Path) -> dict[str, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    trainer = payload.get("trainer", {})
    return {
        "step": int(trainer.get("step", payload.get("step", 0))),
        "epoch": int(trainer.get("epoch", 0)),
        "tokens_seen": int(trainer.get("tokens_seen", 0)),
    }
