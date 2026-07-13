from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    grad_scaler: torch.amp.GradScaler | None = None,
    precision: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if grad_scaler is not None:
        payload["grad_scaler"] = grad_scaler.state_dict()
    if precision is not None:
        payload["precision"] = precision
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> int:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if grad_scaler is not None and "grad_scaler" in payload:
        grad_scaler.load_state_dict(payload["grad_scaler"])
    return int(payload.get("step", 0))
