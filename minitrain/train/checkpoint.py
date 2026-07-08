from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0))

