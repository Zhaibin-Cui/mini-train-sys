"""Small, atomic recovery checkpoints for independent SynBioS probe jobs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


PROBE_RECOVERY_FORMAT_VERSION = 2


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def trainable_probe_state(probe: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return only the probe head and embedding delta, never the frozen backbone."""

    return {
        key: value.detach().cpu()
        for key, value in probe.state_dict().items()
        if not key.startswith("backbone.")
    }


def save_probe_result(
    path: str | Path,
    *,
    probe: torch.nn.Module,
    result: dict[str, object],
) -> None:
    """Atomically publish a completed probe checkpoint as the task marker."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save({"probe": trainable_probe_state(probe), "result": result}, temporary)
    os.replace(temporary, destination)


def save_probe_recovery(
    path: str | Path,
    *,
    probe: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss_curve: list[dict[str, object]],
    data_generator: torch.Generator,
    epoch_generator_state: torch.Tensor,
    batches_consumed_in_epoch: int,
    metadata: dict[str, object],
) -> None:
    """Atomically save enough state to continue a probe without saving its backbone."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload: dict[str, Any] = {
        "format_version": PROBE_RECOVERY_FORMAT_VERSION,
        "metadata": metadata,
        "step": int(step),
        "probe": trainable_probe_state(probe),
        "optimizer": optimizer.state_dict(),
        "loss_curve": loss_curve,
        "data_generator_state": data_generator.get_state(),
        "epoch_generator_state": epoch_generator_state,
        "batches_consumed_in_epoch": int(batches_consumed_in_epoch),
        "torch_rng_state": torch.get_rng_state(),
    }
    probe_device = _module_device(probe)
    if probe_device.type == "cuda":
        # Each probe worker owns one GPU. Reading every CUDA RNG state would
        # unnecessarily create contexts on the other workers' GPUs.
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(probe_device)
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_probe_recovery(
    path: str | Path,
    *,
    probe: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_generator: torch.Generator,
    expected_metadata: dict[str, object],
) -> dict[str, object]:
    """Restore a recovery checkpoint and reject state from a different probe job."""

    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("format_version") != PROBE_RECOVERY_FORMAT_VERSION:
        raise ValueError(f"unsupported probe recovery format: {payload.get('format_version')}")
    if payload.get("metadata") != expected_metadata:
        raise ValueError("probe recovery metadata does not match the requested job")
    incompatible = probe.load_state_dict(payload["probe"], strict=False)
    if incompatible.unexpected_keys or any(
        not key.startswith("backbone.") for key in incompatible.missing_keys
    ):
        raise ValueError(f"incompatible probe recovery state: {incompatible}")
    optimizer.load_state_dict(payload["optimizer"])
    data_generator.set_state(payload["data_generator_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    probe_device = _module_device(probe)
    if probe_device.type == "cuda" and "cuda_rng_state" in payload:
        torch.cuda.set_rng_state(payload["cuda_rng_state"], device=probe_device)
    return {
        "step": int(payload["step"]),
        "loss_curve": list(payload.get("loss_curve", [])),
        "epoch_generator_state": payload["epoch_generator_state"],
        "batches_consumed_in_epoch": int(payload["batches_consumed_in_epoch"]),
    }
