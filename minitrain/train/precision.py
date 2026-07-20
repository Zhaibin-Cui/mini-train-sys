from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import torch


_PRECISION_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def recommended_precision(device: torch.device) -> str:
    """Select the production mixed-precision policy for ``device``.

    CUDA prefers bf16 for its fp32-sized exponent range. Older CUDA devices
    fall back to fp16 with dynamic loss scaling, while CPU keeps fp32 as the
    conservative portable default.
    """

    if device.type != "cuda":
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


@dataclass(frozen=True)
class PrecisionPolicy:
    """Runtime precision contract shared by the model and trainer.

    Parameters stay in fp32. ``activation_dtype`` controls the residual stream
    and autocast controls eligible operator compute. Only fp16 needs dynamic
    loss scaling because bf16 has the same exponent range as fp32.
    """

    name: str
    activation_dtype: torch.dtype
    autocast_enabled: bool
    grad_scaling_enabled: bool

    def autocast_context(self, device: torch.device) -> ContextManager:
        if not self.autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=self.activation_dtype)


def resolve_precision_policy(name: str, device: torch.device) -> PrecisionPolicy:
    normalized = name.lower()
    if normalized == "auto":
        normalized = recommended_precision(device)
    if normalized not in _PRECISION_DTYPES:
        choices = "auto, " + ", ".join(_PRECISION_DTYPES)
        raise ValueError(f"Unknown precision {name!r}; expected one of: {choices}")

    if normalized == "fp16" and device.type != "cuda":
        raise ValueError("fp16 mixed-precision training is supported only on CUDA")
    if normalized == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 was requested, but the selected CUDA device does not support bf16")

    return PrecisionPolicy(
        name=normalized,
        activation_dtype=_PRECISION_DTYPES[normalized],
        autocast_enabled=normalized != "fp32",
        grad_scaling_enabled=normalized == "fp16",
    )
