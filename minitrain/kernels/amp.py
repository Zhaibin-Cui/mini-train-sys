from __future__ import annotations

import torch


_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def cuda_autocast_activation_dtype() -> torch.dtype | None:
    """Return the active CUDA autocast dtype, or ``None`` when disabled."""

    if not torch.is_autocast_enabled("cuda"):
        return None
    dtype = torch.get_autocast_dtype("cuda")
    if dtype not in _LOW_PRECISION_DTYPES:
        raise RuntimeError(f"Unsupported CUDA autocast activation dtype: {dtype}")
    return dtype


def cast_cuda_autocast_activations(
    *tensors: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Cast CUDA floating-point activations to the active autocast dtype.

    Callers deliberately pass activations only. Parameters such as RMSNorm
    weights stay in their model/FSDP-managed dtype.
    """

    dtype = cuda_autocast_activation_dtype()
    if dtype is None:
        return tensors
    return tuple(
        tensor.to(dtype=dtype)
        if tensor.is_cuda and tensor.is_floating_point() and tensor.dtype != dtype
        else tensor
        for tensor in tensors
    )
