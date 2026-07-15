"""Triton RMSNorm implementation.

This file keeps the operator layering inside
mini-train-sys' backend architecture:

1. JIT kernels live in this one operator file.
2. Small Python launchers flatten model tensors, choose constexpr meta-params,
   allocate scratch buffers, and launch Triton.
3. `MiniTrainRMSNormFunction` bridges those launchers into PyTorch autograd.
4. `rmsnorm()` is the function consumed by `TritonOpsBackend`.

The model layer never imports this file directly. It only calls the `OpsBackend`
contract, so a CUDA C++ or future accelerator backend can replace this later
without touching transformer blocks.
"""

from __future__ import annotations

import math

import torch

from minitrain.kernels.amp import cast_cuda_autocast_activations
from minitrain.kernels.triton.cache import configure_triton_cache


configure_triton_cache()

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by environments without Triton.
    triton = None
    tl = None


_MAX_FUSED_SIZE = 65536
_BLOCK_ROW = 16
_BLOCK_ROW_SWITCH_N_ROWS = 4096 * 8


def _calculate_settings(n_cols: int) -> tuple[int, int]:
    """Choose Triton block settings from the normalized hidden dimension.

    One power-of-two block covers
    the whole hidden dimension, and larger reductions get more warps. Different
    `BLOCK_SIZE` / `num_warps` pairs are part of Triton's compile key, so each
    hidden-size bucket gets compiled once and then reused from cache.
    """

    if triton is None:
        raise RuntimeError("Triton is not installed. Install mini-train-sys[triton].")
    block_size = triton.next_power_of_2(n_cols)
    if block_size > _MAX_FUSED_SIZE:
        raise RuntimeError(
            f"RMSNorm hidden size {n_cols} is too large for one Triton block "
            f"(max supported block size: {_MAX_FUSED_SIZE})."
        )

    num_warps = 4
    if block_size >= 32768:
        num_warps = 32
    elif block_size >= 8192:
        num_warps = 16
    elif block_size >= 2048:
        num_warps = 8
    return block_size, num_warps


def _use_row_kernel(n_rows: int, block_size: int, row_mode: bool | None = None) -> bool:
    """Choose the row-vs-block routing rule.

    Row mode launches one Triton program per input row. That is good when each
    row has enough columns to amortize scheduling overhead, or when the number
    of rows is not huge. Block mode packs several short rows into one program,
    which helps small-hidden / large-row batches.
    """

    return block_size > 256 or n_rows < _BLOCK_ROW_SWITCH_N_ROWS or bool(row_mode)


def is_rmsnorm_supported(x: torch.Tensor, weight: torch.Tensor | None = None) -> bool:
    """Return whether this tensor should use the Triton RMSNorm path."""

    supported = (
        triton is not None
        and x.is_cuda
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
    )
    if weight is None:
        return supported
    return (
        supported
        and weight.is_cuda
        and weight.device == x.device
        and weight.dtype in (torch.float32, torch.float16, torch.bfloat16)
    )


if triton is not None:

    @triton.jit
    def _rmsnorm_forward_kernel(
        y_ptr,
        y_row_stride,
        x_ptr,
        x_row_stride,
        w_ptr,
        rstd_ptr,
        rstd_row_stride,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute one RMSNorm row per Triton program.

        `RSTD` is cached per row so backward does not recompute the reduction.
        The reduction runs in fp32; the store casts back to `Y`'s dtype, which
        is the input activation dtype.
        """

        row_idx = tl.program_id(0).to(tl.int64)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        x_row = tl.load(x_ptr + row_idx * x_row_stride + col_offsets, mask=mask, other=0.0)
        x_row = x_row.to(tl.float32)
        w_row = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)

        variance = tl.sum(x_row * x_row, axis=0) / n_cols
        rstd = tl.rsqrt(variance + eps)
        tl.store(rstd_ptr + row_idx * rstd_row_stride, rstd)

        y_row = x_row * rstd * w_row
        tl.store(y_ptr + row_idx * y_row_stride + col_offsets, y_row, mask=mask)

    @triton.jit
    def _block_rmsnorm_forward_kernel(
        y_ptr,
        y_row_stride,
        x_ptr,
        x_row_stride,
        w_ptr,
        rstd_ptr,
        rstd_row_stride,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_ROW: tl.constexpr,
    ):
        """Forward path for small hidden sizes and many rows.

        One Triton program owns a
        `[BLOCK_ROW, BLOCK_SIZE]` tile. The math is identical to the row kernel,
        but the scheduling overhead is spread across multiple short rows.
        """

        row_offsets = tl.program_id(0).to(tl.int64) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        row_mask = row_offsets < n_rows
        col_mask = col_offsets < n_cols

        x_tile = tl.load(
            x_ptr + row_offsets[:, None] * x_row_stride + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w_row = tl.load(w_ptr + col_offsets, mask=col_mask, other=0.0)

        variance = tl.sum(x_tile * x_tile, axis=1) / n_cols
        rstd = tl.rsqrt(variance + eps)
        tl.store(rstd_ptr + row_offsets * rstd_row_stride, rstd, mask=row_mask)

        y_tile = x_tile * rstd[:, None] * w_row[None, :]
        tl.store(
            y_ptr + row_offsets[:, None] * y_row_stride + col_offsets[None, :],
            y_tile,
            mask=row_mask[:, None] & col_mask[None, :],
        )

    @triton.jit
    def _rmsnorm_backward_kernel(
        dy_ptr,
        dy_row_stride,
        dx_ptr,
        dx_row_stride,
        x_ptr,
        x_row_stride,
        w_ptr,
        rstd_ptr,
        rstd_row_stride,
        partial_dw_ptr,
        partial_dw_row_stride,
        n_rows,
        n_cols,
        rows_per_program,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute dX and per-program partial dW.

        The math is the standard RMSNorm derivative:

        m = dY * W
        dX = rstd * (m - X * mean(m * X) * rstd^2)
        dW = sum(dY * X * rstd)

        Each program owns a contiguous row range and writes one partial dW row.
        Python sums those partials after the kernel returns.
        """

        program_id = tl.program_id(0).to(tl.int64)
        row_start = program_id * rows_per_program
        row_end = min(row_start + rows_per_program, n_rows)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        w_row = tl.load(w_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        dw_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for row_idx in range(row_start, row_end):
            dy_row = tl.load(
                dy_ptr + row_idx * dy_row_stride + col_offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            x_row = tl.load(
                x_ptr + row_idx * x_row_stride + col_offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            rstd = tl.load(rstd_ptr + row_idx * rstd_row_stride).to(tl.float32)

            m = dy_row * w_row
            projection = tl.sum(m * x_row, axis=0) / n_cols
            dx_row = rstd * (m - x_row * projection * rstd * rstd)
            dw_row += dy_row * x_row * rstd

            tl.store(dx_ptr + row_idx * dx_row_stride + col_offsets, dx_row, mask=mask)

        tl.store(partial_dw_ptr + program_id * partial_dw_row_stride + col_offsets, dw_row, mask=mask)

    @triton.jit
    def _block_rmsnorm_backward_kernel(
        dy_ptr,
        dy_row_stride,
        dx_ptr,
        dx_row_stride,
        x_ptr,
        x_row_stride,
        w_ptr,
        rstd_ptr,
        rstd_row_stride,
        partial_dw_ptr,
        partial_dw_row_stride,
        n_rows,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_ROW: tl.constexpr,
    ):
        """Backward path paired with `_block_rmsnorm_forward_kernel`.

        The grid still uses roughly one program per SM. Each program walks the
        row dimension in `BLOCK_ROW` chunks, accumulates a local dW vector, and
        writes one partial dW row for Python to reduce.
        """

        program_id = tl.program_id(0).to(tl.int64)
        num_programs = tl.num_programs(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < n_cols

        w_row = tl.load(w_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
        dw_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for row_start in range(program_id * BLOCK_ROW, n_rows, num_programs * BLOCK_ROW):
            row_offsets = row_start + tl.arange(0, BLOCK_ROW)
            row_mask = row_offsets < n_rows
            dy_tile = tl.load(
                dy_ptr + row_offsets[:, None] * dy_row_stride + col_offsets[None, :],
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            x_tile = tl.load(
                x_ptr + row_offsets[:, None] * x_row_stride + col_offsets[None, :],
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            rstd = tl.load(rstd_ptr + row_offsets * rstd_row_stride, mask=row_mask, other=0.0).to(tl.float32)

            m = dy_tile * w_row[None, :]
            projection = tl.sum(m * x_tile, axis=1) / n_cols
            dx_tile = rstd[:, None] * (m - x_tile * projection[:, None] * rstd[:, None] * rstd[:, None])
            dw_row += tl.sum(dy_tile * x_tile * rstd[:, None], axis=0)

            tl.store(
                dx_ptr + row_offsets[:, None] * dx_row_stride + col_offsets[None, :],
                dx_tile,
                mask=row_mask[:, None] & col_mask[None, :],
            )

        tl.store(partial_dw_ptr + program_id * partial_dw_row_stride + col_offsets, dw_row, mask=col_mask)


def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float, row_mode: bool | None = None):
    """Flatten `x`, choose kernel settings, launch forward, and save metadata."""

    if not is_rmsnorm_supported(x, weight):
        raise RuntimeError("Triton RMSNorm only supports CUDA fp32/fp16/bf16 tensors.")
    if weight.ndim != 1:
        raise ValueError(f"RMSNorm weight must be 1-D, got shape {tuple(weight.shape)}.")

    shape = x.shape
    hidden_size = shape[-1]
    x_2d = x.reshape(-1, hidden_size)
    n_rows, n_cols = x_2d.shape
    if weight.numel() != n_cols:
        raise ValueError(f"RMSNorm weight has {weight.numel()} elements, expected {n_cols}.")

    block_size, num_warps = _calculate_settings(n_cols)
    y = torch.empty_like(x_2d)
    # Keep cached inverse RMS in fp32; it is small compared with activations and
    # saves backward from repeating the row-wise reduction.
    rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    use_row_kernel = _use_row_kernel(n_rows, block_size, row_mode)

    if use_row_kernel:
        _rmsnorm_forward_kernel[(n_rows,)](
            y,
            y.stride(0),
            x_2d,
            x_2d.stride(0),
            weight,
            rstd,
            rstd.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    else:
        _block_rmsnorm_forward_kernel[(triton.cdiv(n_rows, _BLOCK_ROW),)](
            y,
            y.stride(0),
            x_2d,
            x_2d.stride(0),
            weight,
            rstd,
            rstd.stride(0),
            n_rows,
            n_cols,
            eps,
            BLOCK_SIZE=block_size,
            BLOCK_ROW=_BLOCK_ROW,
            num_warps=num_warps,
        )
    return y.view(*shape), x_2d, rstd, block_size, num_warps, use_row_kernel


def rmsnorm_backward(
    dy: torch.Tensor,
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    block_size: int,
    num_warps: int,
    use_row_kernel: bool,
):
    """Launch backward and reduce partial dW."""

    shape = dy.shape
    hidden_size = shape[-1]
    dy_2d = dy.reshape(-1, hidden_size)
    n_rows, n_cols = dy_2d.shape

    # One program per SM provides enough programs to occupy
    # the device while each program accumulates a local dW vector in registers.
    sm_count = torch.cuda.get_device_properties(dy.device).multi_processor_count
    rows_per_program = math.ceil(n_rows / sm_count)
    dx = torch.empty_like(dy_2d)
    partial_dw = torch.empty((sm_count, n_cols), dtype=torch.float32, device=weight.device)

    if use_row_kernel:
        _rmsnorm_backward_kernel[(sm_count,)](
            dy_2d,
            dy_2d.stride(0),
            dx,
            dx.stride(0),
            x_2d,
            x_2d.stride(0),
            weight,
            rstd,
            rstd.stride(0),
            partial_dw,
            partial_dw.stride(0),
            n_rows,
            n_cols,
            rows_per_program,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    else:
        _block_rmsnorm_backward_kernel[(sm_count,)](
            dy_2d,
            dy_2d.stride(0),
            dx,
            dx.stride(0),
            x_2d,
            x_2d.stride(0),
            weight,
            rstd,
            rstd.stride(0),
            partial_dw,
            partial_dw.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=block_size,
            BLOCK_ROW=_BLOCK_ROW,
            num_warps=num_warps,
        )
    return dx.view(*shape), partial_dw.sum(dim=0).to(dtype=weight.dtype)


class MiniTrainRMSNormFunction(torch.autograd.Function):
    """Autograd bridge around the Triton RMSNorm launchers."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        # Triton kernels assume compact last-dimension rows. Making tensors
        # contiguous here preserves the model/backend contract and keeps each
        # kernel focused on math instead of stride edge cases.
        x = x.contiguous()
        weight = weight.contiguous()
        y, x_2d, rstd, block_size, num_warps, use_row_kernel = rmsnorm_forward(x, weight, eps)
        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.block_size = block_size
        ctx.num_warps = num_warps
        ctx.use_row_kernel = use_row_kernel
        return y

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dy: torch.Tensor):
        x_2d, weight, rstd = ctx.saved_tensors
        dy = dy.contiguous()
        dx, dweight = rmsnorm_backward(
            dy,
            x_2d,
            weight,
            rstd,
            ctx.block_size,
            ctx.num_warps,
            ctx.use_row_kernel,
        )
        return dx, dweight, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Public Triton RMSNorm entry point used by `TritonOpsBackend`."""

    (x,) = cast_cuda_autocast_activations(x)
    return MiniTrainRMSNormFunction.apply(x, weight, eps)
