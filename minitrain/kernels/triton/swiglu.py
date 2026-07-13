"""Triton SwiGLU implementation.

This operator follows the same layering as the RMSNorm Triton port:

1. JIT kernels implement the row-wise forward and backward math.
2. Python launchers flatten model tensors and choose constexpr meta-params.
3. `MiniTrainSwiGLUFunction` bridges those launchers into PyTorch autograd.
4. `swiglu()` is the function consumed by `TritonOpsBackend`.

The model layer keeps calling `OpsBackend.swiglu(gate, up)`, so future fused MLP
variants can replace this operator without changing transformer blocks.
"""

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


def _calculate_settings(n_cols: int) -> tuple[int, int]:
    """Choose Triton block settings for one activation row."""

    if triton is None:
        raise RuntimeError("Triton is not installed. Install mini-train-sys[triton].")

    block_size = triton.next_power_of_2(n_cols)
    if block_size > _MAX_FUSED_SIZE:
        raise RuntimeError(
            f"SwiGLU hidden size {n_cols} is too large for one Triton block "
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


def is_swiglu_supported(gate: torch.Tensor, up: torch.Tensor) -> bool:
    """Return whether the tensors should use the Triton SwiGLU path."""

    return (
        triton is not None
        and gate.is_cuda
        and up.is_cuda
        and gate.shape == up.shape
        and gate.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and up.dtype == gate.dtype
    )


if triton is not None:

    @triton.jit
    def _silu(x):
        return x * tl.sigmoid(x)

    @triton.jit
    def _swiglu_forward_kernel(
        gate_ptr,
        gate_row_stride,
        up_ptr,
        up_row_stride,
        out_ptr,
        out_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute `silu(gate) * up` for one row per Triton program."""

        row_idx = tl.program_id(0).to(tl.int64)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        gate_row = tl.load(
            gate_ptr + row_idx * gate_row_stride + col_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up_row = tl.load(
            up_ptr + row_idx * up_row_stride + col_offsets,
            mask=mask,
            other=0.0,
        )

        out_row = _silu(gate_row).cast(up_row.dtype) * up_row
        tl.store(out_ptr + row_idx * out_row_stride + col_offsets, out_row, mask=mask)

    @triton.jit
    def _swiglu_backward_kernel(
        dout_ptr,
        dout_row_stride,
        gate_ptr,
        gate_row_stride,
        up_ptr,
        up_row_stride,
        dgate_ptr,
        dgate_row_stride,
        dup_ptr,
        dup_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute gradients for `out = silu(gate) * up`.

        The sigmoid is recomputed from `gate` instead of saved from forward,
        matching Liger's memory-saving strategy.
        """

        row_idx = tl.program_id(0).to(tl.int64)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        dout_row = tl.load(
            dout_ptr + row_idx * dout_row_stride + col_offsets,
            mask=mask,
            other=0.0,
        )
        gate_row = tl.load(
            gate_ptr + row_idx * gate_row_stride + col_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up_row = tl.load(
            up_ptr + row_idx * up_row_stride + col_offsets,
            mask=mask,
            other=0.0,
        )

        sig_gate = tl.sigmoid(gate_row)
        silu_gate = gate_row * sig_gate
        dup_row = dout_row * silu_gate
        dgate_row = dout_row * up_row * (silu_gate * (1.0 - sig_gate) + sig_gate)

        tl.store(
            dgate_ptr + row_idx * dgate_row_stride + col_offsets,
            dgate_row,
            mask=mask,
        )
        tl.store(dup_ptr + row_idx * dup_row_stride + col_offsets, dup_row, mask=mask)


def swiglu_forward(gate: torch.Tensor, up: torch.Tensor):
    """Flatten inputs, choose kernel settings, and launch forward."""

    if not is_swiglu_supported(gate, up):
        raise RuntimeError("Triton SwiGLU only supports matching CUDA fp32/fp16/bf16 tensors.")

    shape = gate.shape
    hidden_size = shape[-1]
    gate_2d = gate.reshape(-1, hidden_size)
    up_2d = up.reshape(-1, hidden_size)
    n_rows, n_cols = gate_2d.shape
    block_size, num_warps = _calculate_settings(n_cols)

    out = torch.empty_like(gate_2d)
    _swiglu_forward_kernel[(n_rows,)](
        gate_2d,
        gate_2d.stride(0),
        up_2d,
        up_2d.stride(0),
        out,
        out.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out.view(*shape), gate_2d, up_2d, block_size, num_warps


def swiglu_backward(
    dout: torch.Tensor,
    gate_2d: torch.Tensor,
    up_2d: torch.Tensor,
    block_size: int,
    num_warps: int,
):
    """Launch backward and return `dgate, dup`."""

    shape = dout.shape
    hidden_size = shape[-1]
    dout_2d = dout.reshape(-1, hidden_size)
    n_rows, n_cols = dout_2d.shape

    dgate = torch.empty_like(gate_2d)
    dup = torch.empty_like(up_2d)
    _swiglu_backward_kernel[(n_rows,)](
        dout_2d,
        dout_2d.stride(0),
        gate_2d,
        gate_2d.stride(0),
        up_2d,
        up_2d.stride(0),
        dgate,
        dgate.stride(0),
        dup,
        dup.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return dgate.view(*shape), dup.view(*shape)


class MiniTrainSwiGLUFunction(torch.autograd.Function):
    """Autograd bridge around the Triton SwiGLU launchers."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor):
        gate = gate.contiguous()
        up = up.contiguous()
        out, gate_2d, up_2d, block_size, num_warps = swiglu_forward(gate, up)
        ctx.save_for_backward(gate_2d, up_2d)
        ctx.block_size = block_size
        ctx.num_warps = num_warps
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout: torch.Tensor):
        gate_2d, up_2d = ctx.saved_tensors
        dout = dout.contiguous()
        dgate, dup = swiglu_backward(
            dout,
            gate_2d,
            up_2d,
            ctx.block_size,
            ctx.num_warps,
        )
        return dgate, dup


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Public Triton SwiGLU entry point used by `TritonOpsBackend`."""

    gate, up = cast_cuda_autocast_activations(gate, up)
    return MiniTrainSwiGLUFunction.apply(gate, up)
