"""Triton RoPE implementation.

This port keeps the same backend layering used by the RMSNorm and SwiGLU
kernels:

1. A JIT kernel applies rotary embeddings to Q and K in one launch.
2. Python launchers adapt MiniTrain's `(batch, heads, seq, head_dim)` tensors to
   the contiguous layout consumed by the kernel.
3. `MiniTrainRoPEFunction` bridges the launchers into PyTorch autograd.
4. `rope()` is the function consumed by `TritonOpsBackend`.

The rotation follows the HuggingFace LLaMA convention used by the reference
backend: split the head into two halves and rotate `[x1, x2]`.
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


def is_rope_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    """Return whether the tensors should use the Triton RoPE path."""

    return (
        triton is not None
        and q.is_cuda
        and k.is_cuda
        and cos.is_cuda
        and sin.is_cuda
        and q.ndim == 4
        and k.ndim == 4
        and cos.shape == sin.shape
        and cos.ndim in (2, 3)
        and q.shape[0] == k.shape[0]
        and q.shape[2] == k.shape[2]
        and q.shape[-1] == k.shape[-1]
        and q.shape[-1] % 2 == 0
        and q.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and k.dtype == q.dtype
        and cos.dtype == q.dtype
        and sin.dtype == q.dtype
    )


def _canonicalize_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize cos/sin to `(1 or batch, seq, head_dim)`."""

    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin must have matching shapes, got {cos.shape} and {sin.shape}.")
    if cos.ndim == 2:
        if cos.shape != (seq_len, head_dim):
            raise ValueError(
                f"2-D cos/sin must have shape {(seq_len, head_dim)}, got {tuple(cos.shape)}."
            )
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    elif cos.ndim == 3:
        if cos.shape[1:] != (seq_len, head_dim):
            raise ValueError(
                f"3-D cos/sin must have trailing shape {(seq_len, head_dim)}, "
                f"got {tuple(cos.shape)}."
            )
        if cos.shape[0] not in (1, batch_size):
            raise ValueError(
                f"cos/sin batch dimension must be 1 or {batch_size}, got {cos.shape[0]}."
            )
    else:
        raise ValueError(f"cos/sin must be 2-D or 3-D, got ndim={cos.ndim}.")
    return cos.contiguous(), sin.contiguous()


def _empty_seq_major_heads_tensor(
    *,
    batch_size: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    like: torch.Tensor,
) -> torch.Tensor:
    """Allocate a `(batch, heads, seq, head_dim)` view with seq-major storage."""

    return torch.empty_strided(
        (batch_size, n_heads, seq_len, head_dim),
        (seq_len * n_heads * head_dim, head_dim, n_heads * head_dim, 1),
        device=like.device,
        dtype=like.dtype,
    )


if triton is not None:

    @triton.jit
    def _rope_kernel(
        q_ptr,
        q_row_stride,
        k_ptr,
        k_row_stride,
        cos_ptr,
        cos_batch_stride,
        cos_row_stride,
        sin_ptr,
        sin_batch_stride,
        sin_row_stride,
        seq_len: tl.constexpr,
        cos_batch_size: tl.constexpr,
        n_q_heads: tl.constexpr,
        n_k_heads: tl.constexpr,
        head_dim: tl.constexpr,
        PAD_N_Q_HEADS: tl.constexpr,
        PAD_N_K_HEADS: tl.constexpr,
        PAD_HEAD_DIM: tl.constexpr,
        BACKWARD_PASS: tl.constexpr,
    ):
        """Apply rotary embedding for one `(batch, token)` row per program."""

        row_idx = tl.program_id(0).to(tl.int64)
        batch_idx = row_idx // seq_len
        token_idx = row_idx - batch_idx * seq_len

        q_ptr = q_ptr + row_idx * q_row_stride
        k_ptr = k_ptr + row_idx * k_row_stride

        cos_batch_offset = tl.where(cos_batch_size == 1, 0, batch_idx * cos_batch_stride)
        sin_batch_offset = tl.where(cos_batch_size == 1, 0, batch_idx * sin_batch_stride)
        cos_ptr = cos_ptr + cos_batch_offset + token_idx * cos_row_stride
        sin_ptr = sin_ptr + sin_batch_offset + token_idx * sin_row_stride

        half_offsets = tl.arange(0, PAD_HEAD_DIM // 2)
        half_mask = half_offsets < head_dim // 2
        cos_row = tl.load(cos_ptr + half_offsets, mask=half_mask, other=0.0)
        sin_row = tl.load(sin_ptr + half_offsets, mask=half_mask, other=0.0)

        q_head_offsets = tl.arange(0, PAD_N_Q_HEADS)[:, None] * head_dim
        k_head_offsets = tl.arange(0, PAD_N_K_HEADS)[:, None] * head_dim
        dim_offsets = tl.arange(0, PAD_HEAD_DIM // 2)[None, :]

        q_first_offsets = q_head_offsets + dim_offsets
        k_first_offsets = k_head_offsets + dim_offsets
        q_mask = (tl.arange(0, PAD_N_Q_HEADS)[:, None] < n_q_heads) & (
            dim_offsets < head_dim // 2
        )
        k_mask = (tl.arange(0, PAD_N_K_HEADS)[:, None] < n_k_heads) & (
            dim_offsets < head_dim // 2
        )

        q_first = tl.load(q_ptr + q_first_offsets, mask=q_mask, other=0.0).to(cos_row.dtype)
        k_first = tl.load(k_ptr + k_first_offsets, mask=k_mask, other=0.0).to(cos_row.dtype)

        q_second_offsets = q_first_offsets + head_dim // 2
        k_second_offsets = k_first_offsets + head_dim // 2
        q_second = tl.load(q_ptr + q_second_offsets, mask=q_mask, other=0.0).to(cos_row.dtype)
        k_second = tl.load(k_ptr + k_second_offsets, mask=k_mask, other=0.0).to(cos_row.dtype)

        if not BACKWARD_PASS:
            new_q_first = q_first * cos_row - q_second * sin_row
            new_q_second = q_second * cos_row + q_first * sin_row
            new_k_first = k_first * cos_row - k_second * sin_row
            new_k_second = k_second * cos_row + k_first * sin_row
        else:
            new_q_first = q_first * cos_row + q_second * sin_row
            new_q_second = q_second * cos_row - q_first * sin_row
            new_k_first = k_first * cos_row + k_second * sin_row
            new_k_second = k_second * cos_row - k_first * sin_row

        tl.store(q_ptr + q_first_offsets, new_q_first, mask=q_mask)
        tl.store(q_ptr + q_second_offsets, new_q_second, mask=q_mask)
        tl.store(k_ptr + k_first_offsets, new_k_first, mask=k_mask)
        tl.store(k_ptr + k_second_offsets, new_k_second, mask=k_mask)

    @triton.jit
    def _rope_strided_kernel(
        q_ptr,
        q_batch_stride,
        q_head_stride,
        q_seq_stride,
        q_dim_stride,
        k_ptr,
        k_batch_stride,
        k_head_stride,
        k_seq_stride,
        k_dim_stride,
        q_out_ptr,
        q_out_batch_stride,
        q_out_head_stride,
        q_out_seq_stride,
        q_out_dim_stride,
        k_out_ptr,
        k_out_batch_stride,
        k_out_head_stride,
        k_out_seq_stride,
        k_out_dim_stride,
        cos_ptr,
        cos_batch_stride,
        cos_row_stride,
        sin_ptr,
        sin_batch_stride,
        sin_row_stride,
        seq_len: tl.constexpr,
        cos_batch_size: tl.constexpr,
        n_q_heads: tl.constexpr,
        n_k_heads: tl.constexpr,
        head_dim: tl.constexpr,
        PAD_N_Q_HEADS: tl.constexpr,
        PAD_N_K_HEADS: tl.constexpr,
        PAD_HEAD_DIM: tl.constexpr,
        BACKWARD_PASS: tl.constexpr,
    ):
        """Read Q/K with arbitrary strides and write rotated outputs out-of-place."""

        row_idx = tl.program_id(0).to(tl.int64)
        batch_idx = row_idx // seq_len
        token_idx = row_idx - batch_idx * seq_len

        cos_batch_offset = tl.where(cos_batch_size == 1, 0, batch_idx * cos_batch_stride)
        sin_batch_offset = tl.where(cos_batch_size == 1, 0, batch_idx * sin_batch_stride)
        cos_ptr = cos_ptr + cos_batch_offset + token_idx * cos_row_stride
        sin_ptr = sin_ptr + sin_batch_offset + token_idx * sin_row_stride

        half_dim_ids = tl.arange(0, PAD_HEAD_DIM // 2)
        half_dim_mask = half_dim_ids < head_dim // 2
        cos_row = tl.load(cos_ptr + half_dim_ids, mask=half_dim_mask, other=0.0)
        sin_row = tl.load(sin_ptr + half_dim_ids, mask=half_dim_mask, other=0.0)

        q_head_ids = tl.arange(0, PAD_N_Q_HEADS)[:, None]
        k_head_ids = tl.arange(0, PAD_N_K_HEADS)[:, None]
        dim_ids = tl.arange(0, PAD_HEAD_DIM // 2)[None, :]
        second_dim_ids = dim_ids + head_dim // 2

        q_mask = (q_head_ids < n_q_heads) & (dim_ids < head_dim // 2)
        k_mask = (k_head_ids < n_k_heads) & (dim_ids < head_dim // 2)

        q_base = q_ptr + batch_idx * q_batch_stride + token_idx * q_seq_stride
        k_base = k_ptr + batch_idx * k_batch_stride + token_idx * k_seq_stride
        q_first_offsets = q_head_ids * q_head_stride + dim_ids * q_dim_stride
        q_second_offsets = q_head_ids * q_head_stride + second_dim_ids * q_dim_stride
        k_first_offsets = k_head_ids * k_head_stride + dim_ids * k_dim_stride
        k_second_offsets = k_head_ids * k_head_stride + second_dim_ids * k_dim_stride

        q_first = tl.load(q_base + q_first_offsets, mask=q_mask, other=0.0).to(cos_row.dtype)
        q_second = tl.load(q_base + q_second_offsets, mask=q_mask, other=0.0).to(cos_row.dtype)
        k_first = tl.load(k_base + k_first_offsets, mask=k_mask, other=0.0).to(cos_row.dtype)
        k_second = tl.load(k_base + k_second_offsets, mask=k_mask, other=0.0).to(cos_row.dtype)

        if not BACKWARD_PASS:
            new_q_first = q_first * cos_row - q_second * sin_row
            new_q_second = q_second * cos_row + q_first * sin_row
            new_k_first = k_first * cos_row - k_second * sin_row
            new_k_second = k_second * cos_row + k_first * sin_row
        else:
            new_q_first = q_first * cos_row + q_second * sin_row
            new_q_second = q_second * cos_row - q_first * sin_row
            new_k_first = k_first * cos_row + k_second * sin_row
            new_k_second = k_second * cos_row - k_first * sin_row

        q_out_base = q_out_ptr + batch_idx * q_out_batch_stride + token_idx * q_out_seq_stride
        k_out_base = k_out_ptr + batch_idx * k_out_batch_stride + token_idx * k_out_seq_stride
        q_out_first_offsets = q_head_ids * q_out_head_stride + dim_ids * q_out_dim_stride
        q_out_second_offsets = (
            q_head_ids * q_out_head_stride + second_dim_ids * q_out_dim_stride
        )
        k_out_first_offsets = k_head_ids * k_out_head_stride + dim_ids * k_out_dim_stride
        k_out_second_offsets = (
            k_head_ids * k_out_head_stride + second_dim_ids * k_out_dim_stride
        )

        tl.store(q_out_base + q_out_first_offsets, new_q_first, mask=q_mask)
        tl.store(q_out_base + q_out_second_offsets, new_q_second, mask=q_mask)
        tl.store(k_out_base + k_out_first_offsets, new_k_first, mask=k_mask)
        tl.store(k_out_base + k_out_second_offsets, new_k_second, mask=k_mask)


def _launch_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    backward_pass: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transpose inputs, launch the fused Q/K kernel, and restore layout."""

    if triton is None:
        raise RuntimeError("Triton is not installed. Install mini-train-sys[triton].")

    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"q and k must be 4-D, got {tuple(q.shape)} and {tuple(k.shape)}.")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q and k have incompatible shapes: {tuple(q.shape)} and {tuple(k.shape)}.")
    if q.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {q.shape[-1]}.")

    # The transpose itself is metadata-only: it restores the logical
    # `(batch, seq, heads, head_dim)` view expected by the kernel. The
    # contiguous call is the step that may allocate/copy. With the current fused
    # QKV projection, Q and K usually come from `qkv.chunk(...)` and therefore
    # have gaps in storage, so this often packs them into independent buffers.
    q_seq_major = q.transpose(1, 2).contiguous()
    k_seq_major = k.transpose(1, 2).contiguous()
    batch_size, seq_len, n_q_heads, head_dim = q_seq_major.shape
    n_k_heads = k_seq_major.shape[2]
    cos, sin = _canonicalize_cos_sin(
        cos,
        sin,
        batch_size=batch_size,
        seq_len=seq_len,
        head_dim=head_dim,
    )

    pad_head_dim = triton.next_power_of_2(head_dim)
    pad_n_q_heads = triton.next_power_of_2(n_q_heads)
    pad_n_k_heads = triton.next_power_of_2(n_k_heads)
    n_rows = batch_size * seq_len

    _rope_kernel[(n_rows,)](
        q_seq_major,
        q_seq_major.stride(1),
        k_seq_major,
        k_seq_major.stride(1),
        cos,
        cos.stride(0),
        cos.stride(1),
        sin,
        sin.stride(0),
        sin.stride(1),
        seq_len,
        cos.shape[0],
        n_q_heads,
        n_k_heads,
        head_dim,
        PAD_N_Q_HEADS=pad_n_q_heads,
        PAD_N_K_HEADS=pad_n_k_heads,
        PAD_HEAD_DIM=pad_head_dim,
        BACKWARD_PASS=backward_pass,
    )

    return q_seq_major.transpose(1, 2), k_seq_major.transpose(1, 2), cos, sin


def rope_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the forward RoPE kernel."""

    if not is_rope_supported(q, k, cos, sin):
        raise RuntimeError("Triton RoPE only supports CUDA fp32/fp16/bf16 Q/K and cos/sin tensors.")
    return _launch_rope(q, k, cos, sin, backward_pass=False)


def rope_backward(
    dq: torch.Tensor,
    dk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the backward RoPE kernel."""

    dq, dk, _, _ = _launch_rope(dq, dk, cos, sin, backward_pass=True)
    return dq, dk


def _launch_rope_strided(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    backward_pass: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the out-of-place RoPE kernel without packing inputs first."""

    if triton is None:
        raise RuntimeError("Triton is not installed. Install mini-train-sys[triton].")

    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"q and k must be 4-D, got {tuple(q.shape)} and {tuple(k.shape)}.")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q and k have incompatible shapes: {tuple(q.shape)} and {tuple(k.shape)}.")
    if q.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {q.shape[-1]}.")

    batch_size, n_q_heads, seq_len, head_dim = q.shape
    n_k_heads = k.shape[1]
    cos, sin = _canonicalize_cos_sin(
        cos,
        sin,
        batch_size=batch_size,
        seq_len=seq_len,
        head_dim=head_dim,
    )

    # This experimental path skips that pre-pack. The kernel reads Q/K through
    # their actual incoming strides and writes the rotated values directly into
    # fresh output buffers. `empty_strided` makes those buffers look like
    # `(B, heads, S, D)` to attention while their physical storage is seq-major,
    # i.e. equivalent to packed `(B, S, heads, D)`, which matches the kernel's
    # per-token write pattern.
    #
    # It still allocates output tensors because mutating `qkv.chunk(...)` views
    # in-place is unsafe for autograd. The intended win is removing one global
    # memory read/write pass from the explicit contiguous pack, not eliminating
    # output materialization entirely. Keep this as a benchmarked opt-in path
    # until it wins across the target sequence/head sizes.
    q_out = _empty_seq_major_heads_tensor(
        batch_size=batch_size,
        n_heads=n_q_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        like=q,
    )
    k_out = _empty_seq_major_heads_tensor(
        batch_size=batch_size,
        n_heads=n_k_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        like=k,
    )

    pad_head_dim = triton.next_power_of_2(head_dim)
    pad_n_q_heads = triton.next_power_of_2(n_q_heads)
    pad_n_k_heads = triton.next_power_of_2(n_k_heads)
    n_rows = batch_size * seq_len

    _rope_strided_kernel[(n_rows,)](
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        q_out,
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        q_out.stride(3),
        k_out,
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        k_out.stride(3),
        cos,
        cos.stride(0),
        cos.stride(1),
        sin,
        sin.stride(0),
        sin.stride(1),
        seq_len,
        cos.shape[0],
        n_q_heads,
        n_k_heads,
        head_dim,
        PAD_N_Q_HEADS=pad_n_q_heads,
        PAD_N_K_HEADS=pad_n_k_heads,
        PAD_HEAD_DIM=pad_head_dim,
        BACKWARD_PASS=backward_pass,
    )
    return q_out, k_out, cos, sin


def rope_strided_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the optimized RoPE kernel that reads input strides directly."""

    if not is_rope_supported(q, k, cos, sin):
        raise RuntimeError("Triton RoPE only supports CUDA fp32/fp16/bf16 Q/K and cos/sin tensors.")
    return _launch_rope_strided(q, k, cos, sin, backward_pass=False)


def rope_strided_backward(
    dq: torch.Tensor,
    dk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the optimized backward kernel that reads gradient strides directly."""

    dq, dk, _, _ = _launch_rope_strided(dq, dk, cos, sin, backward_pass=True)
    return dq, dk


class MiniTrainRoPEFunction(torch.autograd.Function):
    """Autograd bridge around the Triton RoPE launchers."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        q_out, k_out, cos, sin = rope_forward(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q_out, k_out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dq: torch.Tensor, dk: torch.Tensor):
        cos, sin = ctx.saved_tensors
        dq, dk = rope_backward(dq.contiguous(), dk.contiguous(), cos, sin)
        return dq, dk, None, None


def rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Public Triton RoPE entry point used by `TritonOpsBackend`."""

    q, k, cos, sin = cast_cuda_autocast_activations(q, k, cos, sin)
    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)
    return MiniTrainRoPEFunction.apply(q, k, cos, sin)


class MiniTrainStridedRoPEFunction(torch.autograd.Function):
    """Autograd bridge around the out-of-place, stride-aware RoPE launchers."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        q_out, k_out, cos, sin = rope_strided_forward(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q_out, k_out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dq: torch.Tensor, dk: torch.Tensor):
        cos, sin = ctx.saved_tensors
        dq, dk = rope_strided_backward(dq, dk, cos, sin)
        return dq, dk, None, None


def rope_strided(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental RoPE entry point that avoids pre-packing Q/K inputs.

    Unlike `rope()`, this path reads the incoming Q/K tensors with their actual
    strides and writes rotated outputs into fresh seq-major buffers. It is meant
    for notebook benchmarking before replacing the default backend path.
    """

    q, k, cos, sin = cast_cuda_autocast_activations(q, k, cos, sin)
    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)
    return MiniTrainStridedRoPEFunction.apply(q, k, cos, sin)
