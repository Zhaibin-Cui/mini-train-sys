"""PyTorch autograd bridge for MiniTrain's CUDA FlashAttention extension.

The public API matches the Triton implementation and accepts MiniTrain's native
``(batch, heads, sequence, head_dim)`` layout. C++ maps outer strides directly
to upstream FlashAttention parameters, so the bridge does not transpose Q/K/V.
"""

from __future__ import annotations

import torch

from minitrain.kernels.cuda_ext.build import compiled_dtypes
from minitrain.kernels.cuda_ext.build import compiled_head_dims
from minitrain.kernels.cuda_ext.build import load_cuda_extension


_DTYPE_NAMES = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}
_MAX_HEAD_DIM = 256


def _compiled_bucket(head_dim: int) -> int | None:
    """Return the smallest linked template bucket that can serve ``head_dim``."""

    return next((bucket for bucket in compiled_head_dims() if head_dim <= bucket), None)


def is_flash_attention_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
) -> bool:
    """Return whether the active CUDA build can execute this exact input.

    This predicate must not load or compile the extension. Backend dispatch calls
    it on every attention invocation, including paths that should fall back to
    Triton or PyTorch.
    """

    if not (0.0 <= dropout_p < 1.0):
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if q.device != k.device or q.device != v.device:
        return False
    if q.shape != k.shape or q.shape != v.shape:
        return False
    dtype_name = _DTYPE_NAMES.get(q.dtype)
    if dtype_name is None or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if dtype_name not in compiled_dtypes():
        return False
    head_dim = q.shape[-1]
    if head_dim <= 0 or head_dim > _MAX_HEAD_DIM or head_dim % 8 != 0:
        return False
    if _compiled_bucket(head_dim) is None:
        return False
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return False
    return True


class MiniTrainCudaFlashAttentionFunction(torch.autograd.Function):
    """Connect upstream CUDA forward/backward kernels to PyTorch autograd."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
        dropout_p: float,
    ) -> torch.Tensor:
        """Launch forward and retain only the state required by CUDA backward."""

        # The public predicate already requires contiguous head_dim. Retaining
        # this normalization makes direct Function use robust without forcing a
        # copy for normal projected Q/K/V tensors.
        q, k, v = [tensor if tensor.stride(-1) == 1 else tensor.contiguous() for tensor in (q, k, v)]
        out, lse, rng_state, _ = load_cuda_extension().flash_attn_fwd(
            q,
            k,
            v,
            bool(is_causal),
            float(dropout_p),
            False,
        )

        # LSE replaces the SxS probability matrix. RNG state has two int64
        # values only when dropout is enabled and is empty on the optimized
        # no-dropout branch.
        ctx.save_for_backward(q, k, v, out, lse, rng_state)
        ctx.is_causal = bool(is_causal)
        ctx.dropout_p = float(dropout_p)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        """Launch native CUDA backward; no PyTorch SDPA recomputation remains."""

        q, k, v, out, lse, rng_state = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        dq, dk, dv = load_cuda_extension().flash_attn_bwd(
            dout,
            q,
            k,
            v,
            out,
            lse,
            rng_state,
            ctx.is_causal,
            ctx.dropout_p,
        )
        return dq, dk, dv, None, None


def flash_attention_dropout_mask_for_testing(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    dropout_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CUDA output and its exact dropout keep mask for small tests.

    Production forward never calls this function and never materializes the
    probability tile. Upstream's debug path encodes dropped entries in the sign
    bit; the magnitude is an internal unnormalized softmax value and is ignored.
    """

    if dropout_p <= 0.0:
        raise ValueError("The debug dropout mask requires dropout_p > 0.")
    if not is_flash_attention_supported(q, k, v, dropout_p=dropout_p):
        raise RuntimeError("The active CUDA build does not support these debug-mask inputs.")
    out, _, _, encoded_softmax = load_cuda_extension().flash_attn_fwd(
        q,
        k,
        v,
        bool(is_causal),
        float(dropout_p),
        True,
    )
    seqlen_q, seqlen_k = q.shape[-2], k.shape[-2]
    keep = ~torch.signbit(encoded_softmax[..., :seqlen_q, :seqlen_k])
    return out, keep


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    dropout_p: float,
) -> torch.Tensor:
    """Compute dense attention with the selected CUDA specialization matrix."""

    if not is_flash_attention_supported(q, k, v, dropout_p=dropout_p):
        raise RuntimeError(
            "CUDA FlashAttention requires matching CUDA fp16/bf16 Q/K/V tensors, "
            "shape (batch, heads, sequence, head_dim), contiguous head_dim, a "
            "head_dim multiple of 8 through 256, and a dtype/head bucket included "
            "by the active MINITRAIN_CUDA_BUILD_PROFILE."
        )
    return MiniTrainCudaFlashAttentionFunction.apply(
        q,
        k,
        v,
        bool(is_causal),
        float(dropout_p),
    )
