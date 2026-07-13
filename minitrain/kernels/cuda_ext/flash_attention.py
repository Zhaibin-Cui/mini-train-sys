"""PyTorch autograd bridge for MiniTrain's CUDA FlashAttention extension.

The public API matches the Triton implementation and accepts MiniTrain's native
``(batch, heads, sequence, head_dim)`` layout. C++ maps outer strides directly
to upstream FlashAttention parameters, so the bridge does not transpose Q/K/V.
"""

from __future__ import annotations

import struct

import torch

from minitrain.kernels.cuda_ext.build import compiled_dtypes
from minitrain.kernels.cuda_ext.build import compiled_head_dims
from minitrain.kernels.cuda_ext.build import load_cuda_extension


_DTYPE_NAMES = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}
_MAX_HEAD_DIM = 256
_D256_DROPOUT_BWD_SMEM_BYTES = 144 * 1024

# Older PyTorch releases do not expose cudaDevAttrMaxSharedMemoryPerBlockOptin
# through ``get_device_properties``. These are the conservative FA2-relevant
# values used only as a no-load fallback for that API gap. The C++ adapter still
# queries the CUDA runtime directly before every D=256 dropout backward launch.
_KNOWN_OPTIN_SMEM_BYTES = {
    (8, 0): 163 * 1024,  # A100-class Ampere
    (8, 6): 99 * 1024,   # Consumer Ampere
    (8, 7): 163 * 1024,  # Orin-class Ampere
    (8, 9): 99 * 1024,   # Ada
    (9, 0): 227 * 1024,  # Hopper
}


def _canonical_dropout_p(dropout_p: float) -> float | None:
    """Return the exact float32 probability consumed by CUDA, if valid.

    Pybind exposes dropout as a Python/C++ double while ``Flash_fwd_params``
    stores it as float. Canonicalizing before capability dispatch and autograd
    state avoids selecting dropout forward with a value that underflows to zero
    in CUDA, or accepting a value below 1.0 that rounds up to float32 1.0.
    """

    value = float(dropout_p)
    if not (0.0 <= value < 1.0):
        return None
    value_f32 = struct.unpack("f", struct.pack("f", value))[0]
    return value_f32 if 0.0 <= value_f32 < 1.0 else None


def _device_optin_smem_bytes(
    device: torch.device,
    capability: tuple[int, int],
) -> int | None:
    """Read opt-in block shared memory without loading the CUDA extension.

    PyTorch's device-properties surface differs by release. Prefer an exact
    runtime value when available, then use the audited architecture table. An
    unknown architecture deliberately returns ``None`` so dispatch remains
    conservative until that architecture has been validated.
    """

    properties = torch.cuda.get_device_properties(device)
    for attribute in (
        "shared_memory_per_block_optin",
        "max_shared_memory_per_block_optin",
    ):
        value = getattr(properties, attribute, None)
        if value is not None:
            return int(value)
    return _KNOWN_OPTIN_SMEM_BYTES.get(capability)


def _dropout_backward_supported_on_arch(
    head_dim: int,
    dropout_p: float,
    max_optin_smem_bytes: int | None,
) -> bool:
    """Return whether upstream FA2 can launch this dropout backward kernel.

    The upstream D=256 backward launcher needs at least 144 KiB of opt-in
    shared memory when dropout is enabled. Consumer Ampere sm86 and Ada sm89
    expose only about 99 KiB per block, and the upstream launcher intentionally
    has no ``Is_dropout=true`` kernel in that branch. Dimensions 200--256 all
    dispatch through the 256 bucket and therefore share this restriction.

    Keeping the threshold comparison pure makes boundary cases independently
    testable. The C++ boundary checks the actual CUDA attribute again so direct
    pybind calls cannot bypass this conservative dispatch.
    """

    if dropout_p == 0.0 or head_dim <= 192:
        return True
    return (
        max_optin_smem_bytes is not None
        and max_optin_smem_bytes >= _D256_DROPOUT_BWD_SMEM_BYTES
    )


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

    dropout_p_f32 = _canonical_dropout_p(dropout_p)
    if dropout_p_f32 is None:
        return False
    dropout_p = dropout_p_f32
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if q.device != k.device or q.device != v.device:
        return False
    if q.shape != k.shape or q.shape != v.shape:
        return False
    # CUDA launch grids and the upstream BlockInfo contract require at least
    # one batch, head, and sequence row. Reject empty training tensors here so
    # CudaOpsBackend can continue to Triton/PyTorch without loading the JIT
    # extension only to receive the duplicate C++ validation error.
    if any(size <= 0 for size in q.shape[:3]):
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
    # The vendored Ampere-family kernel requires sm80+. Keep this check before
    # extension loading: older CUDA devices should use the normal backend
    # fallback instead of reaching the identical TORCH_CHECK in C++.
    capability = torch.cuda.get_device_capability(q.device)
    if capability[0] < 8:
        return False

    # Reject the upstream D=256 dropout backward hardware hole before loading
    # the extension so dispatch can continue CUDA -> Triton -> PyTorch.
    if dropout_p > 0.0 and head_dim > 192:
        max_optin_smem_bytes = _device_optin_smem_bytes(q.device, capability)
        if not _dropout_backward_supported_on_arch(
            head_dim,
            dropout_p,
            max_optin_smem_bytes,
        ):
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

        dropout_p_f32 = _canonical_dropout_p(dropout_p)
        if dropout_p_f32 is None:
            raise ValueError("dropout_p must remain in [0, 1) after float32 conversion")
        dropout_p = dropout_p_f32

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
        # The C++ boundary normalizes exceptional expanded/stride-0 gradients.
        # Keeping it there also protects direct pybind callers and avoids two
        # independent layout policies in Python and C++.
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

    dropout_p_f32 = _canonical_dropout_p(dropout_p)
    if dropout_p_f32 is None or dropout_p_f32 <= 0.0:
        raise ValueError("The debug dropout mask requires dropout_p > 0.")
    dropout_p = dropout_p_f32
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

    dropout_p_f32 = _canonical_dropout_p(dropout_p)
    if dropout_p_f32 is None or not is_flash_attention_supported(
        q, k, v, dropout_p=dropout_p_f32
    ):
        raise RuntimeError(
            "CUDA FlashAttention requires matching CUDA fp16/bf16 Q/K/V tensors, "
            "shape (batch, heads, sequence, head_dim), contiguous head_dim, a "
            "head_dim multiple of 8 through 256, and a dtype/head bucket included "
            "by the active MINITRAIN_CUDA_BUILD_PROFILE. D>192 dropout also "
            "requires a GPU with at least 144 KiB opt-in shared memory per block."
        )
    return MiniTrainCudaFlashAttentionFunction.apply(
        q,
        k,
        v,
        bool(is_causal),
        dropout_p_f32,
    )
