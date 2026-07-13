"""CUDA C++ operator backend for mini-train-sys.

This package mirrors the shape of ``minitrain.kernels.triton``: a small facade
inherits the Triton backend, replaces only the operators that have a local CUDA
implementation, and therefore falls back in the fixed order CUDA -> Triton ->
PyTorch for everything else.
"""

from minitrain.kernels.amp import cast_cuda_autocast_activations
from minitrain.kernels.cuda_ext.flash_attention import (
    flash_attention as cuda_flash_attention,
)
from minitrain.kernels.cuda_ext.flash_attention import (
    is_flash_attention_supported,
)
from minitrain.kernels.triton import TritonOpsBackend


class CudaOpsBackend(TritonOpsBackend):
    """CUDA extension backend facade.

    Keep this class intentionally thin. Kernel-specific validation and build
    loading live next to the kernel itself. Unsupported CUDA inputs call
    ``super()``, which tries Triton and then the PyTorch reference backend.
    """

    name = "cuda"

    def attention(self, q, k, v, *, is_causal, dropout_p):
        """Try CUDA first, then preserve Triton and PyTorch fallbacks."""

        q, k, v = cast_cuda_autocast_activations(q, k, v)
        if is_flash_attention_supported(q, k, v, dropout_p=dropout_p):
            return cuda_flash_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)
        return super().attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)
