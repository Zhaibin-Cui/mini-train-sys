from minitrain.kernels.amp import cast_cuda_autocast_activations
from minitrain.kernels.torch_ops import TorchOpsBackend
from minitrain.kernels.triton.cache import configure_triton_cache
from minitrain.kernels.triton.flash_attention import flash_attention as triton_flash_attention
from minitrain.kernels.triton.flash_attention import is_flash_attention_supported
from minitrain.kernels.triton.rmsnorm import is_rmsnorm_supported
from minitrain.kernels.triton.rmsnorm import rmsnorm as triton_rmsnorm
from minitrain.kernels.triton.rope import is_rope_supported
# from minitrain.kernels.triton.rope import rope as triton_rope
from minitrain.kernels.triton.rope import rope_strided as triton_rope
from minitrain.kernels.triton.swiglu import is_swiglu_supported
from minitrain.kernels.triton.swiglu import swiglu as triton_swiglu


configure_triton_cache()


class TritonOpsBackend(TorchOpsBackend):
    """Triton backend facade.

    Start by replacing one method at a time with kernels from this package.
    Until a method is replaced, it falls back to the PyTorch implementation so
    the model and trainer remain runnable.
    """

    name = "triton"

    def rmsnorm(self, x, weight, eps):
        """Run the Triton RMSNorm when the current device supports it.

        The backend still inherits the PyTorch implementation as a portability
        fallback. That keeps CPU smoke tests and future non-Triton devices
        usable while the optimized kernel matrix grows one architecture at a
        time.
        """

        (x,) = cast_cuda_autocast_activations(x)
        if is_rmsnorm_supported(x, weight):
            return triton_rmsnorm(x, weight, eps)
        return super().rmsnorm(x, weight, eps)

    def swiglu(self, gate, up):
        """Run the Triton SwiGLU when the current device supports it."""

        gate, up = cast_cuda_autocast_activations(gate, up)
        if is_swiglu_supported(gate, up):
            return triton_swiglu(gate, up)
        return super().swiglu(gate, up)

    def rope(self, q, k, cos, sin):
        """Run the Triton RoPE when the current device supports it."""

        q, k, cos, sin = cast_cuda_autocast_activations(q, k, cos, sin)
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
        if is_rope_supported(q, k, cos, sin):
            return triton_rope(q, k, cos, sin)
        return super().rope(q, k, cos, sin)

    def attention(self, q, k, v, *, is_causal, dropout_p):
        """Run local Triton FlashAttention when available for the current tensors."""

        q, k, v = cast_cuda_autocast_activations(q, k, v)
        if is_flash_attention_supported(q, k, v, dropout_p=dropout_p):
            return triton_flash_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)
        return super().attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)
