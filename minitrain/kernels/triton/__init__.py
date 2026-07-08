from minitrain.kernels.torch_ops import TorchOpsBackend
from minitrain.kernels.triton.cache import configure_triton_cache
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

        if is_rmsnorm_supported(x):
            return triton_rmsnorm(x, weight, eps)
        return super().rmsnorm(x, weight, eps)

    def swiglu(self, gate, up):
        """Run the Triton SwiGLU when the current device supports it."""

        if is_swiglu_supported(gate, up):
            return triton_swiglu(gate, up)
        return super().swiglu(gate, up)

    def rope(self, q, k, cos, sin):
        """Run the Triton RoPE when the current device supports it."""

        if is_rope_supported(q, k, cos, sin):
            return triton_rope(q, k, cos, sin)
        return super().rope(q, k, cos, sin)
