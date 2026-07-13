from typing import Protocol

import torch


class OpsBackend(Protocol):
    """Small contract between model code and kernel implementations.

    The transformer imports this protocol only. Concrete implementations can use
    plain PyTorch, Triton kernels, or CUDA C++ extensions while preserving the
    same call sites and benchmark harnesses.
    """

    name: str

    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """Reduce in fp32 and return a tensor with ``x.dtype``."""
        ...

    def rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE and return rotated Q/K in the shared activation dtype."""
        ...

    def swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Return ``silu(gate) * up`` in the shared activation dtype."""
        ...

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        dropout_p: float,
    ) -> torch.Tensor:
        """Accumulate attention statistics in fp32 and return ``q.dtype``."""
        ...

    def cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy with an fp32 scalar loss under mixed precision."""
        ...

    def fused_linear_cross_entropy(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute linear plus cross entropy and return an fp32 mixed-precision loss."""
        ...


def get_ops_backend(name: str) -> OpsBackend:
    """Construct an operator backend by name.

    Keep this function intentionally small. More complex setup, such as reading
    configs or validating device support, belongs in `minitrain.runtime.factory`.
    """

    if name == "torch":
        from minitrain.kernels.torch_ops import TorchOpsBackend

        return TorchOpsBackend()
    if name == "triton":
        from minitrain.kernels.triton import TritonOpsBackend

        return TritonOpsBackend()
    if name == "cuda":
        from minitrain.kernels.cuda_ext import CudaOpsBackend

        return CudaOpsBackend()
    raise ValueError(f"Unknown ops backend: {name}")
