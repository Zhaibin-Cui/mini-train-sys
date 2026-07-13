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
        """Normalize the last dimension and apply a learned scale."""
        ...

    def rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to query and key tensors."""
        ...

    def swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Return the elementwise SwiGLU activation `silu(gate) * up`."""
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
        """Compute scaled dot-product attention for projected Q/K/V tensors."""
        ...

    def cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute language-model cross entropy from materialized logits."""
        ...

    def fused_linear_cross_entropy(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute `linear(x, weight)` and cross entropy as one logical op."""
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
