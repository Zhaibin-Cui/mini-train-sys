import torch
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class TorchOpsBackend:
    """Reference backend implemented with standard PyTorch ops.

    This backend is the correctness oracle for Triton and CUDA extensions. Every
    optimized kernel should match these semantics before its speed numbers are
    trusted.
    """

    name = "torch"

    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """RMSNorm reference: reduce in fp32, then return activation dtype.

        Liger-style RMSNorm computes the reduction in fp32 for stability, but
        the normalized activation is cast back before it leaves the op. Casting
        the weight to the activation dtype keeps the reference backend aligned
        with the Triton kernel when model weights are fp32 and activations are
        bf16/fp16.
        """
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps).to(dtype=x.dtype)
        return x * weight.to(dtype=x.dtype)

    def rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = cos[None, None, :, :].to(dtype=q.dtype)
        sin = sin[None, None, :, :].to(dtype=q.dtype)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)

    def swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * up

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        dropout_p: float,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            dropout_p=dropout_p,
        )

    def cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets)

    def fused_linear_cross_entropy(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Reference implementation that still materializes logits.

        Triton/CUDA versions should keep the same output while avoiding the large
        `[tokens, vocab]` intermediate where possible.
        """
        return F.cross_entropy(F.linear(x, weight), targets)
