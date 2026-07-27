import torch
import torch.nn.functional as F

from minitrain.model.ops import RouterPostprocessOutput


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
        """RMSNorm reference: compute normalization in fp32, then cast once.

        Keeping both the inverse RMS and normalization multiply in fp32 matches
        the Triton kernel and avoids accumulating an artificial fp16/bf16
        quantization error in the weight gradient. Parameters remain fp32 in
        mixed-precision training, while the public activation contract remains
        ``x.dtype``.
        """
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + eps)
        return (normalized * weight.float()).to(dtype=x.dtype)

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
        # Loss reductions stay fp32 even when callers invoke the backend
        # directly outside autocast. This is the same contract implemented by
        # the fused Triton kernels and used by production mixed precision.
        return F.cross_entropy(logits.float(), targets)

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
        logits = F.linear(x, weight)
        return F.cross_entropy(logits.float(), targets)

    def router_postprocess(
        self,
        logits: torch.Tensor,
        top_k: int,
        *,
        normalize: bool,
    ) -> RouterPostprocessOutput:
        """Readable correctness reference for router-logit processing."""

        if logits.ndim != 2 or logits.dtype != torch.float32:
            raise ValueError("router logits must be a 2D fp32 tensor")
        if logits.shape[0] == 0:
            raise ValueError("router logits must contain at least one token")
        if not 1 <= top_k <= logits.shape[1]:
            raise ValueError(f"top_k must be in [1, {logits.shape[1]}], got {top_k}")

        log_normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
        probabilities = torch.exp(logits - log_normalizer)
        expert_weights, expert_indices = torch.topk(probabilities, top_k, dim=-1)
        if normalize:
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(expert_weights.dtype).eps
            )

        return RouterPostprocessOutput(
            expert_weights=expert_weights,
            expert_indices=expert_indices.to(torch.int32),
            probability_per_expert=probabilities.mean(dim=0),
            z_loss=log_normalizer.square().mean(),
            entropy=-(probabilities * probabilities.clamp_min(1e-9).log())
            .sum(dim=-1)
            .mean()
            .detach(),
        )

    def fused_moe(
        self,
        x: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Reference expert loop used as the correctness oracle."""

        output = torch.zeros_like(x)
        active_experts = torch.unique(top_k_index)
        for expert in active_experts:
            # Keep every Top-K route in the reference path, including routes
            # whose current weight is zero. Triton evaluates the same fixed
            # T*K routing graph, and dropping zero-weight routes here changes
            # the derivative with respect to top_k_weights at zero.
            # route_mask = (top_k_index == expert) & (top_k_weights != 0)
            route_mask = top_k_index == expert
            token_index, top_k_position = torch.where(route_mask)
            if token_index.numel() == 0:
                continue
            current = x[token_index]
            gate, up = F.linear(current, gate_up_proj[expert]).chunk(2, dim=-1)
            current = F.silu(gate) * up
            current = F.linear(current, down_proj[expert])
            current = current * top_k_weights[token_index, top_k_position, None]
            output.index_add_(0, token_index, current.to(output.dtype))
        return output
