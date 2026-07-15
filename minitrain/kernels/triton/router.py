"""Triton postprocessing for fp32 MoE router logits.

The router projection remains a library GEMM. This module fuses the row-wise
softmax, Top-K selection, selected-weight normalization, and training
statistics that would otherwise reread the full logits tensor several times.
"""

from __future__ import annotations

import torch

from minitrain.kernels.triton.cache import configure_triton_cache
from minitrain.model.ops import RouterPostprocessOutput


configure_triton_cache()

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - environments without Triton.
    triton = None
    tl = None


_MAX_EXPERTS = 1024
_MAX_TOP_K = 8


def is_router_postprocess_supported(logits: torch.Tensor, top_k: int, normalize: bool) -> bool:
    del normalize  # Both normalized and raw full-softmax weights are supported.
    return (
        triton is not None
        and logits.is_cuda
        and logits.ndim == 2
        and logits.dtype == torch.float32
        and logits.shape[0] > 0
        and 1 <= top_k <= min(logits.shape[1], _MAX_TOP_K)
        and logits.shape[1] <= _MAX_EXPERTS
        and not torch.are_deterministic_algorithms_enabled()
    )


def _launch_settings(num_experts: int) -> tuple[int, int, int]:
    if triton is None:
        raise RuntimeError("Triton is not installed. Install mini-train-sys[triton].")
    block_size = triton.next_power_of_2(num_experts)
    if block_size <= 32:
        block_rows = 8
    elif block_size <= 128:
        block_rows = 4
    elif block_size <= 512:
        block_rows = 2
    else:
        block_rows = 1
    num_warps = 4 if block_size * block_rows <= 512 else 8
    return block_size, block_rows, num_warps


if triton is not None:

    @triton.jit
    def _router_forward_kernel(
        logits_ptr,
        weights_ptr,
        indices_ptr,
        probability_mean_ptr,
        z_loss_ptr,
        entropy_ptr,
        num_tokens,
        num_experts,
        TOP_K: tl.constexpr,
        NORMALIZE: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        experts = tl.arange(0, BLOCK_SIZE)
        row_mask = rows < num_tokens
        expert_mask = experts < num_experts
        matrix_mask = row_mask[:, None] & expert_mask[None, :]

        logits = tl.load(
            logits_ptr + rows[:, None] * num_experts + experts[None, :],
            mask=matrix_mask,
            other=-float("inf"),
        ).to(tl.float32)
        # Give padded rows a finite dummy distribution so reductions stay
        # defined; all their contributions are masked before stores/atomics.
        logits = tl.where(
            row_mask[:, None],
            logits,
            tl.where(experts[None, :] == 0, 0.0, -float("inf")),
        )

        row_max = tl.max(logits, axis=1, keep_dims=True)
        numerator = tl.exp(logits - row_max)
        denominator = tl.sum(numerator, axis=1, keep_dims=True)
        probabilities = numerator / denominator
        log_normalizer = row_max + tl.log(denominator)

        candidates = logits
        selected = tl.zeros((BLOCK_ROWS, BLOCK_SIZE), tl.int1)
        for _ in range(0, TOP_K):
            index = tl.argmax(candidates, axis=1, keep_dims=True)
            is_selected = experts[None, :] == index
            selected = selected | is_selected
            candidates = tl.where(is_selected, -float("inf"), candidates)
        selected_mass = tl.sum(tl.where(selected, probabilities, 0.0), axis=1, keep_dims=True)

        candidates = logits
        for rank in range(0, TOP_K):
            index = tl.argmax(candidates, axis=1, keep_dims=True)
            is_selected = experts[None, :] == index
            probability = tl.sum(tl.where(is_selected, probabilities, 0.0), axis=1, keep_dims=True)
            weight = probability / selected_mass if NORMALIZE else probability
            tl.store(
                indices_ptr + rows[:, None] * TOP_K + rank,
                index,
                mask=row_mask[:, None],
            )
            tl.store(
                weights_ptr + rows[:, None] * TOP_K + rank,
                weight,
                mask=row_mask[:, None],
            )
            candidates = tl.where(is_selected, -float("inf"), candidates)

        valid_probabilities = tl.where(row_mask[:, None], probabilities, 0.0)
        probability_partial = tl.sum(valid_probabilities, axis=0) / num_tokens
        tl.atomic_add(
            probability_mean_ptr + experts,
            probability_partial,
            mask=expert_mask,
        )

        log_normalizer_rows = tl.sum(log_normalizer, axis=1)
        z_partial = (
            tl.sum(
                tl.where(row_mask, log_normalizer_rows * log_normalizer_rows, 0.0),
                axis=0,
            )
            / num_tokens
        )
        tl.atomic_add(z_loss_ptr, z_partial)

        log_probabilities = logits - log_normalizer
        entropy_per_row = -tl.sum(
            tl.where(expert_mask[None, :], probabilities * log_probabilities, 0.0),
            axis=1,
        )
        entropy_partial = tl.sum(tl.where(row_mask, entropy_per_row, 0.0), axis=0) / num_tokens
        tl.atomic_add(entropy_ptr, entropy_partial)

    @triton.jit
    def _router_backward_kernel(
        logits_ptr,
        weights_ptr,
        indices_ptr,
        grad_weights_ptr,
        grad_probability_mean_ptr,
        grad_z_loss_ptr,
        grad_logits_ptr,
        num_tokens,
        num_experts,
        TOP_K: tl.constexpr,
        NORMALIZE: tl.constexpr,
        BLOCK_TOP_K: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        experts = tl.arange(0, BLOCK_SIZE)
        row_mask = rows < num_tokens
        expert_mask = experts < num_experts
        matrix_mask = row_mask[:, None] & expert_mask[None, :]

        logits = tl.load(
            logits_ptr + rows[:, None] * num_experts + experts[None, :],
            mask=matrix_mask,
            other=-float("inf"),
        ).to(tl.float32)
        logits = tl.where(
            row_mask[:, None],
            logits,
            tl.where(experts[None, :] == 0, 0.0, -float("inf")),
        )
        row_max = tl.max(logits, axis=1, keep_dims=True)
        numerator = tl.exp(logits - row_max)
        denominator = tl.sum(numerator, axis=1, keep_dims=True)
        probabilities = numerator / denominator
        log_normalizer = row_max + tl.log(denominator)

        probability_grad = tl.load(
            grad_probability_mean_ptr + experts,
            mask=expert_mask,
            other=0.0,
        )[None, :] / num_tokens + tl.zeros((BLOCK_ROWS, BLOCK_SIZE), tl.float32)

        if not NORMALIZE:
            for rank in range(0, TOP_K):
                index = tl.load(
                    indices_ptr + rows[:, None] * TOP_K + rank,
                    mask=row_mask[:, None],
                    other=0,
                )
                grad_weight = tl.load(
                    grad_weights_ptr + rows[:, None] * TOP_K + rank,
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                probability_grad += tl.where(experts[None, :] == index, grad_weight, 0.0)

        probability_dot = tl.sum(probabilities * probability_grad, axis=1, keep_dims=True)
        grad_logits = probabilities * (probability_grad - probability_dot)

        grad_z_loss = tl.load(grad_z_loss_ptr).to(tl.float32)
        grad_logits += grad_z_loss * (2.0 / num_tokens) * log_normalizer * probabilities

        if NORMALIZE:
            ranks = tl.arange(0, BLOCK_TOP_K)
            rank_mask = ranks < TOP_K
            saved_weights = tl.load(
                weights_ptr + rows[:, None] * TOP_K + ranks[None, :],
                mask=row_mask[:, None] & rank_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            selected_grads = tl.load(
                grad_weights_ptr + rows[:, None] * TOP_K + ranks[None, :],
                mask=row_mask[:, None] & rank_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            selected_dot = tl.sum(saved_weights * selected_grads, axis=1, keep_dims=True)
            for rank in range(0, TOP_K):
                index = tl.load(
                    indices_ptr + rows[:, None] * TOP_K + rank,
                    mask=row_mask[:, None],
                    other=0,
                )
                weight = tl.load(
                    weights_ptr + rows[:, None] * TOP_K + rank,
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                grad_weight = tl.load(
                    grad_weights_ptr + rows[:, None] * TOP_K + rank,
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                grad_logits += tl.where(
                    experts[None, :] == index,
                    weight * (grad_weight - selected_dot),
                    0.0,
                )

        tl.store(
            grad_logits_ptr + rows[:, None] * num_experts + experts[None, :],
            grad_logits,
            mask=matrix_mask,
        )


class _RouterPostprocessFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, logits: torch.Tensor, top_k: int, normalize: bool):
        logits = logits.contiguous()
        num_tokens, num_experts = logits.shape
        block_size, block_rows, num_warps = _launch_settings(num_experts)
        grid = (triton.cdiv(num_tokens, block_rows),)

        weights = torch.empty((num_tokens, top_k), device=logits.device, dtype=torch.float32)
        indices = torch.empty((num_tokens, top_k), device=logits.device, dtype=torch.int32)
        probability_mean = torch.zeros(num_experts, device=logits.device, dtype=torch.float32)
        z_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
        entropy = torch.zeros((), device=logits.device, dtype=torch.float32)
        _router_forward_kernel[grid](
            logits,
            weights,
            indices,
            probability_mean,
            z_loss,
            entropy,
            num_tokens,
            num_experts,
            TOP_K=top_k,
            NORMALIZE=normalize,
            BLOCK_ROWS=block_rows,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

        ctx.save_for_backward(logits, weights, indices)
        ctx.top_k = top_k
        ctx.normalize = normalize
        ctx.block_size = block_size
        ctx.block_rows = block_rows
        ctx.num_warps = num_warps
        ctx.mark_non_differentiable(indices, entropy)
        return weights, indices, probability_mean, z_loss, entropy

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx,
        grad_weights,
        _grad_indices,
        grad_probability_mean,
        grad_z_loss,
        _grad_entropy,
    ):
        logits, weights, indices = ctx.saved_tensors
        grad_logits = torch.empty_like(logits)
        grid = (triton.cdiv(logits.shape[0], ctx.block_rows),)
        _router_backward_kernel[grid](
            logits,
            weights,
            indices,
            grad_weights.contiguous(),
            grad_probability_mean.contiguous(),
            grad_z_loss.contiguous(),
            grad_logits,
            logits.shape[0],
            logits.shape[1],
            TOP_K=ctx.top_k,
            NORMALIZE=ctx.normalize,
            BLOCK_TOP_K=triton.next_power_of_2(ctx.top_k),
            BLOCK_ROWS=ctx.block_rows,
            BLOCK_SIZE=ctx.block_size,
            num_warps=ctx.num_warps,
        )
        return grad_logits, None, None


def router_postprocess(
    logits: torch.Tensor, top_k: int, *, normalize: bool
) -> RouterPostprocessOutput:
    """Process router logits without materializing the full probability matrix."""

    if not is_router_postprocess_supported(logits, top_k, normalize):
        raise RuntimeError(
            "Triton router postprocess requires non-empty 2D CUDA fp32 logits, "
            f"1 <= top_k <= {_MAX_TOP_K}, at most {_MAX_EXPERTS} experts, and "
            "non-deterministic execution mode."
        )
    weights, indices, probability_mean, z_loss, entropy = _RouterPostprocessFunction.apply(
        logits, top_k, normalize
    )
    return RouterPostprocessOutput(
        expert_weights=weights,
        expert_indices=indices,
        probability_per_expert=probability_mean,
        z_loss=z_loss,
        entropy=entropy,
    )
