"""Memory-efficient fused linear cross entropy.

Instead of materializing logits for every token, this operator partitions the
token dimension.  Each chunk performs the linear projection, invokes the same
online-softmax Triton kernel as :mod:`cross_entropy`, and immediately consumes
the in-place logits gradient to accumulate input and weight gradients.
"""

from __future__ import annotations

import os

import torch

from minitrain.kernels.amp import cast_cuda_autocast_activations
from minitrain.kernels.triton.cross_entropy import MAX_FUSED_SIZE
from minitrain.kernels.triton.cross_entropy import _element_mul_kernel
from minitrain.kernels.triton.cross_entropy import _num_warps
from minitrain.kernels.triton.cross_entropy import cross_entropy_forward

try:
    import triton
except ImportError:  # pragma: no cover - environments without Triton.
    triton = None


def is_fused_linear_cross_entropy_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
) -> bool:
    """Return whether tensors satisfy the local fused-kernel contract."""

    return (
        triton is not None
        and x.is_cuda
        and weight.is_cuda
        and targets.is_cuda
        and x.device == weight.device == targets.device
        and x.ndim == 2
        and weight.ndim == 2
        and targets.ndim == 1
        and x.shape[0] == targets.shape[0]
        and x.shape[1] == weight.shape[1]
        and x.shape[0] > 0
        and weight.shape[0] > 0
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and targets.dtype == torch.int64
    )


def fused_linear_cross_entropy_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Compute loss plus already-materialized input/weight gradients.

    The chunk heuristic chooses enough chunks that a
    chunk's logical ``tokens * vocab`` footprint is on the order of the input
    ``tokens * hidden`` footprint, then round the token count to a power of two.
    """

    if not is_fused_linear_cross_entropy_supported(x, weight, targets):
        raise RuntimeError("Fused Triton CE requires compatible 2-D CUDA input/weight tensors.")

    tokens, hidden = x.shape
    vocab = weight.shape[0]
    valid = targets != ignore_index
    total_non_ignore = valid.sum(dtype=torch.int32)

    # A pure input-footprint heuristic produces very small GEMMs on common
    # language-model shapes.  Use the largest power-of-two chunk fitting a
    # bounded logits workspace instead, improving Tensor Core utilization while
    # retaining a predictable peak-memory ceiling.
    workspace_mb = max(1, int(os.getenv("MINITRAIN_FUSED_CE_WORKSPACE_MB", "64")))
    workspace_bytes = workspace_mb * 1024 * 1024
    rows_by_budget = max(1, workspace_bytes // (vocab * x.element_size()))
    chunk_limit = min(tokens, rows_by_budget)
    chunk_size = 1 << (chunk_limit.bit_length() - 1)
    num_chunks = triton.cdiv(tokens, chunk_size)

    need_grad_x = x.requires_grad
    need_grad_weight = weight.requires_grad
    need_logits_gradient = need_grad_x or need_grad_weight
    grad_x = torch.empty_like(x) if need_grad_x else None
    grad_weight = None
    loss = None

    # Linear under CUDA autocast normally computes in the activation dtype even
    # when master weights are fp32.  Make that contract explicit because custom
    # autograd forward executes with gradient recording disabled.
    matmul_weight = weight if weight.dtype == x.dtype else weight.to(dtype=x.dtype)
    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, tokens)
        x_chunk = x[start:end]
        logits = x_chunk @ matmul_weight.t()
        chunk_loss, grad_logits = cross_entropy_forward(
            logits,
            targets[start:end],
            ignore_index=ignore_index,
            needs_gradient=need_logits_gradient,
            normalization_count=total_non_ignore,
            overwrite_logits=True,
        )
        loss = chunk_loss if loss is None else loss + chunk_loss

        if need_grad_x:
            grad_x[start:end] = grad_logits @ matmul_weight
        if need_grad_weight:
            chunk_grad_weight = (grad_logits.t() @ x_chunk).to(dtype=weight.dtype)
            if grad_weight is None:
                grad_weight = chunk_grad_weight
            else:
                grad_weight.add_(chunk_grad_weight)
    assert loss is not None
    return loss, grad_x, grad_weight


def _scale_2d_gradient(
    gradient: torch.Tensor | None,
    grad_output: torch.Tensor,
) -> torch.Tensor | None:
    if gradient is None:
        return None
    rows, cols = gradient.shape
    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(cols))
    _element_mul_kernel[(rows,)](
        gradient,
        gradient.stride(0),
        grad_output,
        cols,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )
    return gradient


class MiniTrainFusedLinearCrossEntropyFunction(torch.autograd.Function):
    """Autograd bridge that returns gradients computed during forward."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        loss, grad_x, grad_weight = fused_linear_cross_entropy_forward(x, weight, targets)
        ctx.save_for_backward(grad_x, grad_weight)
        return loss

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        grad_x, grad_weight = ctx.saved_tensors
        if torch.equal(
            grad_output,
            torch.ones((), dtype=grad_output.dtype, device=grad_output.device),
        ):
            return grad_x, grad_weight, None
        return (
            _scale_2d_gradient(grad_x, grad_output),
            _scale_2d_gradient(grad_weight, grad_output),
            None,
        )


def fused_linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Public fused-loss entry point consumed by :class:`TritonOpsBackend`."""

    (x,) = cast_cuda_autocast_activations(x)
    return MiniTrainFusedLinearCrossEntropyFunction.apply(x, weight, targets)
