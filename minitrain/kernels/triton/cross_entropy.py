"""Online-softmax Triton cross entropy for the mini-train backend.

The kernel keeps an in-place gradient memory contract: the forward pass computes
the loss and overwrites a private logits buffer with ``dLoss / dLogits``.  The
autograd node therefore only saves that gradient buffer, rather than the full
softmax output.
"""

from __future__ import annotations

import torch

from minitrain.kernels.amp import cast_cuda_autocast_activations
from minitrain.kernels.triton.cache import configure_triton_cache


configure_triton_cache()

try:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:  # pragma: no cover - NGC Triton layout.
        from triton.language.extra.cuda.libdevice import tanh
except ImportError:  # pragma: no cover - environments without Triton.
    triton = None
    tl = None
    tanh = None


MAX_FUSED_SIZE = 65536 // 2
_element_mul_kernel = None


def is_cross_entropy_supported(logits: torch.Tensor, targets: torch.Tensor) -> bool:
    """Return whether the CUDA kernel can handle these tensors."""

    return (
        triton is not None
        and logits.is_cuda
        and targets.is_cuda
        and logits.device == targets.device
        and logits.ndim == 2
        and targets.ndim == 1
        and logits.shape[0] == targets.shape[0]
        and logits.shape[1] > 0
        and logits.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and targets.dtype == torch.int64
    )


def _num_warps(block_size: int) -> int:
    """Avoid over-subscribing small rows while retaining wide-row throughput."""

    if block_size <= 2048:
        return 4
    if block_size <= 8192:
        return 8
    if block_size <= 16384:
        return 16
    return 32


if triton is not None:
    LOG2_E = tl.constexpr(1.4426950408889634)

    @triton.jit
    def _cross_entropy_kernel(
        x_ptr,
        x_stride,
        y_ptr,
        y_stride,
        loss_ptr,
        loss_stride,
        n_cols,
        n_non_ignore_ptr,
        ignore_index,
        softcap,
        HAS_SOFTCAPPING: tl.constexpr,
        HAS_GRADIENTS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Online-softmax CE kernel specialized for mean reduction.

        Each program owns one token row.  It makes two streaming passes over a
        potentially very large vocabulary: the first computes max/log-sum-exp,
        and the second writes the softmax-minus-one gradient in place.
        """

        row = tl.program_id(0).to(tl.int64)
        y = tl.load(y_ptr + row * y_stride)
        n_non_ignore = tl.load(n_non_ignore_ptr)
        x_ptr += row * x_stride

        if y == ignore_index:
            if HAS_GRADIENTS:
                for start in range(0, n_cols, BLOCK_SIZE):
                    offsets = start + tl.arange(0, BLOCK_SIZE)
                    tl.store(x_ptr + offsets, 0.0, mask=offsets < n_cols)
            # Match PyTorch's NaN for mean reduction when every row is ignored.
            ignored_loss = tl.where(n_non_ignore == 0, float("nan"), 0.0)
            tl.store(loss_ptr + row * loss_stride, ignored_loss)
            return

        x_y = tl.load(x_ptr + y).to(tl.float32)
        if HAS_SOFTCAPPING:
            x_y = softcap * tanh(x_y / softcap)

        # Online softmax over fixed-size vocabulary blocks.
        running_max = float("-inf")
        running_sum = 0.0
        for start in range(0, n_cols, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            values = tl.load(
                x_ptr + offsets,
                mask=offsets < n_cols,
                other=float("-inf"),
            ).to(tl.float32)
            if HAS_SOFTCAPPING:
                values = softcap * tanh(values / softcap)
            block_max = tl.max(values)
            new_max = tl.maximum(running_max, block_max)
            running_sum = running_sum * tl.exp2((running_max - new_max) * LOG2_E)
            running_sum += tl.sum(tl.exp2((values - new_max) * LOG2_E))
            running_max = new_max

        lse = running_max + tl.log(running_sum)

        if HAS_GRADIENTS:
            for start in range(0, n_cols, BLOCK_SIZE):
                offsets = start + tl.arange(0, BLOCK_SIZE)
                values = tl.load(
                    x_ptr + offsets,
                    mask=offsets < n_cols,
                    other=float("-inf"),
                ).to(tl.float32)
                if HAS_SOFTCAPPING:
                    intermediate = tanh(values / softcap)
                    values = softcap * intermediate
                grad = tl.exp2((values - running_max) * LOG2_E) / running_sum
                grad = tl.where(offsets == y, grad - 1.0, grad)
                grad /= n_non_ignore
                if HAS_SOFTCAPPING:
                    grad *= 1.0 - intermediate * intermediate
                tl.store(x_ptr + offsets, grad, mask=offsets < n_cols)

        tl.debug_barrier()
        tl.store(loss_ptr + row * loss_stride, (lse - x_y) / n_non_ignore)

    @triton.jit
    def _element_mul_kernel(
        x_ptr,
        x_stride,
        scalar_ptr,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offsets = tl.arange(0, BLOCK_SIZE)
        values = tl.load(x_ptr + row * x_stride + offsets, mask=offsets < n_cols)
        scalar = tl.load(scalar_ptr)
        tl.store(x_ptr + row * x_stride + offsets, values * scalar, mask=offsets < n_cols)


def cross_entropy_forward(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    softcap: float | None = None,
    needs_gradient: bool | None = None,
    normalization_count: int | torch.Tensor | None = None,
    overwrite_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch forward and return ``(loss, saved_logits_gradient)``."""

    if not is_cross_entropy_supported(logits, targets):
        raise RuntimeError(
            "Triton cross entropy requires compatible 2-D CUDA logits and int64 targets."
        )

    rows, vocab = logits.shape
    if rows == 0:
        raise ValueError("Cross entropy requires at least one token row.")
    if vocab > MAX_FUSED_SIZE:
    # Stream multiple fixed-size blocks; BLOCK_SIZE is intentionally
        # capped to reduce register spilling for very large vocabularies.
        block_size = MAX_FUSED_SIZE
    else:
        block_size = triton.next_power_of_2(vocab)

    if normalization_count is None:
        n_non_ignore = (targets != ignore_index).sum(dtype=torch.int32)
    elif isinstance(normalization_count, torch.Tensor):
        n_non_ignore = normalization_count.to(device=logits.device, dtype=torch.int32)
    else:
        n_non_ignore = torch.tensor(normalization_count, device=logits.device, dtype=torch.int32)

    # The kernel writes gradients over its input. A private contiguous copy keeps
    # that optimization without mutating the tensor supplied by the backend.
    need_gradient = logits.requires_grad if needs_gradient is None else needs_gradient
    if not need_gradient or overwrite_logits:
        gradient = logits.contiguous()
    else:
        gradient = logits.contiguous().clone()
    targets = targets.contiguous()
    losses = torch.empty(rows, dtype=torch.float32, device=logits.device)
    _cross_entropy_kernel[(rows,)](
        gradient,
        gradient.stride(0),
        targets,
        targets.stride(0),
        losses,
        losses.stride(0),
        vocab,
        n_non_ignore,
        ignore_index,
        softcap,
        HAS_SOFTCAPPING=softcap is not None,
        HAS_GRADIENTS=need_gradient,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )
    return losses.sum(), gradient


def cross_entropy_backward(saved_gradient: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """Scale the gradient generated during forward by the upstream scalar."""

    if torch.equal(grad_output, torch.ones((), dtype=grad_output.dtype, device=grad_output.device)):
        return saved_gradient
    rows, vocab = saved_gradient.shape
    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(vocab))
    _element_mul_kernel[(rows,)](
        saved_gradient,
        saved_gradient.stride(0),
        grad_output,
        vocab,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )
    return saved_gradient


class MiniTrainCrossEntropyFunction(torch.autograd.Function):
    """Autograd bridge around the online-softmax CE launchers."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        needs_gradient = ctx.needs_input_grad[0]
        loss, gradient = cross_entropy_forward(logits, targets, needs_gradient=needs_gradient)
        if needs_gradient:
            ctx.save_for_backward(gradient)
        return loss

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        (gradient,) = ctx.saved_tensors
        return cross_entropy_backward(gradient, grad_output), None


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Public entry point consumed by :class:`TritonOpsBackend`."""

    (logits,) = cast_cuda_autocast_activations(logits)
    return MiniTrainCrossEntropyFunction.apply(logits, targets)
