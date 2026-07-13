import os
import math

import pytest
import torch
import torch.nn.functional as F

from minitrain.kernels.cuda_ext.build import compiled_dtypes
from minitrain.kernels.cuda_ext.build import compiled_head_dims
from minitrain.kernels.cuda_ext.flash_attention import flash_attention
from minitrain.kernels.cuda_ext.flash_attention import flash_attention_dropout_mask_for_testing
from minitrain.kernels.cuda_ext.flash_attention import is_flash_attention_supported


# These tests trigger a potentially expensive C++/CUDA build. Normal CPU test
# runs remain fast; CUDA developers and CI opt in after choosing a build profile.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or os.getenv("MINITRAIN_RUN_CUDA_EXT_TESTS") != "1",
    reason=(
        "CUDA FlashAttention tests compile a local extension; set "
        "MINITRAIN_RUN_CUDA_EXT_TESTS=1 on a CUDA/nvcc machine to run them."
    ),
)


_TORCH_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _active_dtypes():
    """Parameterize only dtypes linked by the selected build profile."""

    return [_TORCH_DTYPES[name] for name in compiled_dtypes()]


def _active_head_dim() -> int:
    """Use the smallest exact bucket so minimal CI exercises real dispatch."""

    return min(compiled_head_dims())


@pytest.mark.parametrize("dtype", _active_dtypes())
@pytest.mark.parametrize("is_causal", [False, True])
def test_cuda_flash_attention_forward_backward_match_torch(dtype, is_causal):
    """Validate native CUDA forward and backward against PyTorch SDPA."""

    head_dim = _active_head_dim()
    q = torch.randn(2, 3, 32, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    dout = torch.randn_like(q)

    assert is_flash_attention_supported(q, k, v, dropout_p=0.0)
    actual = flash_attention(q, k, v, is_causal=is_causal, dropout_p=0.0)
    actual.backward(dout)
    actual_grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())

    # Recreate leaves so the reference graph cannot share gradient buffers with
    # the custom autograd Function under test.
    q_ref, k_ref, v_ref = [tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v)]
    expected = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        is_causal=is_causal,
        dropout_p=0.0,
    )
    expected.backward(dout)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    for actual_grad, expected_grad in zip(actual_grads, (q_ref.grad, k_ref.grad, v_ref.grad)):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("is_causal", [False, True])
def test_cuda_flash_attention_dropout_replays_rng_in_backward(is_causal):
    """Validate Philox replay and gradients against the exact CUDA keep mask."""

    dtype = _active_dtypes()[0]
    head_dim = _active_head_dim()
    q = torch.randn(2, 2, 32, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    dout = torch.randn_like(q)

    def run(seed: int):
        torch.cuda.manual_seed(seed)
        for tensor in (q, k, v):
            tensor.grad = None
        out = flash_attention(q, k, v, is_causal=is_causal, dropout_p=0.25)
        out.backward(dout)
        return out.detach().clone(), tuple(tensor.grad.detach().clone() for tensor in (q, k, v))

    out_a, grads_a = run(777)
    out_b, grads_b = run(777)
    out_c, _ = run(778)

    assert torch.equal(out_a, out_b)
    assert all(torch.equal(a, b) for a, b in zip(grads_a, grads_b))
    assert not torch.equal(out_a, out_c)
    assert torch.isfinite(out_a).all()
    assert all(torch.isfinite(grad).all() for grad in grads_a)

    # Reset to the same generator state and ask the upstream debug branch to
    # expose its sign-bit-encoded keep mask. Production forward never allocates
    # this SxS tensor.
    torch.cuda.manual_seed(777)
    debug_out, keep = flash_attention_dropout_mask_for_testing(
        q.detach(),
        k.detach(),
        v.detach(),
        is_causal=is_causal,
        dropout_p=0.25,
    )
    assert torch.equal(out_a, debug_out)

    # Build an explicit fp32 reference with the exact CUDA mask. This checks the
    # dropout convention and all three gradients, not only RNG determinism.
    q_ref, k_ref, v_ref = [tensor.detach().float().requires_grad_(True) for tensor in (q, k, v)]
    scores = q_ref @ k_ref.transpose(-1, -2) / math.sqrt(head_dim)
    if is_causal:
        causal_mask = torch.ones((32, 32), device="cuda", dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
    expected = (torch.softmax(scores, dim=-1) * keep / (1.0 - 0.25)) @ v_ref
    expected.backward(dout.float())

    torch.testing.assert_close(out_a.float(), expected, atol=3e-2, rtol=3e-2)
    for actual_grad, expected_grad in zip(grads_a, (q_ref.grad, k_ref.grad, v_ref.grad)):
        torch.testing.assert_close(actual_grad.float(), expected_grad, atol=3e-2, rtol=3e-2)


def test_cuda_flash_attention_rejects_dtype_missing_from_build():
    """fp32 intentionally falls through to Triton/PyTorch in the FA2 path."""

    q = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.float32)
    assert not is_flash_attention_supported(q, q, q, dropout_p=0.0)
