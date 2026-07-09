import torch
import pytest

from minitrain.kernels.torch_ops import TorchOpsBackend
from minitrain.kernels.triton import TritonOpsBackend
from minitrain.kernels.triton.flash_attention import flash_attention
from minitrain.kernels.triton.flash_attention import is_flash_attention_supported


def test_flash_attention_support_rejects_cpu_tensors() -> None:
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)

    assert not is_flash_attention_supported(q, k, v, dropout_p=0.0)


def test_triton_attention_falls_back_to_torch_on_cpu() -> None:
    torch.manual_seed(123)
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)

    expected = TorchOpsBackend().attention(q, k, v, is_causal=True, dropout_p=0.0)
    actual = TritonOpsBackend().attention(q, k, v, is_causal=True, dropout_p=0.0)

    torch.testing.assert_close(actual, expected)


def test_flash_attention_direct_rejects_cpu_tensors() -> None:
    torch.manual_seed(123)
    q = torch.randn(2, 4, 8, 16, dtype=torch.float32)
    k = torch.randn(2, 4, 8, 16, dtype=torch.float32)
    v = torch.randn(2, 4, 8, 16, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="Local Triton FlashAttention requires CUDA"):
        flash_attention(q, k, v, is_causal=True, dropout_p=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton FlashAttention")
def test_triton_flash_attention_supports_fp32_cuda() -> None:
    pytest.importorskip("triton")
    torch.manual_seed(123)
    device = torch.device("cuda")
    q = torch.randn(2, 4, 17, 32, device=device, dtype=torch.float32)
    k = torch.randn(2, 4, 17, 32, device=device, dtype=torch.float32)
    v = torch.randn(2, 4, 17, 32, device=device, dtype=torch.float32)

    assert is_flash_attention_supported(q, k, v, dropout_p=0.0)
    expected = TorchOpsBackend().attention(q, k, v, is_causal=True, dropout_p=0.0)
    actual = TritonOpsBackend().attention(q, k, v, is_causal=True, dropout_p=0.0)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton FlashAttention")
def test_triton_flash_attention_matches_sdpa_forward_cuda() -> None:
    pytest.importorskip("triton")
    torch.manual_seed(123)
    device = torch.device("cuda")
    q = torch.randn(2, 4, 17, 32, device=device, dtype=torch.float16)
    k = torch.randn(2, 4, 17, 32, device=device, dtype=torch.float16)
    v = torch.randn(2, 4, 17, 32, device=device, dtype=torch.float16)

    expected = TorchOpsBackend().attention(q, k, v, is_causal=True, dropout_p=0.0)
    actual = TritonOpsBackend().attention(q, k, v, is_causal=True, dropout_p=0.0)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton FlashAttention")
def test_triton_flash_attention_matches_sdpa_backward_cuda() -> None:
    pytest.importorskip("triton")
    torch.manual_seed(123)
    device = torch.device("cuda")
    q = torch.randn(2, 4, 17, 32, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(2, 4, 17, 32, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(2, 4, 17, 32, device=device, dtype=torch.float16, requires_grad=True)
    grad = torch.randn(2, 4, 17, 32, device=device, dtype=torch.float16)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    expected = TorchOpsBackend().attention(q_ref, k_ref, v_ref, is_causal=True, dropout_p=0.0)
    actual = TritonOpsBackend().attention(q, k, v, is_causal=True, dropout_p=0.0)
    expected.backward(grad)
    actual.backward(grad)

    torch.testing.assert_close(q.grad, q_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=5e-2, rtol=5e-2)
