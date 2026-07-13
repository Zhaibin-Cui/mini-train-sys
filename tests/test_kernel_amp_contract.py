import pytest
import torch

from minitrain.kernels.amp import cast_cuda_autocast_activations


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _backward_sum(*outputs: torch.Tensor) -> None:
    sum(output.float().sum() for output in outputs).backward()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_amp_helper_casts_only_activations(dtype: torch.dtype) -> None:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bf16")
    activation = torch.randn(8, device="cuda", dtype=torch.float32)

    with torch.autocast("cuda", dtype=dtype):
        (cast_activation,) = cast_cuda_autocast_activations(activation)

    assert cast_activation.dtype is dtype


def test_triton_backend_obeys_bf16_autocast_for_fp32_inputs() -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bf16")
    from minitrain.kernels.triton import TritonOpsBackend

    backend = TritonOpsBackend()
    weight = torch.ones(32, device="cuda", dtype=torch.float32, requires_grad=True)
    x = torch.randn(2, 8, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    up = torch.randn_like(x, requires_grad=True)
    q = torch.randn(2, 2, 8, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    cos = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        norm_out = backend.rmsnorm(x, weight, 1e-5)
        swiglu_out = backend.swiglu(gate, up)
        q_rot, k_rot = backend.rope(q, k, cos, sin)
        attention_out = backend.attention(q_rot, k_rot, v, is_causal=True, dropout_p=0.0)

    assert norm_out.dtype is torch.bfloat16
    assert swiglu_out.dtype is torch.bfloat16
    assert q_rot.dtype is torch.bfloat16
    assert k_rot.dtype is torch.bfloat16
    assert attention_out.dtype is torch.bfloat16

    _backward_sum(norm_out, swiglu_out, attention_out)
    assert weight.grad is not None and weight.grad.dtype is torch.float32
    for tensor in (x, gate, up, q, k, v):
        assert tensor.grad is not None and tensor.grad.dtype is torch.float32


def test_rope_rejects_mixed_activation_and_cache_dtypes() -> None:
    from minitrain.kernels.triton.rope import is_rope_supported

    q = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    cos = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)

    assert not is_rope_supported(q, k, cos, sin)


@pytest.mark.parametrize("backend_name", ["torch", "triton", "cuda"])
def test_cross_entropy_contract_returns_fp32_loss(backend_name: str) -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bf16")
    if backend_name == "torch":
        from minitrain.kernels.torch_ops import TorchOpsBackend

        backend = TorchOpsBackend()
    elif backend_name == "triton":
        from minitrain.kernels.triton import TritonOpsBackend

        backend = TritonOpsBackend()
    else:
        from minitrain.kernels.cuda_ext import CudaOpsBackend

        backend = CudaOpsBackend()

    logits = torch.randn(16, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, 64, (16,), device="cuda")
    activations = torch.randn(16, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(64, 32, device="cuda", dtype=torch.float32, requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = backend.cross_entropy(logits, targets)
        fused_loss = backend.fused_linear_cross_entropy(activations, weight, targets)

    assert loss.dtype is torch.float32
    assert fused_loss.dtype is torch.float32
    (loss + fused_loss).backward()
    for tensor in (logits, activations, weight):
        assert tensor.grad is not None and tensor.grad.dtype is torch.float32


def test_cuda_flash_attention_public_api_obeys_bf16_autocast() -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bf16")
    from minitrain.kernels.cuda_ext.flash_attention import flash_attention
    from minitrain.kernels.cuda_ext.flash_attention import is_flash_attention_supported

    q_bf16 = torch.randn(1, 2, 8, 32, device="cuda", dtype=torch.bfloat16)
    if not is_flash_attention_supported(q_bf16, q_bf16, q_bf16, dropout_p=0.0):
        pytest.skip("active CUDA extension profile does not include bf16 D=32")

    q = q_bf16.float().requires_grad_()
    k = q_bf16.float().requires_grad_()
    v = q_bf16.float().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = flash_attention(q, k, v, is_causal=True, dropout_p=0.0)

    assert out.dtype is torch.bfloat16
    out.float().sum().backward()
    for tensor in (q, k, v):
        assert tensor.grad is not None and tensor.grad.dtype is torch.float32
