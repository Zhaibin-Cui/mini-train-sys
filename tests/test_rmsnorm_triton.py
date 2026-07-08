import pytest
import torch

from minitrain.kernels.torch_ops import TorchOpsBackend
from minitrain.kernels.triton import TritonOpsBackend
from minitrain.kernels.triton.cache import configure_triton_cache
from minitrain.kernels.triton.rmsnorm import _use_row_kernel


def test_torch_rmsnorm_keeps_activation_dtype() -> None:
    backend = TorchOpsBackend()
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    weight = torch.ones(4, dtype=torch.float32)

    y = backend.rmsnorm(x, weight, eps=1e-5)

    assert y.dtype == x.dtype


def test_triton_cache_is_project_local(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "triton-cache"
    monkeypatch.setenv("MINITRAIN_TRITON_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)

    configured = configure_triton_cache()

    assert configured == cache_dir.resolve()
    assert configured.exists()
    assert configured.is_dir()


def test_rmsnorm_row_block_routing_matches_liger_policy() -> None:
    assert _use_row_kernel(n_rows=4096 * 8, block_size=512)
    assert _use_row_kernel(n_rows=1024, block_size=256)
    assert _use_row_kernel(n_rows=4096 * 8, block_size=256, row_mode=True)
    assert not _use_row_kernel(n_rows=4096 * 8, block_size=256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton RMSNorm test requires CUDA.")
def test_triton_rmsnorm_matches_torch_forward_backward() -> None:
    pytest.importorskip("triton")

    torch.manual_seed(0)
    torch_backend = TorchOpsBackend()
    triton_backend = TritonOpsBackend()

    x_ref = torch.randn(4, 8, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    w_ref = torch.randn(64, device="cuda", dtype=torch.float32, requires_grad=True)
    x_tri = x_ref.detach().clone().requires_grad_(True)
    w_tri = w_ref.detach().clone().requires_grad_(True)
    upstream = torch.randn_like(x_ref)

    y_ref = torch_backend.rmsnorm(x_ref, w_ref, eps=1e-5)
    y_tri = triton_backend.rmsnorm(x_tri, w_tri, eps=1e-5)
    y_ref.backward(upstream)
    y_tri.backward(upstream)

    torch.testing.assert_close(y_tri, y_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x_tri.grad, x_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(w_tri.grad, w_ref.grad, rtol=1e-4, atol=1e-4)
