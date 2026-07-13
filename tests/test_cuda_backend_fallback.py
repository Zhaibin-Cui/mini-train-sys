import torch

from minitrain.kernels.cuda_ext import CudaOpsBackend
from minitrain.kernels.torch_ops import TorchOpsBackend
from minitrain.kernels.triton import TritonOpsBackend


def test_cuda_backend_preserves_cuda_triton_torch_inheritance_chain():
    """The class hierarchy is the dispatch policy for every optimized op."""

    assert issubclass(CudaOpsBackend, TritonOpsBackend)
    assert issubclass(TritonOpsBackend, TorchOpsBackend)


def test_cuda_attention_reaches_torch_for_unsupported_cpu_inputs():
    """CPU tensors are rejected by CUDA and Triton, then handled by PyTorch."""

    q = torch.randn(1, 2, 8, 16)
    backend = CudaOpsBackend()

    actual = backend.attention(q, q, q, is_causal=True, dropout_p=0.0)
    expected = TorchOpsBackend().attention(q, q, q, is_causal=True, dropout_p=0.0)

    torch.testing.assert_close(actual, expected)
