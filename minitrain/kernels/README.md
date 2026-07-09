# Kernel Backends

This folder follows the spirit of Liger-Kernel, but keeps the public contract
smaller while the project is young.

- `torch_ops.py`: correctness oracle and baseline.
- `triton/`: one file per Triton candidate kernel.
- `cuda_ext/`: CUDA C++ extensions when Triton is not fine-grained enough.

The Triton backend may also wrap production third-party kernels when that
preserves fidelity better than reimplementing them locally. FlashAttention is
implemented as a local Triton candidate in `triton/flash_attention.py` for
`(batch, heads, seq, head_dim)` tensors, with PyTorch SDPA as the portability
fallback when the Triton path is unavailable or when dropout is requested.

Every optimized kernel should ship with:

- a correctness test against `TorchOpsBackend`;
- a shape sweep benchmark;
- a short note explaining whether the op is memory-bound or compute-bound.
