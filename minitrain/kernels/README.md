# Kernel Backends

This folder keeps a small backend contract around project-owned kernel paths.

- `torch_ops.py`: correctness oracle and baseline.
- `triton/`: one file per Triton candidate kernel.
- `cuda_ext/`: CUDA C++ extensions when Triton is not fine-grained enough.

The Triton backend may also wrap production third-party kernels when that
preserves fidelity better than reimplementing them locally. FlashAttention is
implemented as a local Triton candidate in `triton/flash_attention.py` for
`(batch, heads, seq, head_dim)` tensors, with PyTorch SDPA as the portability
fallback when the Triton path is unavailable or when dropout is requested.

The loss path includes an online-softmax Triton cross entropy and a
memory-bounded fused linear cross entropy. `MINITRAIN_FUSED_CE_WORKSPACE_MB`
controls the latter's logits chunk budget (64 MiB by default). The MoE path
uses a fused fp32 router postprocess plus grouped-GEMM expert kernels; set
`MINITRAIN_MOE_AUTOTUNE=1` before import to enable the larger tuning search
(the default pins memory-safe configurations).

`triton/router.py` supports up to 1024 experts and Top-K up to 8. It falls back
for CPU, unsupported contracts, missing Triton, or deterministic-algorithm
mode. The projection GEMM and global capacity policy deliberately remain
outside this row-wise kernel. Capacity masking is currently inactive in the
model path; all backends execute the same dropless `T*K` routing graph.

Every optimized kernel should ship with:

- a correctness test against `TorchOpsBackend`;
- a shape sweep benchmark;
- a short note explaining whether the op is memory-bound or compute-bound.

Raw benchmark runs are stored under
`tests/benchmark_results/<gpu>/<operator>/<timestamp>.json`. The general
operators live in `tests/operator_bench.ipynb`; grouped MoE has the focused
`tests/moe_operator_bench.ipynb` sweep.
