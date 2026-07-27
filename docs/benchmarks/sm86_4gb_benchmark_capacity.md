# sm86 4 GiB benchmark capacity calibration

## Hardware and usable-memory model

The local device is an NVIDIA GeForce RTX 3050 Laptop GPU (compute capability
8.6). PyTorch reports 4,294,443,008 bytes of physical memory (4095 MiB), while
the clean-machine observations during this calibration ranged from roughly
2.8 to 3.2 GiB free. The missing capacity belongs to Windows/WDDM, the CUDA
context, loaded modules, the desktop, and other GPU clients.

The notebook therefore does not target the 4095 MiB physical number directly.
For each operator, the target is approximately 2.6--3.1 GiB after adding:

1. tensors created by `make_case`, which exist before peak-memory accounting;
2. the measured full forward/backward peak delta;
3. a safety margin for the CUDA context, allocator fragmentation, and WDDM.

This is intentionally a capacity boundary, not an OOM search. A benchmark that
only succeeds when the desktop is idle would not be reproducible enough to
keep as the notebook maximum.

## Calibration result

The final notebooks were executed in place on 2026-07-16. Both have every code
cell executed and contain no error output:

- `tests/operator_bench_sm86_4gb.ipynb`: 20/20 code cells;
- `tests/moe_operator_bench_sm86_4gb.ipynb`: 6/6 code cells.

Representative largest-case measurements are below. `full delta` is the
PyTorch allocated-memory increase measured by the harness; it excludes the
input case that was already alive at the measurement baseline.

| benchmark | largest logical size | largest-case status | full delta |
|---|---:|---|---:|
| RMSNorm | 117,440,512 elements | torch/triton ok | 2688 MiB |
| RoPE | 268,435,456 Q+K elements | torch/triton ok | 1792 MiB |
| SwiGLU | 402,653,184 gate+up elements | torch/triton ok | 1920 MiB |
| Cross entropy | 234,881,024 logits | torch/triton ok | 2240 MiB |
| Fused linear CE | 201,326,592 logical logits | torch/triton ok | 1920 MiB |
| Attention | 301,989,888 Q+K+V elements | torch/cuda ok; triton unsupported | 1548 MiB |
| Router postprocess | 2,097,152 tokens | torch/triton ok | 2160 MiB |
| Fused MoE | 114,688 tokens | torch/triton ok | 2809 MiB |

Attention uses a boundary shape of batch 1536 and sequence length 256. This is
deliberate: increasing sequence length until FlashAttention fills memory makes
compute grow quadratically and turns a memory-capacity test into a many-hour
compute test. Increasing batch holds the kernel specialization meaningful
while memory and work grow linearly.

The Triton attention specialization is unsupported for the boundary shape on
this Windows sm86 environment. That row is retained as `unavailable`, while
the PyTorch and native CUDA implementations pass. Unsupported capability is
not reported as a numerical failure.

## Harness fixes found by the boundary run

- A one-element MoE sweep used `(512*16)` rather than `(512*16,)` and therefore
  passed an integer where an iterable was required.
- Correctness comparison previously materialized whole fp32 copies, absolute
  errors, relative errors, and a close mask. At hundreds of millions of
  elements this could OOM after the operator itself had succeeded. Comparison
  now streams through 4M-element chunks with the same tolerance and max-error
  semantics.
- Nsight registration now carries an explicit provider. CUDA FlashAttention
  profiling no longer accidentally requests an unsupported Triton path.
- The standalone Nsight filter now matches `minitrain_flash::flash_bwd_*`, and
  the display cell accepts both `Kernel Name` and legacy `KernelName` columns.

The provider-control and Nsight compatibility changes are synchronized across
the ordinary, Linux-server, and sm86 notebook variants. Hardware-dependent
sweep sizes remain intentionally different.

## Reproduction

From the repository root, with the CUDA build toolchain available:

```powershell
$env:CUDA_MODULE_LOADING = "LAZY"
jupyter nbconvert --to notebook --execute --inplace tests/moe_operator_bench_sm86_4gb.ipynb --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=python3
jupyter nbconvert --to notebook --execute --inplace tests/operator_bench_sm86_4gb.ipynb --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=python3
```

Raw immutable datasets and Nsight reports are under
`tests/benchmark_results/sm86-nvidia-geforce-rtx-3050-laptop-gpu/`; figures are
under `reports/figures/`.
