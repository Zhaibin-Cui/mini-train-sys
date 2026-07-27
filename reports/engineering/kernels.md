# ⚡ Kernel engineering on RTX 4090

> **Headline:** eight Transformer/MoE operator families were optimized with Triton plus an
> upstream-FlashAttention-derived native CUDA path. On industrial project shapes, isolated
> full-step speedups are **1.22–6.51×**, with up to **94.0%** lower peak allocation.

![Kernel benchmark overview](../../results/benchmarks/operator_benchmark/resume_summary/kernel_benchmark_overview.png)

## 1. Question and benchmark contract

The goal is not to win on tiny synthetic tensors. Each candidate must pass Torch parity for
forward, backward, dtype, strided/boundary cases, and fallback behavior before performance is
reported. Every shape runs in a fresh CUDA subprocess; measurements include warmup,
`cudaDeviceSynchronize`, P50/P95 full-step latency, and peak allocated memory.

The primary workload matches the formal 293.49M MoE: 57,344 tokens/rank, hidden 768,
intermediate 1,024, head dimension 64, vocabulary 50,257, 8 experts, top-2 routing, BF16.
Explicit CE baselines OOM at the full vocabulary shape, so their comparative headline uses the
largest common safe 8,192-token point. Fused linear CE additionally proves capacity at the full
57,344-token shape.

## 2. Results

| Operator | Implementation | Primary goal | Full-step speedup | Peak allocation reduction |
|---|---|---|---:|---:|
| RMSNorm | Triton | speed + intermediates | **6.51×** | **78.7%** |
| RoPE | Triton | speed | **2.57×** | 14.3% |
| SwiGLU | Triton | fusion | **1.42×** | 20.0% |
| CrossEntropy | Triton | reduction + memory | **4.54×** | **83.3%** |
| FusedLinearCrossEntropy | Triton | activation memory | **1.39×** | **94.0%** |
| FlashAttention | Triton | attention speed + memory | **1.22×** | **44.1%** |
| FlashAttention | native CUDA | attention speed + memory | **1.15×** | 22.1% |
| Router postprocess | Triton | remove multi-pass routing | **2.43×** | **65.2%** |
| Fused MoE | Triton | expert dispatch/GEMM fusion | **1.46×** | 5.3% |

The speedup denominator is always the shape-matched `TorchOpsBackend`, not a CPU or toy
implementation. Raw JSON/CSV and selected-shape identities are in
[`kernel_benchmark_summary.json`](../../results/benchmarks/operator_benchmark/resume_summary/kernel_benchmark_summary.json).

## 3. How each operator works

### RMSNorm — row-parallel fused reduction

Each Triton program owns one row, or a small block of rows when that increases occupancy. It
loads one hidden vector, accumulates the sum of squares in FP32, computes reciprocal RMS, and
applies the learned scale before one store. Backward computes `dx` in the same row-parallel
layout and emits partial `dw` reductions that Python combines.

The gain comes from collapsing square, mean, reciprocal square root, normalization, and scale
into one memory pass. Torch must schedule several generic operations and retain/read more
intermediate state. This is why RMSNorm improves both speed and allocation, with speed being the
largest isolated win.

### RoPE — token/head tiled rotation

Programs parallelize across batch×sequence rows and vectorize over padded head and half-head
dimensions. Q and K rotations share the same loaded cosine/sine values. A separate strided path
preserves correctness without forcing callers to materialize a contiguous copy.

RoPE is bandwidth and launch bound rather than GEMM bound. Fusing the paired-coordinate rotation
and handling Q/K together removes small elementwise kernels; memory savings are modest because
the output itself still has to be written.

### SwiGLU — fuse SiLU and gate multiplication

One program processes a hidden row: it loads gate/up tiles, evaluates SiLU in registers, multiplies
by `up`, and writes one result. Backward produces gate/up gradients in one pass.

The optimization eliminates the materialized `silu(gate)` tensor and associated launch/read. It is
a balanced fusion improvement rather than a change in asymptotic complexity.

### CrossEntropy — online softmax by vocabulary block

Each token row streams the vocabulary in fixed-size blocks. An online max/sum recurrence computes
log-sum-exp without storing a probability matrix. A second streamed pass writes
`softmax - one_hot` into a private logits buffer for backward, while ignored targets and mean
normalization are handled in-kernel.

This reduces repeated full-vocabulary traffic and avoids a separate softmax output, explaining
the simultaneous **4.54×** speedup and **83.3%** allocation reduction.

### Fused Linear CrossEntropy — bounded logits workspace

Materializing `[tokens, vocab]` logits is the dominant LM-head activation. The fused operator
chunks token rows according to a configurable workspace budget (64 MiB by default), performs
`x @ Wᵀ`, immediately feeds each chunk to online CE, and consumes the in-place logits gradient
to accumulate `dx` and `dW`. No full-sequence logits tensor survives.

This operator is deliberately memory-first: **94.0%** lower peak allocation at the common
benchmark point. At the full 57,344-token formal shape it remains runnable with a 356 MiB peak
allocation delta, where the explicit Torch comparison cannot complete.

### FlashAttention — tiled online softmax, not quadratic materialization

The Triton kernel tiles query rows (`BLOCK_M`) against key/value columns (`BLOCK_N`), maintains
running row max/normalizer, and accumulates output in registers. Backward uses fixed Q/KV tiles
and recomputation instead of saving an `N×N` attention matrix. Causal masking and head-dimension
specialization are compile-time paths.

The native CUDA backend wraps upstream FlashAttention launch templates/kernels and compiles only
the required dtype, causal mode, SM target, and head-dimension buckets. It is not eight custom
handwritten CUDA kernels; native CUDA is used specifically where lower-level control is useful.

#### Why beating Torch here is meaningful

The Torch baseline calls `torch.nn.functional.scaled_dot_product_attention`, and profiler evidence
shows dispatch to PyTorch's fused CUDA Flash-SDPA forward and backward—not eager quadratic
attention. Therefore Triton **1.22×** and native CUDA **1.15×** are improvements over an already
optimized official flash backend.

Both sides already avoid the quadratic attention matrix. The remaining advantage is
shape-specific specialization, launch/workspace behavior, and saved-tensor choices on this exact
BF16 causal D=64 workload. The measured allocation reductions are real, but the current evidence
does not attribute every byte or microsecond to one source; that would require a dedicated
liveness/profiler decomposition.

Profiler dispatch evidence:
[`torch_attention_backend.json`](../../results/benchmarks/operator_benchmark/resume_summary/torch_attention_backend.json).

### Router postprocess — one fused rows×experts pass

Programs own blocks of token rows and vectorize across the expert dimension. The forward kernel
combines FP32 softmax, iterative top-k selection, selected-weight normalization, probability
means, z-loss, and entropy statistics. Backward reconstructs the softmax Jacobian contribution
without launching a chain of generic kernels.

Router tensors are small per token but touched repeatedly in a conventional graph. Fusion removes
those rereads and intermediate arrays, producing **2.43×** speedup and **65.2%** lower allocation.

### Fused MoE — expert-tiled scatter, GEMMs, and gather

The MoE path first builds routing metadata with histogram and prefix-sum kernels, then scatters
top-k token routes into contiguous expert tiles. Triton programs execute expert-specific
gate/up projection, fused SwiGLU, down projection, and weighted token gather. Backward mirrors
the tiled layout for input and expert-weight gradients.

Parallelism is two-dimensional: independent expert token tiles form the M dimension, while
intermediate/hidden columns form N. Grouping routes turns irregular token dispatch into dense
expert GEMM tiles and avoids large repeated expert-selection tensors. Its primary value is
throughput (**1.46×**); the selected formal shape already uses relatively compact Torch storage,
so isolated peak-memory reduction is only 5.3%.

## 4. Correctness, fallback, and limitations

- FP32/BF16/FP16 support and unsupported-shape fallback are explicit per operator.
- FP32 is retained for normalization, router logits, softmax, and loss reductions.
- Performance is one 4×RTX 4090 PCIe server, not a claim about all GPUs or model dimensions.
- CUDA facade operators other than attention may fall back to Triton/Torch; reports record actual
  dispatch and never label the whole stack as handwritten CUDA.
- Benchmark speed is not correctness evidence; the test suite and Torch-vs-Triton gradient checks
  are separate release gates.

## 5. Reproduction and evidence

- Executed dense notebook:
  [`operator_bench_linux_server_executed_20260725_162544.ipynb`](../../results/notebooks/operator_bench_linux_server_executed_20260725_162544.ipynb)
- Executed MoE notebook:
  [`moe_operator_bench_linux_server_executed_20260725_163846.ipynb`](../../results/notebooks/moe_operator_bench_linux_server_executed_20260725_163846.ipynb)
- Aggregate JSON/CSV/figure:
  [`results/benchmarks/operator_benchmark/resume_summary/`](../../results/benchmarks/operator_benchmark/resume_summary/)
- Implementation:
  [`minitrain/kernels/triton/`](../../minitrain/kernels/triton/) and
  [`minitrain/kernels/cuda_ext/`](../../minitrain/kernels/cuda_ext/)
