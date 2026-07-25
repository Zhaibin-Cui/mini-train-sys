# Industrial kernel benchmark summary

## Question or hypothesis

Do MiniTrain's Triton kernels and native CUDA FlashAttention improve full-step latency and peak
allocated memory over shape-matched Torch implementations on one RTX 4090?

## Exact compared conditions

All paired rows use the same shape, BF16 dtype, GPU, 50 ms warmup, 200 ms measurement window, and
fresh CUDA process per shape. The primary profile is the formal 293.49M SynBioS MoE per-rank
workload: local batch 112, sequence 512, 57,344 tokens, H=768, I=1,024, D=64, vocab=50,257,
E=8, K=2. Mixtral-class scans are retained in raw artifacts as an appendix.

## Run and workload identity

The lifecycle and commands are recorded in `HISTORY.md` entries “RTX 4090 industrial per-kernel
benchmark”, “MoE relative-gradient gate and fused-loss capacity follow-up”, and “Industrial
kernel scan boundary extensions”. This is an operator benchmark and has no train/validation
dataset split.

## Primary metrics

| Kernel | Backend | Shape tokens | Full P50 ms | Speedup | Peak alloc MB | Memory Δ |
|---|---|---:|---:|---:|---:|---:|
| rmsnorm | triton | 57,344 | 1.196 | 6.51× | 252.6 | +78.7% |
| rope | triton | 57,344 | 2.364 | 2.57× | 504.0 | +14.3% |
| swiglu | triton | 57,344 | 1.822 | 1.42× | 448.0 | +20.0% |
| cross_entropy | triton | 8,192 | 3.972 | 4.54× | 786.0 | +83.3% |
| fused_linear_cross_entropy | triton | 8,192 | 23.324 | 1.39× | 284.0 | +94.0% |
| attention | triton | 57,344 | 2.623 | 1.22× | 425.3 | +44.1% |
| attention | cuda | 57,344 | 2.776 | 1.15× | 593.3 | +22.1% |
| router_postprocess | triton | 57,344 | 0.756 | 2.43× | 3.5 | +65.2% |
| fused_moe | triton | 57,344 | 17.411 | 1.45× | 1578.0 | +5.3% |

Explicit CrossEntropy and the Torch fused-loss reference cannot complete the full 57,344-token
vocab=50,257 comparison within 24 GB, so their paired headline uses the predeclared largest
common-backend throughput-optimal shape. Separately, Triton fused loss completed the formal
57,344-token capacity case after correctness passed at 32,768
tokens: full P50 **163.12 ms**, peak allocated delta
**356.1 MiB**.

## Supporting artifacts

- Machine-readable summary: `kernel_benchmark_summary.json`
- Flat table: `kernel_benchmark_summary.csv`
- Overview: `kernel_benchmark_overview.png`
- Dense raw summary: `artifacts/operator_benchmark/rtx4090_24gb/dense/summary.json`
- MoE quality-gated summary: `artifacts/operator_benchmark/rtx4090_24gb/moe_r2_relative_gate/summary.json`
- Boundary extensions: `artifacts/operator_benchmark/rtx4090_24gb/extensions/raw`
- Fused-loss capacity evidence: `artifacts/operator_benchmark/rtx4090_24gb/dense_capacity/fused_linear_cross_entropy_project_formal_57344_triton.json`

## Interpretation

The table reports every optimized result, including slowdowns. Native CUDA is claimed only for
FlashAttention; RMSNorm, RoPE, SwiGLU, both losses, Router, and fused MoE are Triton. A speed result
is not a correctness proof; all selected paired rows passed forward and backward checks.

## Limitations or threats to validity

Peak memory is allocator **allocated delta**, not whole-process reserved VRAM. Fused-loss formal
capacity is not a same-shape speedup because the Torch reference OOMed. BF16 fused-MoE gradients
use both retained elementwise statistics and a scale-aware gate (relative L2 <=1%, cosine
>=0.9999). End-to-end FSDP results are a separate stage.

## Next decision/action

Use these operator results as the kernel-level résumé table. Do not infer whole-training speedup
from them; run the formal four-GPU fixed-workload and fixed-92%-reserved-VRAM comparisons.
