# Canonical result index

This file is the compact cross-run index. Narrative interpretation lives in `reports/`; exact
commands and lifecycle live in the retained run records; every pushable artifact is listed in
[`catalog/artifacts.json`](catalog/artifacts.json).

## Environment

| Item | Value |
|---|---|
| GPU | 4 × NVIDIA RTX 4090 24 GB, PCIe `SYS`, no NVLink |
| Software | PyTorch 2.5.1+cu118, CUDA toolkit/runtime 11.8, Triton 3.1.0 |
| Formal model | 12 layers, hidden 768, 8 experts, top-2; 293.49M total / 123.62M active |
| Precision / sequence | BF16 / 512 |

Machine inventory: [`environment/server_environment.json`](environment/server_environment.json).

## Kernel benchmark

Question: how much do shape-specialized Triton/native-CUDA paths improve full forward+backward
steps over the shape-matched Torch backend?

| Operator | Backend | Speedup | Peak allocation reduction |
|---|---|---:|---:|
| RMSNorm | Triton | **6.51×** | **78.7%** |
| RoPE | Triton | **2.57×** | 14.3% |
| SwiGLU | Triton | **1.42×** | 20.0% |
| CrossEntropy | Triton | **4.54×** | **83.3%** |
| FusedLinearCrossEntropy | Triton | **1.39×** | **94.0%** |
| FlashAttention | Triton | **1.22×** | **44.1%** |
| FlashAttention | native CUDA | **1.15×** | 22.1% |
| Router postprocess | Triton | **2.43×** | **65.2%** |
| Fused MoE | Triton | **1.46×** | 5.3% |

Torch attention is already fused Flash-SDPA on the measured BF16 causal D=64 shape. Correctness,
dispatch, shape selection, and implementation details:
[`reports/engineering/kernels.md`](../reports/engineering/kernels.md).

Machine summary:
[`benchmarks/operator_benchmark/resume_summary/kernel_benchmark_summary.json`](benchmarks/operator_benchmark/resume_summary/kernel_benchmark_summary.json).

## FSDP and end-to-end backend benchmark

Question: is 1→4 GPU parallelism efficient, and what do optimized kernels change at equal work
and equal memory?

### Weak scaling

| GPUs | Global batch | Throughput | Scaling | Efficiency |
|---:|---:|---:|---:|---:|
| 1 | 64 | 93,302 tok/s | 1.00× | 100.00% |
| 4 | 256 | **344,254 tok/s** | **3.69×** | **92.24%** |

Machine summary:
[`benchmarks/synbios_moe_fsdp4/weak_b64/weak_summary.json`](benchmarks/synbios_moe_fsdp4/weak_b64/weak_summary.json).

### Equal local batch 24

| Backend | Throughput | vs Torch | Peak allocated |
|---|---:|---:|---:|
| Torch | 147,879 ± 6,891 tok/s | 1.000× | 16,626 MB |
| Triton | 212,247 ± 1,843 tok/s | **1.435×** | 5,361 MB |
| CUDA | **215,368 ± 2,190 tok/s** | **1.456×** | **5,358 MB** |

After subtracting each backend's batch-1 peak, activation allocated growth is 14,981 MB for
Torch, 3,949 MB for Triton, and 3,948 MB for CUDA: **73.64–73.65% lower**.

### Equal 92% reserved-memory policy

| Backend | Best local/global batch | Throughput | vs Torch |
|---|---:|---:|---:|
| Torch | 24 / 96 | 160,149 tok/s | 1.000× |
| Triton | 96 / 384 | 349,965 tok/s | **2.185×** |
| CUDA | 96 / 384 | **352,544 tok/s** | **2.201×** |

Machine summaries:
[`fixed-work comparison`](benchmarks/synbios_backend_fixed/20260725T113500Z/presentation/backend_comparison.json) and
[`capacity comparison`](benchmarks/synbios_backend_capacity/20260725T113500Z/presentation/capacity_backend_comparison.json).
Interpretation: [`reports/engineering/distributed_training.md`](../reports/engineering/distributed_training.md).

## SynBioS pretraining

Question: do matched-budget single and augmented backbones memorize their source corpora before
representational probes are compared?

| Metric | `single` | `multi5_permute` |
|---|---:|---:|
| Biographies | 100,000 | 500,000 |
| Scheduled tokens | 3,963,617,280 | 3,988,389,888 |
| Final loss | 0.193221 | 0.296150 |
| End-to-end throughput | 312,868 tok/s | 328,308 tok/s |
| Strict source-text field recall | **100.0000%** | **99.9915%** |

This is training-text recall, not held-out generalization. Machine summaries:
[`single`](formal_runs/synbios_moe/results/single_cloze_eval/full_100k/summary.json) and
[`multi5`](formal_runs/synbios_moe/results/multi5_permute_cloze_eval/full_500k/summary.json).

## Formal P/Q probes

Question: does five-way rewriting and field permutation change when facts become linearly
readable?

| Endpoint | `single` | `multi5_permute` | Dense `multi5+permute` |
|---|---:|---:|---:|
| P0 first, excluding birth date | 6.76% | **98.63%** | ≈100% |
| Q-first macro | 12.83% | **98.79%** | 99.93% |
| P1 whole macro | 13.92% | **39.66%** | **93.56%** |
| Q-whole macro | 3.18% | **33.15%** | 92.58% |

Dense `bioS multi5+permute` references: P1-whole is the Figure 13(c), rank-2 macro of 96.4,
76.0, 96.0, 99.7, and 99.7 (**93.56%**); Q-whole is the Figure 7 macro (**92.58%**).

The first-token mechanism is reproduced; the dense-paper whole endpoint is not numerically
reproduced on the MoE before a correct first-token branch is entered.

Machine summary:
[`formal_probe_comparison_20260724/summary.json`](formal_runs/synbios_moe/results/formal_probe_comparison_20260724/summary.json).

## Mechanism follow-ups

| Test | Primary metric | Interpretation |
|---|---:|---|
| Company relation fit | \(R^2\) 0.99968 / 0.99997 | P-whole follows exposure to either related field |
| Route branching | 12/12 layer DiD positive; max **0.676** | different `t2` branches early MoE routes |
| Single fresh P-whole + true `t1` | 44.23% → **56.47%** | modest unlock; position dependence remains |
| Multi5 fresh P-whole + true `t1` | 45.28% → **96.38%** | whole value becomes nearly linearly readable |

Main interpretation:
[`README.md#model-interpretability`](../README.md#model-interpretability).

## Artifact completeness

| Evidence | Location |
|---|---|
| All pushable files | [`catalog/artifacts.json`](catalog/artifacts.json) |
| Server-only large payloads | [`catalog/retention.json`](catalog/retention.json) |
| TensorBoard events | [`tensorboard/index.csv`](tensorboard/index.csv) |
| Executed notebooks | [`notebooks/`](notebooks/) |
| Categorized console logs | [`logs/`](logs/) |
| Dataset/cache lineage | [`datasets/synbios_moe/`](datasets/synbios_moe/) |
| Snapshot checksums | [`MANIFEST.sha256`](MANIFEST.sha256) |

## Validity limits

- One 4×RTX 4090 PCIe server and one formal 293M MoE architecture.
- Kernel speedups are selected by a predefined formal/common-safe shape policy.
- Probe-held-out people were seen by the backbone.
- Route analysis is conditioned on bad cases and lacks a matched `single` route run.
- True-`t1` fresh heads use a 4,000-step matched intervention budget; they do not establish the
  absolute optimization ceiling.
