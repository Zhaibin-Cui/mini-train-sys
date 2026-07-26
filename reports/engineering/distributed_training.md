# 🚀 Distributed and end-to-end training

> **Headline:** 4-GPU FSDP reaches **3.69×** throughput over one GPU
> (**92.24% weak-scaling efficiency**). On the formal 293M MoE, the optimized backend is
> **1.456×** faster at equal batch and **2.201×** faster under the same VRAM budget.

## 1. FSDP scaling

The scaling benchmark holds model, BF16 precision, sequence length 512, local batch 64, and fixed
random token generation constant. One and four RTX 4090 FSDP runs each have two repeats, five
warmup steps, and twenty measured steps.

| GPUs | Global batch | Throughput | Scaling | Efficiency | Peak/GPU |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 93,302 tok/s | 1.00× | 100.00% | 14,940 MB |
| 4 | 256 | **344,254 tok/s** | **3.69×** | **92.24%** | 12,413 MB |

![FSDP weak scaling](../../results/benchmarks/synbios_moe_fsdp4/weak_b64/presentation/weak_overview.png)

Both 4-GPU repeats are stable (92.08% and 92.40%). Data wait is 0.10%, so the residual 7.76%
efficiency loss is dominated by FSDP all-gather/reduce-scatter and synchronization over the
server's PCIe `SYS` topology, not by the DataLoader. This is strong single-node scaling without
NVLink, while still leaving a clear communication ceiling.

## 2. Capacity is optimized for throughput, not maximum occupancy

| Local/global batch | Peak allocated/GPU | Throughput | Decision |
|---:|---:|---:|---|
| 96 / 384 | 74.56% | 365,034 tok/s | pass |
| **112 / 448** | **86.20%** | **370,857 tok/s** | formal choice |
| 120 / 480 | 92.02% | 281,926 tok/s | fits, but 24% slower |
| 128 / 512 | — | — | backward OOM boundary |

The formal configuration selects batch 112 because it is the measured throughput optimum with
headroom, not simply the largest allocation that fits.

## 3. End-to-end backend comparison

All backends train the exact same 293,494,272-parameter, 12-layer, 8-expert top-2 MoE with FSDP4,
BF16, sequence length 512, AdamW, and fixed random token batches.

### Equal work: local batch 24

| Backend | Throughput | vs Torch | Step P50 | Peak allocated/GPU |
|---|---:|---:|---:|---:|
| Torch | 147,879 ± 6,891 tok/s | 1.000× | 334.05 ms | 16,626 MB |
| Triton | 212,247 ± 1,843 tok/s | **1.435×** | 231.72 ms | 5,361 MB |
| CUDA | **215,368 ± 2,190 tok/s** | **1.456×** | **228.44 ms** | **5,358 MB** |

CUDA is 31.61% lower step time than Torch and 1.47% faster than Triton. The latter difference is
small; the dominant improvement comes from the complete optimized stack, especially fused MoE
and loss paths.

### Activation memory with static state removed

Raw peak memory includes shared weights, gradients, optimizer, and FSDP state. The primary
activation estimate subtracts the same backend's batch-1 peak:

| Backend | `allocated(b24) − allocated(b1)` | Per added sample | vs Torch |
|---|---:|---:|---:|
| Torch | 14,981 MB | 651.37 MB | baseline |
| Triton | 3,949 MB | 171.69 MB | **−73.64%** |
| CUDA | 3,948 MB | 171.63 MB | **−73.65%** |

This is an incremental activation estimate rather than an exact tensor-liveness proof, but it
avoids claiming model weights as an activation saving.

### Equal space: 92% peak-reserved VRAM ceiling

| Backend | Selected local/global batch | Peak reserved | Throughput | vs Torch |
|---|---:|---:|---:|---:|
| Torch | 24 / 96 | 85.27% | 160,149 tok/s | 1.000× |
| Triton | 96 / 384 | 81.85% | 349,965 tok/s | **2.185×** |
| CUDA | 96 / 384 | 81.78% | **352,544 tok/s** | **2.201×** |

At the same memory policy, the optimized stack supports 4× local batch and more than doubles
token throughput. Equal-work and equal-space results answer different questions and must not be
collapsed into one speedup.

![Backend capacity comparison](../../results/benchmarks/synbios_backend_capacity/20260725T113500Z/presentation/capacity_backend_comparison.png)

## 4. Is parallelism sufficient?

For the current single-node target, yes:

- 92.24% 1→4 GPU weak-scaling efficiency passes the ≥80% quality gate.
- Compute utilization during formal training averages about 97%.
- Data wait is negligible; no expert is dead and no route is dropped.
- FSDP full-shards model and optimizer state and lowers per-GPU peak relative to one GPU.

It is not “perfect scaling”: 7.76% is lost to PCIe communication/synchronization, and no
cross-node or expert-parallel benchmark exists. The correct claim is efficient single-node
FSDP4, not universal distributed scalability.

## 5. Evidence

- Weak-scaling summary:
  [`weak_summary.json`](../../results/benchmarks/synbios_moe_fsdp4/weak_b64/weak_summary.json)
- Equal-work summary:
  [`backend_comparison.json`](../../results/benchmarks/synbios_backend_fixed/20260725T113500Z/presentation/backend_comparison.json)
- Equal-space summary:
  [`capacity_backend_comparison.json`](../../results/benchmarks/synbios_backend_capacity/20260725T113500Z/presentation/capacity_backend_comparison.json)
- Raw cases, OOM boundaries, repeat aggregates, hardware inventory:
  [`results/benchmarks/`](../../results/benchmarks/)
