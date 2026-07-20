# RTX 4090 benchmark and validation summary

Snapshot date: 2026-07-21 (Asia/Shanghai). Hardware: 4 × NVIDIA GeForce RTX 4090 24 GB; PyTorch
2.5.1+cu118; CUDA Toolkit 11.8; Triton 3.1.0; BF16.

## Formal SynBioS model

The exact model has 293,494,272 parameters, sequence length 512, 12 transformer layers, eight MoE
experts, and top-2 routing. Capacity cases include forward, fused loss, backward, and AdamW update.

| FSDP local batch | Global batch (4 GPU) | Peak allocated | Throughput | Result |
|---:|---:|---:|---:|---|
| 96 | 384 | 74.56% | 365,034 tok/s | pass |
| 112 | 448 | 86.20% | 370,857 tok/s | pass; selected |
| 120 | 480 | 92.02% | 281,926 tok/s | pass; slower and less margin |
| 128 | 512 | — | — | expected backward OOM |

Batch 112 subsequently completed 55 consecutive full steps at 86.20% peak allocated memory and
368,170 tok/s. It is the persisted formal local batch.

## FSDP weak scaling at local batch 64

| GPUs | Mean throughput | Weak-scaling efficiency |
|---:|---:|---:|
| 1 | 93,302 tok/s | baseline |
| 4 | 344,254 tok/s | 92.24% |

Both four-GPU repeats passed (92.08% and 92.40%); data stall was 0.09–0.12%.

## Generic 125M-class server matrix

The notebook benchmark covered single-device, DDP, and FSDP configurations on the installed
1/4-GPU topology. The extended capacity sweep completed 30 cases with 10 expected OOM boundary
cases. At local batch 32, DDP-4 reached 248,200 tok/s at 92.65% peak allocated memory and FSDP-4
reached 278,760 tok/s at 66.18%. These generic results were used for infrastructure validation;
the exact SynBioS benchmark above is authoritative for formal training.

## Correctness and recovery gates

- Full repository regression after checkpoint/logging optimization: 70 passed, 0 failed.
- Ruff: passed.
- Exact real-data FSDP save/restore advanced step 3 to step 5 with model, AdamW, scheduler, counters,
  and RNG restored.
- LR continuity: step 3 `6.542e-05`, resumed step 4 `8.723e-05`, step 5 `1.090e-04`.
- Formal data: 100,000 accepted synthetic biographies and 7,405,102 tokens; all recorded manifest
  SHA256 values were revalidated.
- Optimized recovery checkpoint: epoch 10/step 320, 3,677,964,267 bytes, atomically committed in
  26.929 seconds without the redundant 1.3 GB probe export.

Raw evidence is under `benchmarks/`, `validation/`, and `logs/`; exact commands and stopped/failed
runs are preserved in `../HISTORY.md`.
