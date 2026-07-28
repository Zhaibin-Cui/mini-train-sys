# SynBioS P/Q probe batch-capacity decision

## Question

Select independent P/Q training and held-out validation batches for four-GPU task-parallel probe
runs without changing the Allen-Zhu probe definition or exceeding 92% CUDA reserved memory.

## Conditions

- Hardware: 4 × RTX 4090 24 GB; two independent replicas per probe kind.
- Backbone: completed `multi5_permute` FSDP4 checkpoint at epoch 108 / step 17,388.
- Data: validated protocol-v2 `multi5_permute` cache with 500,000 P and 100,000 Q examples,
  person-level train/validation split, and complete class coverage.
- Worst task: `university_whole`; three warmup and ten measured steps per candidate.
- Training measures backbone forward, probe backward, and AdamW; validation is forward-only.

## Decision

| Workload | Batch | Mean examples/s | Maximum reserved memory |
|---|---:|---:|---:|
| P training | 128 | 580.54 | 43.37% |
| Q training | 768 | 3,938.28 | 35.37% |
| P validation | 512 | 1,502.85 | 10.60% |
| Q validation | 6,144 | 10,699.55 | 14.86% |

The expanded summary is formal-ready and none of the selected values is the largest tested
candidate. Its generated `recommended.env` is the authoritative runtime input for smoke, pilot,
and formal probe stages.

## Evidence

- Formal expanded matrix and recommendation:
  `results/benchmarks/synbios_moe/probe_batch_benchmark/multi5_permute/20260723T060600Z/`

The conservative matrix verified paper-default P=50 and Q=200 training batches on both replicas.
The expanded matrix begins at the earlier right boundary, so its
`paper_batch_safe_on_all=false` means the defaults were not repeated in that matrix, not that they
failed.

## Interpretation and limitations

These batches optimize this server's execution throughput; they are not experimental outcomes and
do not alter labels, positions, losses, optimizer, or training steps. The `multi5_permute`
condition is used as the larger/worst data case. Smoke stages must still demonstrate end-to-end
correctness for both `single` and `multi5_permute`, and any change to model, checkpoint, cache,
sequence construction, or GPU type invalidates reuse through the pipeline identity checks.

Smoke, pilot, and formal P/Q training completed for both conditions with the generated
`recommended.env`. The formal steps and ranks are recorded in [formal_protocol.md](formal_protocol.md).

## Ground-truth-`t1` fresh-whole extension (2026-07-25)

The corrected fresh-whole protocol changes P into separate prefix-plus-`t1` sequences and
therefore has its own replicated capacity gate. It must not silently reuse the preceding original
probe measurements.

| Workload | Selected batch | Mean examples/s | Maximum reserved memory |
|---|---:|---:|---:|
| P training | 128 | 632.52 | 42.31% |
| Q training | 768 | 3,780.62 | 36.99% |
| P validation | 3,072 | 1,616.36 | 38.34% |
| Q validation | 3,072 | 10,124.56 | 9.90% |

Both replicas passed. P validation batch 8,192 reached 98.53% reserved and was rejected, so the
accepted recommendation is bracketed rather than a search boundary. Raw evidence is under
`artifacts/synbios_moe/results/ground_truth_first_batch_benchmark/20260725T094600Z/`. Both
conditions use rank 2, 4,000 steps, training batch 128, and validation batch 3,072; the completed
results are linked from this directory's README.
