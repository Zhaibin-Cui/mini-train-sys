# Formal P/Q probe training decision

## Decision

Use the capacity benchmark's **throughput-optimal** batches, stop already-saturated first-token
tasks early, and reserve paper-scale exposure for unconverged whole-attribute tasks.

| Probe task | Batch | Steps | Exposure | Equivalent epochs | Rationale |
|---|---:|---:|---:|---:|---|
| P first, single | 128 | 4,000 | 512,000 | 10.24 | Pilot held-out already saturated |
| P first, multi5+permute | 128 | 4,000 | 512,000 | 2.048 | Pilot held-out already saturated |
| Q first, either | 768 | 4,000 | 3,072,000 | 61.44 | Pilot held-out already saturated |
| P whole, single | 128 | 12,000 | 1,536,000 | 30.72 | Paper-scale exposure |
| P whole, multi5+permute | 128 | 12,000 | 1,536,000 | 6.144 | Paper-scale exposure |
| Q whole, either | 768 | 12,000 | 9,216,000 | 184.32 | 4× pilot; held-out safety margin |

Validation remains P=512 and Q=6,144. P uses rank 2 and Q rank 16.

The formal YAML therefore uses a per-kind and target schedule:

```yaml
formal:
  steps:
    p_first: 4000
    p_whole: 12000
    q_first: 4000
    q_whole: 12000
```

## Why this is sufficiently trained

Allen-Zhu and Li use P batch 50 × 30,000 steps and Q batch 200 × 30,000 steps. Those settings
produce 1.5M P and 6.0M Q sampled examples. Whole P matches that exposure closely. Whole Q uses
50% more exposure because the pilot's 46.08 passes left a large held-out gap despite high
probe-train accuracy. It still uses only 40% of the paper's optimizer updates.

The 3,000-step pilot supplies the empirical lower bound:

- Multi5 first-token P and Q already reach 97.1%--99.8% held-out accuracy.
- Whole tasks are still improving and show generalization gaps.
- Formal whole tasks receive 4× the pilot optimizer updates. P whole closely reproduces the
  paper's effective passes over single and multi5; Q whole gets an explicit safety margin.
- Formal first tasks receive only 1.33× pilot steps because all six augmented held-out tasks have
  already saturated.

This is preferable to either extreme: 3,000 steps are insufficient for whole attributes, while
running every first-token task for 12,000 or 30,000 steps would spend substantial time after its
held-out metric has saturated.

## Throughput and expected wall time

The authoritative capacity run recommends:

| Workload | Batch | Measured throughput | Peak reserved memory |
|---|---:|---:|---:|
| P training | 128 | 580.54 examples/s | 43.37% |
| Q training | 768 | 3,938.28 examples/s | 35.37% |
| P validation | 512 | 1,502.85 examples/s | 10.60% |
| Q validation | 6,144 | 10,699.55 examples/s | 14.86% |

Higher-memory candidates were slower; maximizing occupied VRAM would not maximize experimental
throughput. The formal terminal scales each completed pilot task by its own selected step count,
then distinguishes task, phase, and full-pipeline ETA and reports the expected local completion
timestamp.

## Promotion and stop criteria

Launch is allowed only after:

1. both pilot pipelines are `completed`;
2. the pilot trend gate in `pilot_analysis.md` passes;
3. tests cover the per-kind/target schedule and full-pipeline ETA;
4. the existing regression/correctness/checkpoint gates remain satisfied;
5. formal identity records the new P/Q schedule and throughput-optimal batch environment.

During formal execution, a task is considered sufficiently converged when held-out first-token
results preserve the pilot contrast and whole-task results no longer show material improvement
relative to the retained checkpoints/curves. The configured exposure remains the default
completion point; no silent early stopping is introduced.

## Evidence

- Batch decision: `reports/synbios_moe/probes/capacity.md`
- Pilot conclusion: `reports/synbios_moe/probes/pilot_analysis.md`
- Raw capacity matrix:
  `results/benchmarks/synbios_moe/probe_batch_benchmark/multi5_permute/20260723T060600Z/`
- Paper protocol: [Physics of Language Models: Part 3.1](https://arxiv.org/abs/2309.14316),
  Appendices E and F.
