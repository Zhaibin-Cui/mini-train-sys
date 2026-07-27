# Fresh whole probes with ground-truth `t1`

## Question or hypothesis

After the correct first attribute token is supplied, can a fresh rank-matched whole-value probe
read the remaining value more accurately from the frozen `single` MoE than the original
formal whole probe without that token?

## Exact compared conditions

The baseline is the original formal whole probe. The intervention trains a fresh
P `AttributeProbe` with the same rank-2 low-rank input delta and reads the inserted true `t1`
after each biography prefix. It uses the same frozen backbone, whole-value class mappings,
person split, seed, and cache. This run uses 4000 optimizer steps per head and P batch
128.

`t1` is taken from the original formal P-first probe cache, converted back to its exact GPT-2
token, and round-trip checked before the frozen-backbone readout. It is not a prediction from the
fresh whole-value head.

## Run/checkpoint and dataset identity

- Backbone checkpoint: `/data/mini-train-sys/artifacts/synbios_moe/checkpoints/synbios_moe_single_fsdp_4gpu/epoch_000540_step_000017280`
- Probe cache: `/data/mini-train-sys/artifacts/synbios_moe/single/probe_cache`
- Probe-cache manifest SHA256: `9dcfd2cff38d6f3d29d7f10c2a3247b634f31395b3cda3755a755c8964ffaf5b`
- Baseline summary: `/data/mini-train-sys/artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/summary/summary.json`
- Evaluation: complete person-held-out probe validation; these people were seen during backbone
  pretraining, so this is representation readout rather than unseen-person generalization.

## Primary metrics

| Endpoint | Formal no-`t1` baseline | Fresh whole probe + true `t1` | Delta |
|---|---:|---:|---:|
| P, all six source positions × five attributes | 44.23% | 56.47% | +12.24 pp |
| P0, five attributes | 3.16% | 15.22% | +12.06 pp |

## Supporting artifacts

- Machine-readable aggregate: `summary.json`
- Position-level table: `summary.csv`
- P formal-baseline-vs-intervention heatmap: `figures/ground_truth_first_p_overview.png`
- Individual task JSON/PT, loss curves, recovery checkpoints, and operation logs are retained in
  this run directory; lifecycle and failed/stopped predecessors are in repository-root
  `HISTORY.md`.

## Interpretation

An increase shows that the complete value is more linearly extractable after true `t1` changes
the context and a matched fresh head is trained. It does not show that the unchanged original
head was causally unlocked, and it does not by itself locate the value in a particular MoE
expert.

## Limitations and threats to validity

This is a user-selected 4,000-step pilot-budget full matrix, one third of the prior 12,000-step
whole-probe updates. P expands each biography into six separate sequences, so its label exposure
is also lower than an original P probe that reads six positions in one forward. Any visibly
unconverged P task is a budget limitation, not evidence that true `t1` contains no useful
information. The intervention also changes sequence length and readout coordinates.

## Next decision/action

Use these complete held-out curves and tables to decide whether the qualitative baseline-vs-true
`t1` contrast is stable. If only P remains optimization-limited, extend P alone with the same
protocol and report the extension separately; do not silently replace this pilot-budget result.
