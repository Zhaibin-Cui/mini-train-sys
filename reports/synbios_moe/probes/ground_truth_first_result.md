# Fresh whole-probe result with ground-truth `t1`

## Question

After the correct first attribute token is supplied, can a fresh P whole-value probe read the
remaining value more accurately from the frozen `multi5_permute` MoE?

## Compared conditions

The baseline is the valid formal no-`t1` P-whole probe, which includes its rank-2 low-rank input
delta. It is not the withdrawn no-LoRA experiment. The intervention trains the same
`AttributeProbe` architecture and reads the appended true `t1` after each biography prefix.
Backbone, class mappings, person split, cache, and seed are fixed. Each of the five P heads uses
the user-selected 4,000-step pilot budget, batch 128, and held-out batch 3,072.

## Run, checkpoint, and dataset identity

- Backbone: `multi5_permute_fsdp_4gpu`, epoch 108 / step 17,388.
- Probe cache manifest SHA256:
  `acd78360d0daa7cf0d2c557fc9f68f07431bc3063cee1145daa3f14c320a232f`.
- Evaluation: complete person-held-out probe split; these people were present in backbone
  pretraining, so this measures representation readout rather than unseen-person generalization.
- Run ID/root:
  `ground_truth_first_whole_rank_matched_pilot4000_20260725T100100Z`.
- Lifecycle: the retained run records, 2026-07-25 18:11 entry.

## Primary metrics

| Endpoint | Formal no-`t1` baseline | Fresh whole + true `t1` | Delta |
|---|---:|---:|---:|
| P, all 30 attribute/position cells | 45.28% | 96.38% | +51.09 pp |
| P0, five attributes | 32.59% | 95.62% | +63.03 pp |

## Supporting artifacts

- Git-safe result root:
  `results/formal_runs/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/ground_truth_first_whole_rank_matched_pilot4000_20260725T100100Z/`
- Machine aggregate: `summary.json`; position table: `summary.csv`.
- P heatmap: `figures/ground_truth_first_p_overview.png`.
- Individual P task JSON/PT, recovery states, and logs are retained with the mounted result root.

## Interpretation

Supplying true `t1` and training a fresh P head makes P whole values nearly completely linearly
readable. This shows that the post-`t1` biography context contains much more usable whole-value
information than the original pre-attribute readout exposes. It does not prove that a particular
expert stores the fact.

## Limitations and threats to validity

This is a 4,000-step pilot-budget P matrix, one third of the original 12,000-step whole-probe
updates. P expands each biography into six sequences, and the intervention changes sequence
length/readout coordinates.

## Next decision

Treat the P result as the accepted pilot trend. Any later extension should keep the same P-only
protocol and be reported separately rather than overwriting this pilot.
