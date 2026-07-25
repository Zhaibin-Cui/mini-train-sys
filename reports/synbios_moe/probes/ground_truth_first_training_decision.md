# Ground-truth-`t1` fresh whole-probe launch decision

## Question

Does supplying the correct first attribute token make the remaining whole value linearly readable
from the frozen `multi5_permute` MoE backbone when the fresh P/Q probe exactly restores the
original low-rank input delta and rank?

## Compared conditions

The new head is compared with the existing formal whole probe on the same held-out people.
P uses the biography prefix plus cached true `t1` and reads the inserted token. Q uses
`[EOS, name, true t1, EOS]` and reads the final EOS. Both train a fresh whole-value classifier;
P uses rank 2 and Q rank 16. The original formal whole probe is the no-`t1` baseline.

## Run, checkpoint, and dataset identity

- Backbone: `multi5_permute_fsdp_4gpu`, epoch 108 / step 17,388, 293,494,272 total parameters.
- Data/cache: `artifacts/synbios_moe/multi5_permute/probe_cache`, manifest SHA256
  `acd78360d0daa7cf0d2c557fc9f68f07431bc3063cee1145daa3f14c320a232f`.
- Split: person-held-out probe validation; the people were seen by backbone pretraining.
- Semantic audit:
  `artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/ground_truth_first_semantic_audit/audit.json`.

## Launch metrics and budget

| Item | P | Q |
|---|---:|---:|
| Rank | 2 | 16 |
| Training batch | 128 | 768 |
| Held-out evaluation batch | 3,072 | 3,072 |
| Selected training throughput | 632.52 examples/s | 3,780.62 examples/s |
| Selected peak reserved memory | 42.31% | 36.99% |
| Steps selected by user | 4,000 | 4,000 |
| Sampled training examples | 512,000 | 3,072,000 |
| Original formal whole steps | 12,000 | 12,000 |
| Update budget vs original formal | 33.3% | 33.3% |

The complete P/Q × five-attribute matrix and complete held-out validation are retained. This is
therefore a **pilot-budget full-matrix run**, not an original-formal-budget reproduction.

## Supporting artifacts

- Replicated capacity gate:
  `artifacts/synbios_moe/results/ground_truth_first_batch_benchmark/20260725T094600Z/summary.json`
- P/Q lifecycle and exact recovery gate:
  `artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/ground_truth_first_smoke/20260725T095800Z/`
- Lifecycle and failures: repository-root `HISTORY.md`.

## Interpretation

The batch decision is industrial rather than memory-maximal: it selects measured throughput
knees below the accepted 92% reserved-memory ceiling. The scientific intervention is isolated
from the earlier invalid run: true rather than predicted `t1`, the exact original
`AttributeProbe`, and the original P/Q ranks are all mechanically gated.

## Limitations and threats to validity

P expands each biography into six separate prefix-plus-`t1` examples. Thus 4,000 P updates cover
only about one third of the 1.50M derived training examples and provide fewer position-label
exposures than a 4,000-step original P probe, which reads all six positions in one forward pass.
The selected budget follows the user's pilot-budget decision; weak P convergence must not be
interpreted as evidence that true `t1` is uninformative. If loss curves or held-out results remain
clearly unconverged, the predefined follow-up is a P-only exposure-matched extension, not a change
to the reported pilot result.

## Next decision/action

After the full regression gate passes, launch all ten heads at 4,000 steps, complete full
held-out validation, and generate the original-vs-true-`t1` P heatmaps and Q comparison chart.
