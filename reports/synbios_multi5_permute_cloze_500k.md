# SynBioS `multi5_permute` progressive-cloze evaluation

Evaluation date: 2026-07-21 (Asia/Shanghai)

## Conclusion

The augmentation run trained successfully for training-corpus recall. On all 500,000 original
training biographies (100,000 people, five independently rendered and field-permuted biographies
per person), the final checkpoint exactly restored 2,999,746 of 3,000,000 removed fields
(99.991533%). It restored all six fields in 499,813 of 500,000 biographies (99.9626%). No field
generation reached the token limit.

This is strong evidence that the 4-GPU FSDP optimization and augmented-data pipeline worked. It is
not a held-out validation result: every evaluated text was in the pretraining corpus. Compared with
the `single` model's 100% strict training-corpus recall, `multi5_permute` is nearly perfect but not
strictly better on this metric. The intended benefit—robustness to unseen wording, people, or fact
combinations—still requires a person-level held-out experiment.

## Protocol

For each original biography, the evaluator removes the exact BPE-token span of all six real facts.
It then fills the holes from left to right in their source-text order with greedy decoding (at most
16 tokens per field). Each prediction is inserted into the biography before the next field is
predicted. The known right-hand literal is used only to stop generation and is never supplied in
the causal prompt. The primary metric is case-sensitive exact string equality.

The 500,000 rows were split into four contiguous, non-overlapping 125,000-row ranges. The
aggregator verifies that the ranges have neither gaps nor overlap before producing the summary.

## Two worked examples

Both examples below concern person 0 and therefore demonstrate how the five-way augmentation
changes wording and field order while preserving the same underlying profile.

### Example 1 — first rendering of person 0

Source-order cloze:

> The profile states Jonah15 Blair13 Carter36's birthday is `[birth_date]`. In life, He owes early
> roots to `[birth_city]`. According to records, He joined `[company]`. In particular, He received
> mentorship at `[university]`. It is recorded that He was professionally based in
> `[company_city]`. The profile states He developed a foundation in `[major]`.

Progressive greedy fills:

1. birth date: `October 24, 2046` → exact
2. birth city: `Tacoma9, NC` → exact
3. company: `Brooks9 Group` → exact
4. university: `Chicago8 University` → exact
5. company city: `Albany3, GA` → exact
6. major: `Political Science2` → exact

### Example 2 — second rendering of person 0

Source-order cloze:

> In life, Jonah15 Blair13 Carter36 spent early years in `[birth_city]`. He developed a foundation
> in `[major]`. Biographical notes say He worked in `[company_city]`. In particular, He received
> mentorship at `[university]`. Biographical notes say He joined `[company]`. The profile states He
> celebrates a birthday on `[birth_date]`.

Progressive greedy fills:

1. birth city: `Tacoma9, NC` → exact
2. major: `Political Science2` → exact
3. company city: `Albany3, GA` → exact
4. university: `Chicago8 University` → exact
5. company: `Brooks9 Group` → exact
6. birth date: `October 24, 2046` → exact

The two texts ask for the same six values in different orders and phrasings. Earlier generated
answers—not hidden ground truth—remain in context for each later answer.

## Strict and fuzzy scoring

The auxiliary fuzzy score case-folds the strings, collapses whitespace, and computes normalized
Levenshtein similarity:

`1 - edit_distance(prediction, target) / max(len(prediction), len(target))`.

Fuzzy matching can overestimate factual correctness. For example, `Tacoma8, NC` versus
`Tacoma9, NC`, or `October 24, 2045` versus `October 24, 2046`, receives a high similarity even
though the distinguishing fact is wrong. Consequently fuzzy scores are diagnostic only.

At thresholds 0.50, 0.80, and 0.90 the micro accuracies were respectively 99.993433%,
99.991733%, and 99.991633%. The 0.90 criterion credits 3 fields that fail exact match; using it as
the headline would therefore overstate the result. All conclusions here use strict exact accuracy.

## Training conditions and outcome

The model architecture and optimization settings match the validated `single` run: 293.49M total
parameters, 12 layers, hidden size 768, eight experts with top-2 routing, BF16, AdamW, 4-GPU FSDP,
local/global batch 112/448, peak LR `1e-3`, 1,000-step warmup, cosine decay to `1e-4`, and global
grad-norm clipping at 5.0. The augmented corpus contains 37,046,556 tokens. Training ran for 108
epochs, 17,388 optimizer steps, and 3,988,389,888 scheduled tokens.

Total loss fell from 10.948931 to 0.296150 (minimum logged 0.293688). Final LM cross-entropy was
0.285855, final MoE regularization was 0.010295, final grad norm was 0.06513, and there were no
dead experts or dropped routes. End-to-end time was 12,148.31 seconds and average throughput was
328,308 tokens/s. These signals and the cloze result support a healthy, effective training run.

Documents are shuffled through randomized-document packing with a 1,024-document shuffle window.
This preserves each biography as a document but is a bounded/windowed shuffle rather than a single
uniform random permutation of all 500,000 documents.

## Metrics and evidence

| Metric | Result |
|---|---:|
| Biographies / people | 500,000 / 100,000 |
| Evaluated fields | 3,000,000 |
| Strict exact fields | 2,999,746 |
| Strict micro field accuracy | 99.991533% |
| All-six-fields biographies | 499,813 |
| All-six-fields biography accuracy | 99.9626% |
| Mean normalized character similarity | 99.994013% |
| Fuzzy accuracy at 0.50 / 0.80 / 0.90 | 99.993433% / 99.991733% / 99.991633% |
| Unterminated generations | 0 |
| Four-GPU parallel wall time | 2,450.64 s |
| Aggregate throughput | 204.03 biographies/s |

| Field | Exact / 500,000 | Strict accuracy |
|---|---:|---:|
| Birth date | 499,984 | 99.9968% |
| Birth city | 499,983 | 99.9966% |
| University | 499,976 | 99.9952% |
| Major | 499,855 | 99.9710% |
| Company | 499,973 | 99.9946% |
| Company city | 499,975 | 99.9950% |

- Aggregate JSON: `artifacts/synbios_moe/results/multi5_permute_cloze_eval/full_500k/summary.json`
- Shards: `artifacts/synbios_moe/results/multi5_permute_cloze_eval/full_500k/shard_{0,1,2,3}.json`
- Exact checkpoint: `artifacts/synbios_moe/checkpoints/synbios_moe_multi5_permute_fsdp_4gpu/epoch_000108_step_000017388/`
- Training events: `artifacts/synbios_moe/runs/synbios_moe_multi5_permute_fsdp_4gpu/20260721-144408/events.jsonl`
- Full operational history: `HISTORY.md`
