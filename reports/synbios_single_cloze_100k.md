# SynBioS `single` progressive cloze conclusion

Evaluation date: 2026-07-21 (Asia/Shanghai)

## Conclusion

The current final model trained effectively for its intended training-corpus memorization task.
On every one of the 100,000 original `single` biographies, it restored all six removed facts
exactly: 600,000/600,000 strict case-sensitive field matches, and 100,000/100,000 biographies
with 6/6 fields correct. Birth date, birth city, university, major, company, and company city were
each 100% exact.

This result establishes exact in-distribution recall of the training biographies. It does **not**
establish generalization to unseen people, unseen fact combinations, or unseen writing templates;
that requires a held-out generation/evaluation split.

## Protocol

For each original biography, the evaluator removes the exact BPE-token span of each real fact,
not a reconstructed or separately generated prompt. It fills holes from left to right with greedy
decoding. A prediction is inserted back into the biography before the next hole is filled, so a
later result sees the model's earlier answers rather than the hidden ground truth. The known
right-hand literal marks the end of a hole but is not supplied in the causal prompt. Strict exact
match is the primary metric.

## Two worked examples

### Example 1 — person 0

Initial original-text cloze (all six source spans removed):

> In life, Jonah15 Blair13 Carter36 entered the world on `[birth_date]`. It is recorded that He
> was born in `[birth_city]`. The profile states He received mentorship at `[university]`.
> Notably, He specialized in `[major]`. Notably, He was employed by `[company]`. Public records
> show He was professionally based in `[company_city]`.

Progressive greedy fills, with each answer retained as context for the next one:

1. birth date: `October 24, 2046` → exact
2. birth city: `Tacoma9, NC` → exact
3. university: `Chicago8 University` → exact
4. major: `Political Science2` → exact
5. company: `Brooks9 Group` → exact
6. company city: `Albany3, GA` → exact

The fully reconstructed biography is byte-for-byte identical to the source text.

### Example 2 — person 1

Initial original-text cloze:

> The profile states Maya16 Morgan9 Dennis47's birthday is `[birth_date]`. Biographical notes say
> They calls `[birth_city]` a birthplace. According to records, They studied at `[university]`.
> Notably, They studied `[major]`. The profile states They had a professional role at `[company]`.
> Historically, They was professionally based in `[company_city]`.

Progressive greedy fills:

1. birth date: `December 7, 2000` → exact
2. birth city: `Oakland4, MA` → exact
3. university: `Miami2 University` → exact
4. major: `Finance2` → exact
5. company: `Hart7 Group` → exact
6. company city: `Tacoma3, NC` → exact

This example also checks the special `… a birthplace.` template: ` a birthplace` is the
right-hand literal, not part of the city answer.

## How fuzzy matching works, and whether it can overestimate

The auxiliary fuzzy score first applies Unicode-safe case folding and collapses repeated
whitespace. It then computes normalized Levenshtein similarity:

`1 - edit_distance(prediction, target) / max(len(prediction), len(target))`.

The report records mean similarity and the fraction of fields at or above 0.50, 0.80, and 0.90.
This can overestimate semantic correctness. A wrong city such as `Tacoma8, NC` versus
`Tacoma9, NC`, or a wrong year such as `October 24, 2045` versus `October 24, 2046`, shares nearly
all characters and receives a high fuzzy score despite the key fact being wrong. Case folding can
also forgive a case error. Therefore fuzzy thresholds are diagnostics for near misses, not the
headline accuracy.

In this run fuzzy scoring does not inflate the conclusion: strict case-sensitive exact accuracy
was already 100%, and every fuzzy score was consequently 1.0.

## Metrics and evidence

| Metric | Result |
|---|---:|
| Biographies | 100,000 |
| Evaluated fields | 600,000 |
| Strict micro field accuracy | 100% |
| All-six-fields biography accuracy | 100% |
| Each individual field accuracy | 100% |
| Mean normalized character similarity | 100% |
| Fuzzy accuracy at 0.50 / 0.80 / 0.90 | 100% / 100% / 100% |
| Unterminated generations | 0 |
| Four-GPU parallel wall time | 419.48 s |

- Aggregate JSON: `artifacts/synbios_moe/results/single_cloze_eval/full_100k/summary.json`
- Shards: `artifacts/synbios_moe/results/single_cloze_eval/full_100k/shard_{0,1,2,3}.json`
- Pilot JSON: `artifacts/synbios_moe/results/single_cloze_eval/pilot_1000.json`
- Exact checkpoint: `artifacts/synbios_moe/checkpoints/synbios_moe_single_fsdp_4gpu/epoch_000540_step_000017280/`
- Full execution history, including rejected pilot/scoring attempts and duplicate-range protection:
  `HISTORY.md`
