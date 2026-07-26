# 🧠 From memorization to structured readout

> **Result:** the Allen-Zhu first-token augmentation mechanism is reproduced on a 293M MoE,
> while whole-value readability remains weaker than the dense-paper result. Follow-up probes
> support a token-conditioned, path-dependent representation rather than one flat whole-attribute
> vector at the name position.

![Formal study overview](../../results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/figures/formal_study_overview.png)

## 1. Experimental question

Two backbones see exactly the same 100,000 synthetic people and facts:

| Condition | Text/person | Field order | Pretraining budget | Strict training-text recall |
|---|---:|---|---:|---:|
| `single` | 1 | fixed | 3.964B scheduled tokens | **100.0000%** |
| `multi5_permute` | 5 | independently rewritten and permuted | 3.988B | **99.9915%** |

Both are the same 12-layer, hidden-768, 8-expert top-2 MoE (293.49M total / 123.62M
token-active parameters), trained with 4-GPU FSDP and BF16. Since both nearly perfectly recover
their own source biographies, probe differences are about representation/readout—not failure to
memorize the corpus.

### Probe definitions

| Probe | Input/readout | `first` target | `whole` target |
|---|---|---|---|
| P | Biography prefixes P0–P5 | first BPE token of an attribute | complete attribute class |
| Q | `[EOS, full_name, EOS]` | first BPE token | complete attribute class |

Probe validation holds out 50,118 people from probe-head training, but the frozen backbone saw
all people during pretraining. It measures cross-person linear readability, not OOD people.

## 2. The Allen-Zhu pattern: first token succeeds

| Formal endpoint | `single` | `multi5_permute` | Change |
|---|---:|---:|---:|
| P0 first, excluding fixed-first birth date | 6.76% | **98.63%** | +91.87 pp |
| Q-first, six-attribute macro | 12.83% | **98.79%** | +85.96 pp |
| Fixed-order target-position P-first | **99.97%** | 99.87% | −0.10 pp |

![P-first heatmaps](../../results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/figures/formal_p_first_heatmaps.png)

`single` forms a staircase: each fact becomes readable only after its fixed textual location.
After five rewrites and field permutation, nearly every first token is readable immediately from
the name or earliest prefix. This is the project's strongest reproduction of Allen-Zhu & Li
Part 3.1.

## 3. The puzzle: whole values do not follow first tokens

| Whole endpoint, five-attribute macro | `single` | `multi5_permute` | Allen-Zhu multi5+permute |
|---|---:|---:|---:|
| P0 whole | 3.16% | **32.59%** | ≈93.5% |
| Q whole | 3.18% | **33.15%** | **92.58%** |

![P-whole heatmaps](../../results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/figures/formal_p_whole_heatmaps.png)

The multi Q-whole heads fit their probe training split well (74.51–98.18%) but validate at only
8.48–51.04%. Therefore the gap is not simply “the probe did not optimize”; complete values are
not represented by one cross-person-consistent linear direction as strongly as first tokens.

## 4. Working representation model

The observations motivate two **readout graphs**:

```text
single, fixed order

name → attr₁::t₁ → attr₁::t₂ → … → attr₂::t₁ → attr₂::t₂ → …
          └──────── context chain follows biography order ────────┘
```

```text
multi5+permute

name ─┬→ attr₁::t₁ → attr₁::t₂ → …
      ├→ attr₂::t₁ → attr₂::t₂ → …
      └→ attrᵢ::t₁ → attrᵢ::t₂ → …
          independent name-to-attribute entry points
```

Here `::` means “conditioned readout transition,” not a claim that bytes are literally stored in
a linked list. The model predicts:

1. fixed-order `single` should depend strongly on position;
2. augmentation should make each `attrᵢ::t₁` directly reachable from the name;
3. predicting a full value before entering its `t₁` branch can remain hard;
4. after true `t₁` is supplied, a fresh matched head should recover much more whole information;
5. different `t₂` values should produce measurable MoE route branching after a shared `t₁`.

The next three analyses test distinct parts of this model.

## 5. Linked facts: company ↔ company city

In randomly permuted biographies, company/company-city P-whole accuracy follows the probability
that either related field has appeared before the readout:

\[
P(\text{company or company-city seen before }P_j)
=1-\frac{\binom{4}{j}}{\binom{6}{j}}.
\]

| Target | Fitted baseline | Saturation | RMSE | \(R^2\) |
|---|---:|---:|---:|---:|
| Company | 45.06% | 94.59% | **0.311 pp** | **0.99968** |
| Company city | 48.44% | 99.72% | **0.097 pp** | **0.99997** |

![Company pair fit](../../results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/company_pair_position_fit/company_pair_position_fit.png)

The six-position curves are reconstructed within 0.1–0.3 pp. The asymmetry is also data-aware:
company uniquely determines city, while 263 companies map to 200 cities and 36 companies share
New York. This is quantitative evidence that P-whole uses an attribute relation, not merely
absolute position.

## 6. Route branching appears when the second token differs

For 162,044 multi5 `Q-first correct / Q-whole wrong` multi-token cases, samples are paired within
the same attribute and same `t1`. The contrast compares identical-`t2` controls against
different-`t2` branches:

\[
\mathrm{DiD}
= [J(t1)-J(t2)]_{\mathrm{different}\ t2}
- [J(t1)-J(t2)]_{\mathrm{same}\ t2}.
\]

![Route branching by layer](../../results/formal_runs/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/report/figures/route_layer_did_portfolio.png)

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Route DiD | .513 | **.676** | .274 | .258 | .146 | .106 | .054 | .155 | .041 | .065 | .077 | .095 |

All 12 layer aggregates are positive and layers 0–3 are strongest. This matches the prediction
that a shared first token enters a similar route and token identity causes early branching when
the second token differs. It offers a plausible MoE-specific reason why one static name-position
whole classifier trails Allen-Zhu's dense result: information is accessed along token-conditioned
expert trajectories.

This is mechanism evidence, not a proof that experts are literal fact databases. It is conditioned
on bad cases and lacks a matched `single` route run.

## 7. Decisive readout test: train a fresh P-whole head after true `t1`

The clean intervention appends cached ground-truth `t1`, reads its hidden state, and trains a new
rank-matched P-whole head. It keeps the backbone, person split, labels, seed, rank 2, batch 128,
and 4,000-step budget fixed across conditions. The withdrawn no-LoRA/Q extension is not part of
these results.

| Condition | Formal no-`t1`, all 30 cells | Fresh + true `t1` | Gain | P0 after true `t1` |
|---|---:|---:|---:|---:|
| `single` | 44.23% | **56.47%** | +12.24 pp | 15.22% |
| `multi5_permute` | 45.28% | **96.38%** | **+51.09 pp** | **95.62%** |

### Single

![Single true-t1 comparison](../../results/formal_runs/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/diagnostics/ground_truth_first_whole_p_pilot4000_20260726T033800Z/figures/ground_truth_first_p_overview.png)

### Multi5+permute

![Multi true-t1 comparison](../../results/formal_runs/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/ground_truth_first_whole_rank_matched_pilot4000_20260725T100100Z/figures/ground_truth_first_p_overview.png)

The left panel in each figure is only a 100% input-consistency check: the supplied token matches
the cached label. It is not whole-value accuracy. The middle/right panels are the valid formal
no-`t1` baseline and fresh true-`t1` P-whole result.

The augmented model becomes almost completely linearly readable after entering the correct
first-token branch, whereas `single` improves only modestly and retains position dependence.
Together with first-token and route results, this supports the proposed independent
`name → attrᵢ::t₁ → attrᵢ::t₂...` readout structure for augmentation.

## 8. What is established—and what is not

### Supported

- Both backbones memorized their source biographies.
- Augmentation transforms first-token readout from positional to name-linked.
- Whole-value readout is substantially weaker than Allen-Zhu's dense result before `t1`.
- Company/company-city curves quantitatively follow related-field exposure.
- Multi5 bad cases show positive token-conditioned route branching in every layer aggregate.
- A fresh matched P head after true `t1` recovers 96.38% multi5 whole accuracy.

### Not yet established

- Experts are the physical storage location of individual facts.
- Route branching is unique to augmentation; matched `single` route statistics are absent.
- The 4,000-step fresh-head result is the absolute single-condition ceiling.
- Person-held-out probe validation implies generalization to people unseen by the backbone.

The defensible conclusion is:

> **Data augmentation creates independent name-to-attribute entry points. On this MoE, first
> tokens are globally linearly readable, while complete values become readable along
> token-conditioned routes; this reproduces Allen-Zhu's first-token mechanism and explains,
> rather than hides, the remaining dense-vs-MoE whole-value gap.**

## 9. Evidence map

- [Formal P/Q comparison](probes/formal_comparison.md)
- [Company/company-city quantitative fit](probes/company_pair_position_fit.md)
- [Layer-wise route analysis](probes/diagnostics/bad_case_routes.md)
- [Single true-`t1` P-whole](probes/ground_truth_first_single_result.md)
- [Multi5 true-`t1` P-whole](probes/ground_truth_first_result.md)
- [Canonical machine metrics](../../results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/)
