# 🔬 SynBioS probe reports

The cross-stage interpretation is centralized in
[`../storage_story.md`](../storage_story.md). This directory retains the formal report,
mechanism analyses, exact protocols, and historical budget decisions.

## Canonical evidence

| Topic | Report | Status |
|---|---|---|
| Single vs multi5 P/Q | [Formal comparison](formal_comparison.md) | **formal, canonical** |
| Company ↔ company-city | [Position fit](company_pair_position_fit.md) | **formal derived analysis** |
| Q-whole inference checks | [Diagnostic index](diagnostics/README.md) | **formal diagnostics** |
| Layer-wise MoE routes | [Bad-case routes](diagnostics/bad_case_routes.md) | **formal derived analysis** |
| Single true-`t1` P-whole | [Result](ground_truth_first_single_result.md) | **completed 5/5** |
| Multi5 true-`t1` P-whole | [Result](ground_truth_first_result.md) | **completed 5/5** |
| Probe protocol | [Formal protocol](formal_protocol.md) | executed |
| Capacity | [Batch capacity](capacity.md) | executed |

## Headline

| Endpoint | `single` | `multi5_permute` |
|---|---:|---:|
| P0 first, excluding birth date | 6.76% | **98.63%** |
| Q-first macro | 12.83% | **98.79%** |
| Q-whole macro | 3.18% | **33.15%** |
| Fresh P-whole + true `t1` | 56.47% | **96.38%** |

> **First-token augmentation is reproduced; whole-value readability is path-dependent on the
> MoE and does not numerically reproduce the dense-paper endpoint before true `t1`.**

## Historical provenance

The following are retained because they document decisions or superseded milestones, but they
must not be used as headline sources:

- `pilot_analysis.md`, `pilot_comparison.md`
- `formal_training_decision.md`, `ground_truth_first_training_decision.md`
- `ground_truth_first_single_training_decision.md`
- `single_formal.md`
- `q_whole_moe_diagnostics.md` (compatibility pointer)

## Terminology

- **Training-text recall:** exact recovery of biographies used for backbone training.
- **Person-held-out validation:** people held out from probe-head training only.
- **P0…P5:** hidden state before successive biography attributes.
- **Q:** name-only `[EOS, full_name, EOS]` input.
- **First:** leading BPE token; **whole:** complete value as one classification class.
- The 100% true-`t1` panel checks supplied-input identity; it is not whole accuracy.

Canonical machine data:
[`results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/`](../../../results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/).
