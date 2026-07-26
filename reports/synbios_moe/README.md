# 🧬 SynBioS MoE research

This directory is the canonical research-report entry for the Allen-Zhu-style knowledge storage
study. The detailed narrative is intentionally centralized in one document:

## [Read the storage study →](storage_story.md)

### Headline metrics

| Endpoint | `single` | `multi5_permute` |
|---|---:|---:|
| Strict training-text field recall | **100.0000%** | **99.9915%** |
| P0 first, excluding birth date | 6.76% | **98.63%** |
| Q-first macro | 12.83% | **98.79%** |
| Q-whole macro | 3.18% | **33.15%** |
| Fresh P-whole + true `t1`, 30-cell macro | 56.47% | **96.38%** |

The result is a **mechanism-level partial reproduction**: first-token augmentation strongly
matches Allen-Zhu, while whole-value readability differs on the MoE. Company/company-city
association, layer-wise route branching, and the true-`t1` fresh-head intervention explain the
gap with a path-dependent readout model.

## Report map

| Stage | Canonical report | Machine evidence |
|---|---|---|
| Dataset + pretraining | [Storage study §1](storage_story.md#1-experimental-question) | [`results/datasets/synbios_moe/`](../../results/datasets/synbios_moe/) |
| Formal P/Q | [Formal comparison](probes/formal_comparison.md) | [`formal_probe_comparison_20260724/`](../../results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/) |
| Company relation | [Quantitative fit](probes/company_pair_position_fit.md) | [`company_pair_position_fit/`](../../results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/company_pair_position_fit/) |
| Route branching | [Layer analysis](probes/diagnostics/bad_case_routes.md) | [`diagnostics/report/`](../../results/formal_runs/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/report/) |
| True-`t1` P-whole | [Single](probes/ground_truth_first_single_result.md) · [Multi5](probes/ground_truth_first_result.md) | Each formal diagnostic run root |
| Source-text cloze | [Single](../synbios_single_cloze_100k.md) · [Multi5](../synbios_multi5_permute_cloze_500k.md) | `results/formal_runs/synbios_moe/results/*_cloze_eval/` |

## Validity labels

- **Training-text recall** is not held-out generalization.
- **Person-held-out probe validation** holds people out from probe training, not backbone
  pretraining.
- **Whole** is complete-value classification, not autoregressive next-token generation.
- **Route branching** supports a dynamic-path hypothesis; it does not prove an expert is a fact
  database.

Run identity, failures, exact commands, checkpoints, and dirty-state provenance are append-only in
[`HISTORY.md`](../../HISTORY.md). Dataset/cache lineage is under
[`results/datasets/synbios_moe/`](../../results/datasets/synbios_moe/).
