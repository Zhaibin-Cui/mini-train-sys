# 📊 Reports

Only selected, auditable conclusions belong here. Raw JSON/CSV/TensorBoard/JUnit evidence lives
under [`results/`](../results/); commands and run lifecycle live in the retained run records.

## Canonical reports

| Area | Report | Headline |
|---|---|---|
| ⚡ Kernels | [Design + isolated benchmark](engineering/kernels.md) | 1.22–6.51× speedup; up to 94% lower allocation |
| 🚀 Distributed | [FSDP + end-to-end](engineering/distributed_training.md) | 92.24% FSDP4 efficiency; 2.20× at equal VRAM |
| 🧠 Research | [SynBioS mechanism analysis](../README.md#model-interpretability) | P/Q readout, route branching, and true-`t1` intervention |
| 🧬 Experiment index | [SynBioS report map](synbios_moe/README.md) | Dataset → training → probes → route → true-`t1` |

## Research detail

- [Formal single vs multi5+permute P/Q comparison](synbios_moe/probes/formal_comparison.md)
- [Company/company-city quantitative fit](synbios_moe/probes/company_pair_position_fit.md)
- [Layer-wise MoE route branching](synbios_moe/probes/diagnostics/bad_case_routes.md)
- [Single fresh P-whole + true `t1`](synbios_moe/probes/ground_truth_first_single_result.md)
- [Multi5 fresh P-whole + true `t1`](synbios_moe/probes/ground_truth_first_result.md)
- [Single 100k source-text cloze](synbios_single_cloze_100k.md)
- [Multi5 500k source-text cloze](synbios_multi5_permute_cloze_500k.md)

## Historical/provenance documents

Pilot analyses, launch decisions, and superseded stage reports remain under
`synbios_moe/probes/` because they explain budget choices and failed boundaries. They are not
headline sources. Compatibility pages at `operator_bench.md`, `distributed_bench.md`,
`server_benchmark_resume.md`, and `q_whole_moe_diagnostics.md` preserve old links while pointing
to the canonical reports above.

## Writing contract

Every conclusion states, in order: question, compared conditions, run/dataset identity, primary
metrics, machine artifact paths, interpretation, limitations, and next action. Training-text
recall, held-out probe validation, smoke tests, failed runs, and formal results are always
labelled separately.
