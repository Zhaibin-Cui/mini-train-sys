# Artifact and report layout

MiniTrainSys separates mutable server payloads, Git-safe evidence, narrative conclusions, and
run provenance. A result should never be copied ad hoc between these layers.

```text
/data/mini-train-sys/artifacts/       authoritative mutable server artifacts
├── logs/                             tmux stdout/stderr
├── notebooks/                        executed server notebooks
├── operator_benchmark/               kernel raw scans
├── distributed_benchmark/            FSDP/backend raw scans
├── validation/                       correctness and recovery artifacts
└── synbios_moe/
    ├── single|multi5_permute/         datasets, token shards, probe cache
    ├── checkpoints/                   model + DCP/Adam shards
    ├── runs/                          pretraining JSONL/TensorBoard
    └── results/                       cloze, probes, diagnostics

results/                              Git-safe machine evidence
├── catalog/                          complete inventory + server-only retention
├── benchmarks/                       raw/aggregate kernel and training measurements
├── datasets/                         manifests, lineage, hashes, compact statistics
├── formal_runs/                      formal events and aggregate experiment outputs
├── logs/{benchmarks,experiments,validation,maintenance}/
├── notebooks/                        executed benchmark notebooks
├── tensorboard/index.csv             discovery index; events remain with owner
├── validation/                       correctness and recovery evidence
└── MANIFEST.sha256                   snapshot integrity

reports/                              selected human conclusions
├── engineering/                      kernels and distributed/end-to-end performance
└── synbios_moe/                      Allen-Zhu-style reproduction and diagnostics

HISTORY.md                            local concise command notes (gitignored)
```

## Path rules

1. Raw data, weights, DCP tensor shards, caches, and large per-example records never enter Git.
2. Dataset/cache manifests identify parents, split semantics, counts, and SHA256.
3. TensorBoard events stay beside the exact run; the central index only points to them.
4. Move a published path only when every code, command, and documentation reference is updated in
   the same change.
5. `reports/` contains interpretation; `results/` contains machine evidence; formal experiment
   links belong in a machine-readable study index.
6. Run `bash scripts/bash/export_test_results.sh` after every benchmark or validation cycle. It
   categorizes logs, exports notebooks, rebuilds the catalog, and refreshes all hashes.

## Push readiness

```bash
bash scripts/bash/export_test_results.sh
python scripts/build_results_manifest.py --results results --check
find results -type f -size +90M -print
git diff --check
```

The expected large-payload inventory is
[`results/catalog/retention.json`](../../results/catalog/retention.json); a missing payload is not
covered merely because its directory name appears there.
