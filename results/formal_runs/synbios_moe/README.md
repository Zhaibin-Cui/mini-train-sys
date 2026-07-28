# SynBioS formal study

This directory contains the retained 4-GPU FSDP runs for the two SynBioS conditions. The
repository paths below are exported copies of the original `artifacts/synbios_moe/...` paths used
on the training server.

| Condition | Training data | Final checkpoint | Cloze recall |
|---|---:|---:|---:|
| `single` | 100,000 people × 1 biography | epoch 540, step 17,280 | 600,000 / 600,000 fields |
| `multi5_permute` | 100,000 people × 5 biographies | epoch 108, step 17,388 | 2,999,746 / 3,000,000 fields |

Both conditions use the same profile table, tokenizer, model configuration, optimization budget,
and probe split. Their profile-table SHA-256 is
`7d239f046cb5e16ac3d8d7636b6901a2430f2ccb8dc1179063e4eaed92256da1`.

## Reproduce the pipeline

Prepare and train one condition:

```bash
python scripts/synbios_moe.py prepare \
  --output artifacts/synbios_moe/single \
  --variant single

NPROC=4 bash scripts/bash/synbios_moe.sh single fsdp
```

Run the formal probes after selecting the retained batch-size benchmark:

```bash
NPROC=4 \
STAGE=formal \
PROBE_BATCH_ENV=<benchmark-directory>/recommended.env \
bash scripts/bash/synbios_probes.sh single fsdp latest
```

Replace `single` with `multi5_permute`; its preparation variant is `multi5+permute`. Smoke and
pilot stages must pass before the formal stage. The launcher checks the dataset manifest,
checkpoint commit marker, probe-cache coverage, and stage prerequisites.

## Retained evidence

The machine-readable [study index](study_index.json) links each condition to its dataset lineage,
run config, training events, final checkpoint, cloze summary, formal probe summary, and report.
The main comparison is in
[formal_probe_comparison_20260724](results/formal_probe_comparison_20260724/).

The result catalog and `results/MANIFEST.sha256` cover these exported files. Regenerate both after
changing retained evidence:

```bash
python scripts/build_results_catalog.py
python scripts/build_results_manifest.py
python scripts/build_results_manifest.py --check
```
