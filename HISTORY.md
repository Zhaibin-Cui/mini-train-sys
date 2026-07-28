# MiniTrainSys experiment notes

This file keeps the commands behind the results that still matter. Raw logs, hashes, and exact run
identity live under `results/`; failed retries and shell-session details are intentionally omitted.

## SynBioS backend benchmark

Purpose: select a stable 4-GPU FSDP batch and compare the Torch, Triton, and native CUDA backends on
the formal 293.49M-parameter model.

```bash
bash scripts/bash/synbios_backend_benchmark.sh
```

Retained results:

- `results/benchmarks/synbios_backend_fixed/`
- `results/benchmarks/synbios_backend_capacity/`
- `reports/engineering/kernels.md`
- `reports/engineering/distributed_training.md`

The accepted training batch is 112 per GPU. The benchmark includes forward, backward, and AdamW;
batch 120 ran but was slower and left less memory headroom.

## SynBioS datasets

Purpose: generate two corpora from the same 100,000-person fact table.

```bash
python scripts/synbios_moe.py prepare \
  --output artifacts/synbios_moe/single \
  --variant single

python scripts/synbios_moe.py prepare \
  --output artifacts/synbios_moe/multi5_permute \
  --variant multi5+permute
```

`single` contains one fixed-order biography per person. `multi5_permute` contains five rewritten,
field-permuted biographies per person. The Git-safe manifests and hashes are under
`results/pretraining/synbios_moe/datasets/`.

## Formal pretraining

Purpose: train the same MoE model and optimization budget on each corpus.

```bash
NPROC=4 bash scripts/bash/synbios_moe.sh single fsdp
NPROC=4 bash scripts/bash/synbios_moe.sh multi5_permute fsdp
```

Retained endpoints:

- `single`: epoch 540, step 17,280
- `multi5_permute`: epoch 108, step 17,388

The configs, successful event streams, and final checkpoint identities are linked from
`results/catalog/studies/synbios_moe.json`.

## Training-corpus cloze recall

Purpose: rule out failed memorization before interpreting probes. Each worker evaluates one
non-overlapping range; `summarize-cloze` rejects gaps and overlaps.

```bash
python scripts/synbios_moe.py cloze-evaluate \
  --data artifacts/synbios_moe/<condition> \
  --model-config configs/synbios_moe/model.yaml \
  --checkpoint artifacts/synbios_moe/checkpoints/<run>/<checkpoint> \
  --device cuda:<gpu> \
  --start-index <range-start> \
  --examples <range-size> \
  --output artifacts/synbios_moe/results/<condition>_cloze_eval/<run>/shard_<gpu>.json

python scripts/synbios_moe.py summarize-cloze \
  --run artifacts/synbios_moe/results/<condition>_cloze_eval/<run> \
  --output artifacts/synbios_moe/results/<condition>_cloze_eval/<run>/summary.json
```

Results:

- `single`: 600,000 / 600,000 fields exact
- `multi5_permute`: 2,999,746 / 3,000,000 fields exact

The exported summaries and shard boundaries are under
`results/cloze/synbios_moe/{single,multi5_permute}/`.

## P/Q probe study

Purpose: compare where the first token and the complete attribute value are linearly readable.
Smoke and pilot stages are prerequisites for the formal stage.

```bash
NPROC=4 STAGE=smoke bash scripts/bash/synbios_probes.sh <condition> fsdp latest
NPROC=4 STAGE=pilot bash scripts/bash/synbios_probes.sh <condition> fsdp latest

NPROC=4 \
STAGE=formal \
PROBE_BATCH_ENV=<benchmark-directory>/recommended.env \
bash scripts/bash/synbios_probes.sh <condition> fsdp latest
```

The matched formal comparison is
`results/probes/synbios_moe/comparisons/formal_20260724/`. Both probe partitions
contain people seen during pretraining; the split measures cross-person linear readout, not
generalization to unseen people.

## First-token intervention

Purpose: test whether supplying the correct first attribute token recovers the remaining value.
The inserted token is the cached label used by the first-token probe protocol, not a token predicted
by a separate classifier.

```bash
bash scripts/bash/synbios_ground_truth_first_whole.sh single
bash scripts/bash/synbios_ground_truth_first_whole.sh multi5_permute
```

The accepted P-only 4,000-step results are:

- `results/probes/synbios_moe/single/formal/diagnostics/ground_truth_first_whole_p_pilot4000_20260726T033800Z/`
- `results/probes/synbios_moe/multi5_permute/formal/diagnostics/ground_truth_first_whole_rank_matched_pilot4000_20260725T100100Z/`

## Route diagnostics and reports

Purpose: measure whether examples sharing the first value token but diverging at the second token
also diverge in their top-2 routes.

```bash
python scripts/synbios_moe.py report-probe-diagnostics \
  --single-formal artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal \
  --multi5-permute-formal artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal \
  --diagnostics artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics \
  --output artifacts/synbios_moe/results/formal_probe_diagnostics
```

Narrative reports are indexed by `reports/synbios_moe/README.md`. Exported artifacts are indexed by
`results/catalog/studies/synbios_moe.json`.

## Refresh exported evidence

```bash
bash scripts/bash/export_results.sh
python scripts/build_results_catalog.py
python scripts/build_results_manifest.py
python scripts/build_results_manifest.py --check
```

The export step copies Git-safe evidence only. Raw datasets, model weights, optimizer shards, and
caches stay on the experiment disk.
