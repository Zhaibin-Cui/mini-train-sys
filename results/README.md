# 📦 Server result snapshot

This directory contains the Git-safe evidence exported from the mounted experiment volume on the
4 × RTX 4090 server. Run `bash scripts/bash/export_test_results.sh` to refresh it.

## Canonical layout

- [`catalog/summary.md`](catalog/summary.md): human-readable inventory by category and scope.
- `catalog/artifacts.json`: machine-readable inventory for every pushable result file.
- `catalog/retention.json`: size, path, policy, and manifest identity for server-only payloads.
- `benchmarks/`: kernel, distributed-training, backend-capacity, and probe-capacity evidence.
- `formal_runs/`: SynBioS pretraining, cloze, formal probes, diagnostics, TensorBoard events,
  and lightweight checkpoint metadata. Published paths remain stable for auditability.
- `datasets/`: authoritative dataset/cache lineage and manifests; no raw payloads.
- `validation/`: JUnit, checkpoint-resume metadata, and correctness-gate events.
- `logs/`: physically separated `benchmarks/`, `experiments/`, `validation/`, and `maintenance/`.
- `notebooks/`: executed server benchmark notebooks and execution logs.
- [`tensorboard/index.csv`](tensorboard/index.csv): central index over all embedded event files.
- `environment/`: server software and hardware inventory.
- `smoke/`: short worker/backend smoke results.
- `MANIFEST.sha256`: content hashes for every exported file except the manifest itself.

Large model weights, optimizer/DCP tensor shards, raw dataset payloads, caches, and credentials are
deliberately excluded. DCP `.metadata` remains included because it is small and records shard
layout without tensor contents. Excluded payload paths, sizes, manifests, and hashes are retained
where available.

See `BENCHMARK_SUMMARY.md` for the current conclusions, `../reports/synbios_moe/README.md` for the
canonical SynBioS dataset-to-diagnostics map, and `../HISTORY.md` for the append-only run timeline
and exact commands.

## TensorBoard

Event files stay beside their owning run so conditions and stages cannot be confused. The central
CSV provides discovery without duplicating event data:

```bash
tensorboard --logdir results/formal_runs/synbios_moe --port 6006 --bind_all
```

Filter `single`, `multi5_permute`, `formal`, or `pilot` paths through
`results/tensorboard/index.csv`. Console logs are separately organized under `results/logs/`.
