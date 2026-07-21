# Server result snapshot

This directory contains the Git-safe evidence exported from the mounted experiment volume on the
4 × RTX 4090 server. Run `bash scripts/bash/export_test_results.sh` to refresh it.

## Layout

- `benchmarks/`: raw JSON, aggregate summaries, failure/OOM logs, CSV, figures, and inventories.
- `validation/`: JUnit, checkpoint-resume metadata (including DCP `.metadata`), and event logs.
- `formal_runs/`: formal training JSONL/TensorBoard events plus lightweight checkpoint and DCP
  layout metadata.
- `datasets/`: generation and token-shard manifests; raw biographies and token shards are excluded.
- `environment/`: server software and hardware inventory.
- `logs/`: console logs for benchmarks, validation, data preparation, and formal training.
- `smoke/`: short worker/backend smoke results.
- `MANIFEST.sha256`: content hashes for every exported file except the manifest itself.

Large model weights, optimizer/DCP tensor shards, raw dataset payloads, caches, and credentials are
deliberately excluded. DCP `.metadata` remains included because it is small and records shard
layout without tensor contents. Excluded payload paths, sizes, manifests, and hashes are retained
where available.

See `BENCHMARK_SUMMARY.md` for the current conclusions and `../HISTORY.md` for the append-only run
timeline and exact commands.
