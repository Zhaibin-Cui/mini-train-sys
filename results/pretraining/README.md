# Pretraining results

This directory contains evidence produced before downstream evaluation:

- `synbios_moe/datasets/`: dataset and token-shard manifests;
- `synbios_moe/runs/`: formal training events for both data conditions;
- `synbios_moe/checkpoints/`: lightweight DCP commit, runtime, and RNG metadata;
- `synbios_moe/preparation_logs/` and `synbios_moe/logs/`: data preparation and training logs;

Raw datasets and tensor checkpoint shards stay on the mounted artifact volume. Their retained size
and location are recorded in [`../catalog/retention.json`](../catalog/retention.json).

See [`synbios_moe/README.md`](synbios_moe/README.md) for the two formal pretraining runs.
