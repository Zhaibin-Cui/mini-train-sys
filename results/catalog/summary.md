# Result catalog

This is the deterministic index of the Git-safe server snapshot. Paths are relative to
`results/`; large server-only payloads are listed in `retention.json`.

| Category | Scope | Files | Size |
|---|---|---:|---:|
| `benchmarks` | `distributed_training` | 370 | 2.71 MiB |
| `benchmarks` | `kernels` | 224 | 8.99 MiB |
| `benchmarks` | `probes` | 89 | 1.06 MiB |
| `catalog` | `export_audit.json` | 1 | 791.00 B |
| `datasets` | `README.md` | 1 | 445.00 B |
| `datasets` | `synbios_moe` | 17 | 122.01 KiB |
| `environment` | `inventory` | 1 | 2.63 KiB |
| `formal_runs` | `synbios_moe` | 1,007 | 160.13 MiB |
| `logs` | `README.md` | 1 | 684.00 B |
| `logs` | `benchmarks` | 28 | 328.22 KiB |
| `logs` | `experiments` | 64 | 4.70 MiB |
| `logs` | `maintenance` | 23 | 32.28 KiB |
| `logs` | `validation` | 7 | 17.40 KiB |
| `notebooks` | `executed_benchmarks` | 5 | 5.39 MiB |
| `root` | `index` | 2 | 8.76 KiB |
| `smoke` | `checkpoints` | 4 | 33.31 KiB |
| `smoke` | `distributed_worker_fix.json` | 1 | 1.23 KiB |
| `smoke` | `local_runs` | 2 | 6.15 KiB |
| `smoke` | `single_worker.json` | 1 | 1.24 KiB |
| `tensorboard` | `index` | 1 | 622.00 B |
| `validation` | `synbios_moe` | 22 | 1.54 MiB |

## TensorBoard

- Event files: **264**
- Total size: **71.36 MiB**
- Machine index: [`../tensorboard/index.csv`](../tensorboard/index.csv)

## Server-only retention

| Group | Files | Size | Location |
|---|---:|---:|---|
| `dataset_single` | 20 | 166.68 MiB | `/data/mini-train-sys/artifacts/synbios_moe/single` |
| `dataset_multi5_permute` | 23 | 696.07 MiB | `/data/mini-train-sys/artifacts/synbios_moe/multi5_permute` |
| `formal_checkpoints_single` | 35 | 11.51 GiB | `/data/mini-train-sys/artifacts/synbios_moe/checkpoints/synbios_moe_single_fsdp_4gpu` |
| `formal_checkpoints_multi5_permute` | 35 | 11.51 GiB | `/data/mini-train-sys/artifacts/synbios_moe/checkpoints/synbios_moe_multi5_permute_fsdp_4gpu` |
| `validation_checkpoints` | 23 | 8.09 GiB | `/data/mini-train-sys/artifacts/validation/synbios_moe/checkpoints` |
| `probe_weights_and_records` | 203 | 969.72 MiB | `/data/mini-train-sys/artifacts/synbios_moe/results` |
