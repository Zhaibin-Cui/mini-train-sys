# Result catalog

This is the deterministic index of the Git-safe server snapshot. Paths are relative to
`results/`; large server-only payloads are listed in `retention.json`.

| Category | Scope | Files | Size |
|---|---|---:|---:|
| `benchmarks` | `distributed_training` | 370 | 2.71 MiB |
| `benchmarks` | `executed_notebooks` | 5 | 5.39 MiB |
| `benchmarks` | `kernels` | 224 | 8.99 MiB |
| `benchmarks` | `logs` | 29 | 332.47 KiB |
| `benchmarks` | `probes` | 89 | 1.06 MiB |
| `catalog` | `audits` | 3 | 10.79 KiB |
| `catalog` | `export_audit.json` | 1 | 791.00 B |
| `catalog` | `studies` | 1 | 3.57 KiB |
| `catalog` | `tensorboard` | 1 | 686.00 B |
| `cloze` | `index` | 1 | 677.00 B |
| `cloze` | `synbios_moe` | 40 | 827.17 KiB |
| `environment` | `inventory` | 1 | 2.63 KiB |
| `pretraining` | `index` | 1 | 684.00 B |
| `pretraining` | `synbios_moe` | 66 | 71.10 MiB |
| `probes` | `index` | 1 | 647.00 B |
| `probes` | `synbios_moe` | 500 | 64.33 MiB |
| `root` | `index` | 2 | 8.23 KiB |

## TensorBoard

- Event files: **155**
- Total size: **60.31 MiB**
- Machine index: [`tensorboard/index.csv`](tensorboard/index.csv)

## Server-only retention

| Group | Files | Size | Location |
|---|---:|---:|---|
| `dataset_single` | 20 | 166.68 MiB | `/data/mini-train-sys/artifacts/synbios_moe/single` |
| `dataset_multi5_permute` | 23 | 696.07 MiB | `/data/mini-train-sys/artifacts/synbios_moe/multi5_permute` |
| `formal_checkpoints_single` | 35 | 11.51 GiB | `/data/mini-train-sys/artifacts/synbios_moe/checkpoints/synbios_moe_single_fsdp_4gpu` |
| `formal_checkpoints_multi5_permute` | 35 | 11.51 GiB | `/data/mini-train-sys/artifacts/synbios_moe/checkpoints/synbios_moe_multi5_permute_fsdp_4gpu` |
| `validation_checkpoints` | 23 | 8.09 GiB | `/data/mini-train-sys/artifacts/validation/synbios_moe/checkpoints` |
| `probe_weights_and_records` | 203 | 969.72 MiB | `/data/mini-train-sys/artifacts/synbios_moe/results` |
