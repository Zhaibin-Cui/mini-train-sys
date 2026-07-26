# Benchmark evidence

| Category | Directory | Purpose |
|---|---|---|
| Kernels | `operator_benchmark/` | Dense/MoE raw scans, summaries, figures, dispatch and Nsight |
| Formal backend, equal work | `synbios_backend_fixed/` | Torch/Triton/CUDA fixed-batch repeats |
| Formal backend, equal memory | `synbios_backend_capacity/` | Capacity frontier and activation-memory subtraction |
| Formal FSDP scaling/capacity | `synbios_moe_fsdp4/` | Exact 293M MoE weak scaling and batch boundary |
| Generic distributed matrix | `rtx4090_125m_moe/` | Single/DDP/FSDP development comparison |
| Probe capacity | `synbios_moe/` | P/Q and true-`t1` probe batch selection |

Canonical conclusions: [`reports/engineering/`](../../reports/engineering/). Raw failures and OOMs
are retained because they define capacity boundaries.
