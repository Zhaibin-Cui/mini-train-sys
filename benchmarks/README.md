# Benchmarks

Benchmark code is separate from correctness tests because it measures hardware-dependent latency,
memory, capacity, and scaling.

## Operators

| Entry | Purpose |
|---|---|
| `operators/operator_bench.ipynb` | Interactive correctness and shape sweeps |
| `operators/operator_bench_linux_server.ipynb` | Isolated dense-kernel runs on the experiment server |
| `operators/moe_operator_bench.ipynb` | Interactive router and fused-MoE checks |
| `operators/moe_operator_bench_linux_server.ipynb` | Isolated router and fused-MoE server runs |
| `operators/dense_runner.py` | Dense benchmark worker |
| `operators/moe_runner.py` | MoE benchmark worker |
| `operators/measurement.py` | Shared timing, memory, correctness, and persistence code |

Runtime output goes to `artifacts/operator_benchmark/`. Git-safe results are exported to
`results/benchmarks/operator_benchmark/`.

## Distributed training

`distributed/distributed_server_benchmark.ipynb` drives the single-device, DDP, and FSDP suites.
Its reusable orchestration code is in `distributed/workflow.py`; measurement workers remain in
`scripts/run_dist_bench.py`.
