# Distributed benchmark presentation artifacts

Source summary: `/data/mini-train-sys/artifacts/distributed_benchmark/rtx4090_125m_moe/weak/weak_summary.json`

- `results_aggregated.csv`: one row per strategy/world-size/local-batch; numeric repeat metrics are averaged.
- `failures.csv`: OOM, timeout, exit code, log path, and error tail.
- `quality_gates.json`: weak-scaling efficiency, data-stall, memory, and completion gates.
- `weak_overview.png`: standard notebook visualization.
