# Distributed benchmark presentation artifacts

Source summary: `/data/mini-train-sys/artifacts/distributed_benchmark/rtx4090_125m_moe/capacity/capacity_summary.json`

- `results_aggregated.csv`: one row per strategy/world-size/local-batch; numeric repeat metrics are averaged.
- `failures.csv`: OOM, timeout, exit code, log path, and error tail.
- `capacity_frontier.json`: largest successful local/global batch for each topology.
- `capacity_overview.png`: standard notebook visualization.
