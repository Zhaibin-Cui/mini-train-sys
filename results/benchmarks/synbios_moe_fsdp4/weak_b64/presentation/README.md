# Distributed benchmark presentation artifacts

Source summary: `results\benchmarks\synbios_moe_fsdp4\weak_b64\weak_summary.json`

- `results_aggregated.csv`: one row per strategy/world-size/local-batch; numeric repeat metrics are averaged.
- `failures.csv`: OOM, timeout, exit code, log path, and error tail.
- `quality_gates.json`: weak-scaling efficiency, data-stall, memory, and completion gates.
- `weak_overview.png`: standard notebook visualization.
