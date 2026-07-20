# 报告目录

这里只保存已经选定、值得长期阅读的结论和图表；可重跑的原始数据留在
`tests/benchmark_results/` 或 `artifacts/`，不要把大批临时 JSON/trace 提交为报告。

- `operator_bench.md`：算子 benchmark 汇总；
- `figures/`：对应图表；
- `nsight/`：选定的 Nsight 证据；
- `training_bench.md`、`distributed_bench.md`：尚未在目标服务器形成稳定结论的占位页。

报告必须写明硬件、软件版本、精度、shape、warmup、重复次数和 raw 数据位置。没有真实
测量时明确写“尚未测量”，不能填推测数字。
