# 报告目录

这里只保存已经选定、值得长期阅读的结论和图表；可重跑的原始数据留在
`tests/benchmark_results/` 或 `artifacts/`，不要把大批临时 JSON/trace 提交为报告。

- `operator_bench.md`：算子 benchmark 汇总；
- `figures/`：对应图表；
- `nsight/`：选定的 Nsight 证据；
- `synbios_moe/probes/capacity.md`：P/Q probe 四卡 batch 容量结论与正式推荐；
- `synbios_moe/README.md`：SynBioS 数据、预训练、cloze、formal probes、diagnostics 和
  路径契约的项目级规范入口；
- `synbios_moe/probes/README.md`：SynBioS MoE 实验链路、正式结论和术语入口；
- `synbios_moe/probes/formal_comparison.md`：single 与 multi5+permute 的最终 formal
  held-out 对照、Allen-Zhu 参照和复刻边界；
- `synbios_moe/probes/pilot_comparison.md`：single 与 multi5+permute 的 held-out P/Q 对照、论文式主表和趋势闸门；
- `synbios_moe/probes/formal_protocol.md`：正式 P/Q 的论文忠实配置和 first/whole 指标决策；
- `synbios_moe/probes/single_formal.md`：single formal 阶段性历史报告；正式 headline
  已由 `formal_comparison.md` 取代；
- `synbios_moe/probes/q_whole_moe_diagnostics.md`：multi5+permute Q-whole 的真实首 token
  oracle 干预与 bad-case 跨层 MoE route 分支的历史合并报告；
- `synbios_moe/probes/diagnostics/README.md`：两个 Q-whole inference-only val 的规范入口、
  完整 whole 对比图和可重建机器结果；
- `synbios_moe/probes/diagnostics/oracle_first_token.md`：全部五个 whole 属性的原 formal
  Q 与 oracle `+ true t1` 对照；
- `synbios_moe/probes/diagnostics/bad_case_routes.md`：全部五属性、12 层的受控 MoE route
  branching 分析；
- `training_bench.md`、`distributed_bench.md`：尚未在目标服务器形成稳定结论的占位页。

报告必须写明硬件、软件版本、精度、shape、warmup、重复次数和 raw 数据位置。没有真实
测量时明确写“尚未测量”，不能填推测数字。
