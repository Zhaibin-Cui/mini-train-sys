# Single formal P/Q probe（阶段性结论）

## 问题与条件

检验未增强 `single` 数据是否呈现 Allen-Zhu 式的“按字段位置逐步可读”模式。运行覆盖 11 个任务（P/Q × first/whole），P rank=2、Q rank=16；P/Q 分别使用吞吐基准推荐 batch 128/768，first 4,000 steps、whole 12,000 steps。验证集是 person-level disjoint split 的 50,118 条 single biographies（seed=1337，variant=single），不是训练集 recall。

## 主要结果

| 条件 | P held-out 平均（各位置） | Q held-out | 解释 |
|---|---:|---:|---|
| first（6 属性） | 61.0% | 12.9% | P 呈明显位置递增；Q 基本未超过先验 |
| whole（5 属性） | 44.2% | 3.2% | P 仍有位置结构但未完全收敛；Q 近 chance |

P first 的 position-0（排除 birth_date）仍为 5.2%–11.2%，而末位置为约 100%；birth_date 是固定首字段，达到 100%。这与 single 的“信息在其固定位置出现”的预期一致。Q 的 formal validation 与 pilot 一致地停留在先验附近（birth_date 41.3%，其余约 0.4%–13.0%），因此不能宣称 Q 已复刻出增强记忆趋势。

## 证据与复现路径

- 运行摘要：`/data/mini-train-sys/artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/summary/summary.json`、`summary.csv`
- 验证 JSON：`/data/mini-train-sys/artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/validation/`
- 临时总图：`/data/mini-train-sys/artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/summary/single_formal_probe_overview.png`
- TensorBoard 日志：`/data/mini-train-sys/artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal`
- 运行 provenance：`HISTORY.md` 中 2026-07-23 17:57 的 formal single 条目。

## 解读、限制与下一步

结论可靠地支持 single-P 的位置化记忆模式，但 Q 结果不是“训练不充分”唯一可解释：它也可能反映 Q 探针/标签协议与目标表示的限制。故下一步应把同一配置的 `multi5_permute` formal 作为对照，重点比较 P/Q 的 position-0 提升；不要用 single 的 Q 结果单独否定复刻。报告为阶段性结论，multi formal 完成后再更新总判断。
