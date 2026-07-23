# SynBioS MoE reproduction

这是服务器上 SynBioS MoE 实验的项目级规范入口。它连接数据集、4-GPU FSDP 预训练、
strict source-text validation、formal P/Q probes、Q-whole 机制诊断、机器结果和运行历史。
任何 headline 结论都应从本页进入，而不是从 TensorBoard、单个临时日志或 pilot 文件推断。

![Formal study overview](../../results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/figures/formal_study_overview.png)

## 研究问题

在人物事实保持完全相同的情况下，将每人一篇固定顺序 biography 改为每人五篇独立措辞、
属性随机排列的 biography，是否会改变 frozen language model 中事实的线性可读出方式？
本项目复刻 Allen-Zhu & Li Part 3.1 的 P/Q probe 思路，并进一步检查 MoE backbone 上
whole-attribute 可读性和动态 expert routing。

## 精确比较条件

| 条件 | 人物 | Biographies | 语料 tokens | 预训练预算 | Formal checkpoint |
|---|---:|---:|---:|---:|---|
| `single` | 100,000 | 100,000 | 7,405,102 | 540 epochs / 17,280 steps / 3.964B scheduled tokens | epoch 540 / step 17,280 |
| `multi5_permute` | 100,000 | 500,000 | 37,046,556 | 108 epochs / 17,388 steps / 3.988B scheduled tokens | epoch 108 / step 17,388 |

两条件使用相同 profiles SHA256
`7d239f046cb5e16ac3d8d7636b6901a2430f2ccb8dc1179063e4eaed92256da1`、
seed 1337、模型配置、4-GPU FSDP、BF16、local batch 112、global batch 448 和近似 4B
scheduled-token exposure。唯一目标差异是 biography 数量、措辞和字段排列。

模型为 12-layer、hidden 768、12-head decoder-only Transformer，8 experts、top-2 routing；
总参数 293.49M，约 123.62M token-active parameters。它不是论文中的 dense GPT-2 small，
因此本项目主张机制级复刻，不主张架构和数值完全等价。

## 实验链与状态

| Stage | `single` | `multi5_permute` | 规范证据 |
|---|---|---|---|
| Dataset generation | 完成 | 完成 | `results/datasets/synbios_moe/` |
| Token shards | 完成 | 完成 | 各 variant `token_shards/manifest.json` |
| 4-GPU FSDP pretraining | 完成 | 完成 | `results/formal_runs/synbios_moe/runs/` |
| Strict source-text progressive cloze | 100k 全量完成 | 500k 全量完成 | 下方 cloze 报告 |
| 22 formal probe heads | 完成 | 完成 | `probe_pipeline/formal/` |
| 22 person-held-out validations | 完成 | 完成 | `probe_pipeline/formal/validation/` |
| Oracle true-`t1` Q-whole | 未运行 | 完成 | `probes/diagnostics/` |
| Bad-case MoE route branching | 未运行 | 完成 | `probes/diagnostics/` |

## 主要结果

| Endpoint | `single` | `multi5_permute` | 解释 |
|---|---:|---:|---|
| Strict training-corpus field recall | 100.0000% | 99.9915% | 两边都充分记忆训练文本 |
| P0 first macro，排除 birth date | 6.76% | **98.63%** | augmentation 消除固定位置依赖 |
| Q first，六属性 macro | 12.83% | **98.79%** | name → first-token 机制强复刻 |
| Q whole，五属性 macro | 3.18% | **33.15%** | 有提升，但未复刻论文 92.58% |
| Q whole `+ true t1` | — | 32.08% | 原读出头未被直接解锁 |
| Bad-case route DiD | — | +0.205 | token 2 后存在受控 route 分支 |

Formal 的 44 个训练/validation jobs 全部完成。P-first 与 Q-first 清楚复刻论文的定性模式；
whole-attribute linear readability 没有达到论文 dense 模型水平。因此项目结论是
**partial replication**：first-token storage/extraction mechanism replicated，
whole-attribute endpoint not replicated。

## 结论导航

- [Formal single vs multi5+permute 主报告](probes/formal_comparison.md)
- [Formal/diagnostic 图表和术语入口](probes/README.md)
- [Oracle 与 route 两个新 val](probes/diagnostics/README.md)
- [Single 100k strict cloze](../synbios_single_cloze_100k.md)
- [Multi5+permute 500k strict cloze](../synbios_multi5_permute_cloze_500k.md)
- [Probe batch capacity](probes/capacity.md)
- [Formal protocol 与预算](probes/formal_protocol.md)
- [Dataset manifests 与 lineage](../../results/datasets/synbios_moe/README.md)
- [Repository path/lineage audit](../../results/formal_runs/synbios_moe/results/repository_audit_20260724/summary.json)
- [完整运行历史](../../HISTORY.md)

## 存储与路径契约

```text
artifacts -> /data/mini-train-sys/artifacts
artifacts/synbios_moe/
├── single/                         原始 single 数据、token shards、probe cache
├── multi5_permute/                 原始增强数据、token shards、probe cache
├── checkpoints/                    DCP/Adam 与 model.pt；不进入 Git
├── runs/                           预训练 JSONL/TensorBoard
├── operation_logs/                 数据准备 logger 输出
└── results/                        cloze、probe、diagnostics、报告生成源产物

results/                             Git-safe、append-only 镜像
├── datasets/synbios_moe/           数据/派生 cache manifests 与 lineage
├── formal_runs/synbios_moe/        小型 run/checkpoint/probe/diagnostic 证据
├── benchmarks/                     容量与吞吐测量
├── validation/                     工程 correctness gates
├── logs/                           控制台日志与索引
└── MANIFEST.sha256                 Git-safe 文件完整性

reports/synbios_moe/                叙事结论；不保存原始模型或数据 payload
```

训练配置只读取 `artifacts/synbios_moe/<variant>/token_shards/manifest.json`；probe 只读取同
variant 下的 `probe_cache/`；formal validation 必须由其 `pipeline.json.identity` 绑定到
精确 checkpoint/data/cache SHA256。报告生成器会拒绝跨目录、跨 checkpoint 或跨 cache
拼接。

## 术语与有效性边界

- **Training-corpus recall**：backbone 对参加预训练的 biography 原文恢复，不是 held-out。
- **Person-held-out probe validation**：人物只对 probe-head training held out；backbone
  预训练见过这些人物。
- **P probe**：读取 biography 中属性出现前的六个观察位置。
- **Q probe**：只输入 `[EOS, full_name, EOS]`，在末尾 EOS 读出。
- **Whole attribute**：完整属性字符串是一个分类类别；不是 next-token 自回归生成。
- Oracle 验证改变了 Q 输入长度和读出位置，是 unchanged-head transport test。
- Route 分析条件化在 bad cases；支持路径分支，不证明 expert 是知识存储单元。
- 当前没有 matched `single` route diagnostic，不能把 +0.205 归因于 augmentation。

## 下一决策

若继续机制研究，优先执行 matched `single` oracle/route protocol，并使用 person-level
bootstrap 比较 route DiD；若研究 whole 信息是否存在但读出坐标改变，则训练独立的
matched-context Q-whole head。两者都必须与本次 unchanged-head 指标分开命名和报告。
