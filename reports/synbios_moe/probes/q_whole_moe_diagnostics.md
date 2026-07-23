# Multi5+permute Q-whole 的首 token 干预与 MoE route 分支

> 本路径保留为两个诊断的历史合并报告。规范的独立报告、全量 whole 对比和重建说明见
> [diagnostics/README.md](diagnostics/README.md)、
> [Oracle intervention](diagnostics/oracle_first_token.md) 与
> [route branching](diagnostics/bad_case_routes.md)。

![Q-whole 诊断总览](../../../results/formal_runs/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/report/figures/diagnostic_study_overview.png)

## 问题与假设

本次验证询问两件事：

1. Q-whole 较弱是否只是因为姓名表示缺少属性首 token；若把真实首 token 放到姓名后，
   同一个已训练 Q-whole 头能否恢复完整属性？
2. 在 `Q-first 正确、Q-whole 错误` 的样本中，同一首 token 的事实是否先走相似 expert
   route，并在第二个 token 不同时发生分支？

## 精确比较条件

- Oracle 基线：`[EOS, name, EOS]`，在末尾 EOS 使用正式 Q-whole 头。
- Oracle 干预：`[EOS, name, ground_truth_t1, EOS]`，仍在末尾 EOS 使用同一个头。
- Route 对照：同属性、同 `t1`、同 `t2` 的 bad-case 样本对。
- Route 分支组：同属性、同 `t1`、不同 `t2` 的 bad-case 样本对。
- `branching_score = Jaccard(route_t1) - Jaccard(route_t2)`；route 是每层 top-2 expert
  集合。Route forward 使用冻结预训练 backbone，不加入 probe embedding delta。
- 两项都是 inference-only，不重新训练参数。

## Run、checkpoint 与数据身份

- 条件：`multi5_permute` formal。
- Backbone：
  `epoch_000108_step_000017388`，模型 SHA256
  `e89075289bb3a774825e7fd03cedc2c7c37957583bf3656e8ab32c52ef02f0dd`。
- Probe：formal 的 22 个已完成分类头；P rank 2、Q rank 16，first 4,000 steps、
  whole 12,000 steps。
- 数据：100,000 profiles、500,000 biographies；本报告只评估 50,118 人的
  person-held-out validation split。
- Cache manifest SHA256：
  `acd78360d0daa7cf0d2c557fc9f68f07431bc3063cee1145daa3f14c320a232f`。
- 生命周期和命令：`HISTORY.md` 的
  “2026-07-24 00:35 — Multi5+permute Q-whole inference diagnostics”。

## 主要指标

### Oracle 真实首 token

| 属性 | name-only | + true t1 | Δ | 基线错误恢复率 | 基线正确受损率 |
|---|---:|---:|---:|---:|---:|
| birth city | 12.92% | 12.93% | +0.01pp | 1.86% | 12.48% |
| university | 8.48% | 8.50% | +0.01pp | 1.18% | 12.54% |
| major | 47.32% | 49.70% | +2.38pp | 9.51% | 5.56% |
| company | 45.97% | 35.52% | −10.45pp | 9.28% | 33.64% |
| company city | 51.04% | 53.76% | +2.73pp | 10.73% | 4.95% |
| **micro overall** | **33.15%** | **32.08%** | **−1.06pp** | **5.38%** | **14.06%** |

每个属性有 50,118 个样本，共 250,590 次比较。真实首 token 能恢复 9,009 个基线错误，
但同时破坏 11,675 个原本正确的预测。因此，结果不支持“只要给现有 Q-whole 头补 t1，
whole 就会被直接解锁”。

### Bad-case route 分支

共得到 162,044 个 `first 正确、whole 错误、真实值至少两个 token` 的样本。跨属性、跨层并按
样本对数加权后：

| 配对组 | t1 route overlap | t2 route overlap | branching score |
|---|---:|---:|---:|
| same t2 control | 0.530 | 0.580 | −0.051 |
| different t2 branch | 0.520 | 0.365 | +0.154 |
| **difference-in-differences** |  |  | **+0.205** |

不同 `t2` 组的 difference-in-differences 在 12 层均为正，逐层为
`[0.513, 0.676, 0.274, 0.258, 0.146, 0.106, 0.054, 0.155, 0.041, 0.065, 0.077, 0.095]`。
信号在第 0–3 层最强，后层仍为正但明显减弱。bad-case 子集上的 token/top-1-expert NMI
并不低：各属性最大 t1 NMI 为 0.360–0.551，最大 t2 NMI 为 0.563–0.651。

## 支持产物

- Oracle machine summary：
  `results/formal_runs/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/oracle_first_token/summary.json`
- Route machine summary 与配对表：
  `results/formal_runs/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/bad_case_routes/`
- 图：`accuracy_before_after.png`、`route_overlap_by_layer.png`、
  `branching_heatmap.png`、`expert_load_t1_vs_t2.png`，位于各自 `figures/`。
- `/data` 原始证据：`oracle_first_token/records.csv`（约 40 MB）、
  `bad_case_routes/bad_cases.csv`（约 34 MB）和
  `bad_case_routes/route_records.csv`（约 108 MB）。
- `route_records.csv` SHA256：
  `2a33ceda133449142a81b6b8fe7da2b7707f2c08c5ead42937e36704f48e7061`。

## 解释

结果支持一个较窄、可审计的结论：在 Q-whole 的 bad cases 中，MoE route 对当前 token
身份高度敏感；共享 `t1` 的样本在 `t1` 位置更相似，而不同 `t2` 到来后 route 显著分化。
这与“属性序列沿 token 条件路径展开”的描述一致，并说明只检查姓名位置或只看一个静态
expert-label NMI 会漏掉动态分支。

但 oracle 结果表明，这不能简化成“把 t1 追加到姓名后，原 Q-whole 线性读出就会变好”。
更可能的情况是完整属性表示依赖位置、上下文和随 token 演化的 hidden-state/route 轨迹，
而不是一个可被原 name-only 分类头直接消费的离散 key。

## 局限与有效性威胁

- Oracle 输入改变了序列长度和 readout 分布；Q-whole 头只在 name-only EOS 上训练，
  所以下降可能部分来自 out-of-distribution readout，而不是信息不存在。
- Route 分析是 bad-case 条件子集，不能与所有样本的 route 频率直接比较，也不能单独证明
  expert 是事实的存储位置。
- 同 token 的路由相似性可能来自 token embedding、频率或语法，而非人物事实；same-t2
  对照和 difference-in-differences 降低但没有消除这些混杂。
- 当前结果只覆盖 multi5+permute；缺少同协议 single 对照，尚不能把动态分支归因于
  augmentation。

## 下一决策

保留这两个 inference-only 验证，不重新训练 probe。下一步在 `single` formal 上运行完全
相同协议，比较 difference-in-differences 的层级曲线和 oracle Δ；只有 multi5 显著增强、
且在 matched controls 下稳定时，才把结论升级为“augmentation 改变了 MoE 的序列化存储
路径”。若需要检验可读出性而非零训练干预，再单独训练 matched
`[EOS, name, true_t1, EOS]` readout，不能与本次 oracle 指标混为一谈。
