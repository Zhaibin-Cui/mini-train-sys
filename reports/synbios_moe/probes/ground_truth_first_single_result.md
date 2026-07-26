# Single fresh P-whole probes with ground-truth `t1`

## Question or hypothesis

在 `single` backbone 中，把正确的属性首 token 追加到每个 biography prefix 后，是否会让
完整属性值比原 formal no-`t1` P-whole probe 更容易线性读出？

## Exact compared conditions

基线是原 `single` formal P-whole：冻结同一 backbone，在原 prefix 读出，rank-2
`AttributeProbe` 训练 12,000 step。干预条件仍冻结同一 backbone，但在 prefix 后追加缓存的
ground-truth `t1`，在 `t1` 位置训练新的 rank-2 whole head；五个属性各训练 4,000 step，
batch 128、held-out batch 3,072、seed 1337。两边使用相同人物 split、类别映射和 probe
cache；本实验没有 fresh Q task。

## Run/checkpoint and dataset identity

- Backbone:
  `artifacts/synbios_moe/checkpoints/synbios_moe_single_fsdp_4gpu/epoch_000540_step_000017280`
- Dataset manifest SHA256:
  `144cf49ea607b4a502e5be277dbb63e0e9a08f296596e994cd19d3c6cfb11e25`
- Probe-cache manifest SHA256:
  `9dcfd2cff38d6f3d29d7f10c2a3247b634f31395b3cda3755a755c8964ffaf5b`
- Run:
  `artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/diagnostics/ground_truth_first_whole_p_pilot4000_20260726T033800Z`
- Evaluation：完整 50,118-person held-out probe split，共 300,708 个属性/位置样本。人物对
  probe-head training held out，但参加过 backbone pretraining，因此这是 representation
  readout，不是 unseen-person generalization。

## Primary metrics

| P-whole endpoint | Formal no-`t1` | Fresh probe + true `t1` | 变化 |
|---|---:|---:|---:|
| 五属性 × 六 source positions macro | 44.23% | **56.47%** | +12.24 pp |
| P0 五属性 macro | 3.16% | **15.22%** | +12.06 pp |

| Whole attribute，六位置 macro | Formal no-`t1` | Fresh + true `t1` |
|---|---:|---:|
| birth city | 75.35% | **83.66%** |
| university | 52.30% | **67.67%** |
| major | 46.34% | **55.47%** |
| company | 19.26% | **38.82%** |
| company city | 27.89% | **36.75%** |

全部五个 fresh heads 完成 4,000 step 和全量验证；验证 macro 分别为 83.66%、67.67%、
55.47%、38.82% 和 36.75%。训练末端与 held-out macro 接近，没有明显 probe-train /
validation gap。对照之下，相同 4,000-step 干预在 `multi5_permute` 将全部 30 cells 的
macro 从 45.28% 提升到 96.38%，P0 从 32.59% 提升到 95.62%。

## Supporting artifacts

- [三栏对比图](../../../results/formal_runs/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/diagnostics/ground_truth_first_whole_p_pilot4000_20260726T033800Z/figures/ground_truth_first_p_overview.png)
- Machine-readable aggregate: run root `summary.json`
- Position-level table: run root `summary.csv`
- Five task JSON/PT files、loss curves、recovery checkpoints 和 operation logs：同一 run root
- Lifecycle and exact command: `HISTORY.md` 的
  “2026-07-26 11:38 — Single ground-truth-t1 fresh P-whole matrix”

图的左栏只确认输入中的 ground-truth `t1` 与缓存标签一致，固定为 100%，不是 whole
预测准确率；中栏和右栏才是 no-`t1` 与 fresh true-`t1` whole accuracy。

## Interpretation

true `t1` 在 `single` 上确实改善完整值的线性可读性，但提升有限，且远弱于
`multi5_permute`。尤其 P0 仍只有 15.22%，说明给出首 token 并不能消除 single
固定字段顺序形成的位置依赖。相反，增强条件在相同 fresh-head 配置下接近饱和，支持
data augmentation 不只强化 name→attribute，还改变了后续 token 在各 prefix 位置上的
可读出结构。

## Limitations or threats to validity

干预 head 只训练 4,000 step，而原 formal baseline 为 12,000 step；但五个 fresh head
训练末端准确率与 held-out macro 接近，当前结论只主张所选 pilot 预算下的明显跨条件
对比，不主张 single 已达到其可实现上限。追加 `t1` 同时改变序列长度和读出坐标，因此
不能把改善完全归因于 token 内容，也不能由此定位某个 MoE expert 为事实存储单元。

## Next decision/action

保留这轮作为 matched 4,000-step single 对照，不追加 fresh Q。下一步若继续验证因果机制，
优先对 `single` 运行与 multi5+permute 完全一致的 inference-only route protocol，再做
person-level bootstrap 的跨条件 route DiD；不要把增加 P probe step 当成 route 证据。
