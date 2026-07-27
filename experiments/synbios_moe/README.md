# SynBioS MoE 实验入口

这个实验沿用 Allen-Zhu/Li bioS 的数据与 probing 机制，但把每层 Dense FFN 换成
MiniTrain dropless top-2 MoE。它是“机制复现 + 架构改造”，不是论文 dense 数值的逐点
复现。差异见 [fidelity ledger](../../docs/experiments/synbios_moe_reproduction_plan.md)，完整代码流程
见 [训练导读](../../docs/experiments/synbios_moe_training_flow.md)。Probe 的缓存、阶段和多卡入口见
[Probe pipeline 手册](../../docs/experiments/synbios_moe_probe_pipeline.md)。

## 最小结构测试

推荐运行 `tests/synbios_moe_end_to_end.ipynb`。命令行等价入口：

```bash
python scripts/synbios_moe.py prepare \
  --output artifacts/synbios_moe/smoke --variant single --num-people 100
python scripts/train.py --device cpu --smoke-steps 2 \
  --config configs/synbios_moe_smoke_pretrain.yaml \
  --model-config configs/synbios_moe_smoke_model.yaml
```

Smoke 只验证调用链，不代表收敛或论文结果。

## 正式数据条件

| 条件 | 每人文本数 | Epoch | 目的 |
|---|---:|---:|---|
| `single` | 1 | 540 | 固定表达、固定属性顺序 |
| `multi5_permute` | 5 | 108 | 多表达并独立打乱句序 |

两边使用相同 seed 生成逐字一致的 `profiles.jsonl`，只改变 biography 表达。5 倍语料配
1/5 epoch，使人物曝光和总 token 预算接近一致。

## 启动

```bash
# 24 GB RTX 4090 单卡
bash scripts/bash/synbios_moe.sh single single
bash scripts/bash/synbios_moe.sh multi5_permute single

# 单机 4/8 卡
NPROC=4 bash scripts/bash/synbios_moe.sh single ddp
NPROC=8 bash scripts/bash/synbios_moe.sh single ddp
NPROC=4 bash scripts/bash/synbios_moe.sh single fsdp
NPROC=8 bash scripts/bash/synbios_moe.sh single fsdp
```

脚本会准备缺失数据，并默认从同 run 最新 committed checkpoint 恢复。设置
`AUTO_RESUME=0` 才会从头开始。`FORCE_PREPARE=1` 会重建语料；存在 checkpoint 时脚本
拒绝这样做，防止模型状态与新数据不匹配。

默认磁盘上最多保留最新两个完整恢复点和一个较老安全锚点，只有最新目录保留额外
`model.pt`。最新 checkpoint 不可用时，用 `RESUME=safety` 启动同一命令即可从安全锚点
恢复完整模型、Adam、scheduler 和 RNG。

正式配置无梯度累计。硬件允许时可提高每卡 batch，但固定使用论文的 LR `1e-3`、
warmup 1,000 step 与 cosine floor `1e-4`，不再按实际 global batch 线性放大。

## 训练后阶段

```bash
# 先跑2个500-step调用链任务
STAGE=smoke NPROC=8 bash scripts/bash/synbios_probes.sh single ddp latest

# 再跑全部22个3k-step任务
STAGE=pilot NPROC=8 bash scripts/bash/synbios_probes.sh single ddp latest

# pilot通过后运行论文30k-step预算
STAGE=formal NPROC=8 bash scripts/bash/synbios_probes.sh single ddp latest
```

脚本先一次性生成全部 probe mmap 数据，再做预训练 attribute-token gate，并把独立任务动态
分给任意数量 GPU。Probe 冻结 backbone，只训练自己的低秩参数，并只读取 checkpoint 的
`model.pt`。`PROBE_GPUS=3` 可让8卡预训练 run 只用3卡做 probe；
`PROBE_DEVICES=cuda:1,cuda:3` 可指定卡号。formal 完成后 Bash 入口继续生成6个 router
analysis；设置 `RUN_ROUTER_ANALYSIS=0` 才会跳过。

## 一键正式全流程

确认模型配置、磁盘和 GPU 后，可以串行完成两个数据条件、三阶段 probe、router 和比较：

```bash
CONFIRM_FULL_EXPERIMENT=1 \
  bash scripts/bash/synbios_full_experiment.sh single

CONFIRM_FULL_EXPERIMENT=1 NPROC=4 \
  bash scripts/bash/synbios_full_experiment.sh ddp
```

入口内部严格执行 `single/multi5_permute pretrain → smoke → pilot → formal → router →
comparison`。中断后重跑同一命令会恢复主训练，并只复用 identity 完全一致的 probe 任务。

## 监控与恢复

`prepare/evaluate/probe/probe-pipeline/analyze` 与主训练共用结构化监控协议，默认写终端、JSONL 和
TensorBoard。可用 `--log-dir PATH`、`--log-interval N`、`--no-tensorboard`、`--quiet`
控制。pipeline 另外提供任务队列状态、失败数和 `--heartbeat-seconds` 心跳；单 probe 记录
accuracy、逐位置 accuracy、grad norm、data wait、GPU利用率和 step time，并周期保存不含
backbone的轻量恢复点。正式运行前可用 `scripts/bash/synbios_probe_batch_benchmark.sh`
在四卡上回归 P/Q 的训练与验证 batch。多卡 checkpoint 的
`distributed/` 用于恢复主训练，`model.pt` 用于任意单卡 evaluate/probe；probe 不加载主训练 Adam。完整字段见
[`docs/training/monitoring_and_recovery.md`](../../docs/training/monitoring_and_recovery.md)。

## 产物

```text
artifacts/synbios_moe/
├── single/ 与 multi5_permute/      # profile、bio、token shards、probe_cache
├── checkpoints/<run>/              # DCP + Adam + model.pt
├── runs/<run>/                      # JSONL/TensorBoard 日志
└── results/<run>/probe_pipeline/    # gate、阶段训练、独立val、summary CSV/JSON
```

判断顺序应是：主训练 loss → attribute accuracy → Q-probe → P-probe → router。预训练尚未
学会事实时，不应把低 probe accuracy 解释为“知识组织方式不同”。
