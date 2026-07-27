# 项目全流程导读

本文以一条真实训练命令为主线，解释每一步调用了谁。先读这里，再进入各专题文档。

## 1. 命令选择两份配置

```bash
torchrun --standalone --nproc_per_node 4 scripts/train.py --device cuda \
  --config configs/server/rtx4090_24gb/runs/ddp_4gpu.yaml \
  --model-config configs/model_125m_moe.yaml
```

run YAML 决定数据、优化器、batch、并行和 checkpoint；model YAML 决定网络结构。
`scripts/train.py` 分别读取后，把 `model` 段合入同一个 `ExperimentConfig`。

## 2. 配置被解析和校验

`load_yaml_dict()` 递归解析 `extends`。例如 `ddp_4gpu.yaml` 由硬件 base、DDP strategy
和 4-GPU topology 合并。`experiment_config_from_dict()` 转成 typed dataclass。
`expected_world_size=4` 会防止错误的 torchrun 进程数。

## 3. 数据、模型和 backend 被构造

`build_training_dataloader()` 根据 `data.source` 选择随机 token、单 token 文件或
token-shard dataset。sampler 从环境变量读取 rank/world size，给各 rank 分配不重复的
样本。DataLoader worker 只准备 CPU batch。

`build_model()` 创建 `MiniTransformer`。每个 `TransformerBlock` 的 FFN 由
`ffn_type=dense|moe` 决定。模型只调用 `OpsBackend`；配置选择 Torch、Triton 或 CUDA。

## 4. 并行包装发生在 optimizer 之前

策略先初始化 process group，再包装模型：

- single：返回原模型；
- DDP：每 rank 一份完整模型；
- FSDP：root 加每个 `TransformerBlock` 成为 FSDP unit。

optimizer 必须在包装之后创建，特别是 FSDP，否则它可能持有错误的参数视图。

## 5. 一个 batch 如何训练

```text
DataLoader CPU tensor
  → pinned-memory non_blocking H2D
  → autocast forward
  → language-model loss + MoE auxiliary losses
  → backward
  → fp16 unscale（仅 fp16）
  → gradient clipping
  → AdamW step
  → zero_grad(set_to_none=True)
  → LR scheduler step
```

这里一个 batch 就是一次 optimizer update，没有梯度累计。Trainer 维护本 rank 的
`step/lr_step/tokens_seen/epoch`；日志展示时再乘 world size 得到全局 token。

## 6. Epoch 层做什么

`TrainingRunner` 在 epoch 开头调用 sampler `set_epoch(epoch)`，因此每轮顺序变化但可
复现。epoch 完成后按 `checkpoint.every_epochs` 保存。若 `max_steps` 在 epoch 中间停止，
该 epoch 不记为完成；恢复会从最后一个完整 epoch 重做这一轮。

## 7. Checkpoint 为什么是目录

FSDP 的模型和 Adam 天然是多 rank 分片，不能安全地只让 rank 0 调普通
`state_dict()`。DCP 让所有 rank 协同写 `distributed/`，再保存 runtime/RNG，最后写
`COMMITTED` 并原子发布目录。`--resume latest` 只选择 committed checkpoint。

## 8. SynBioS 在通用训练之外增加什么

`experiments/synbios_moe/data.py` 先生成固定 Profile，再渲染不同 biography 变体；
`scripts/synbios_moe.py prepare` 把它们转换成通用 token shards。主训练仍走完全相同的
`scripts/train.py`。

训练完成后：

- `evaluate`：属性 token 的 teacher-forced accuracy；
- `probe --kind p`：从属性位置 hidden state 读事实；
- `probe --kind q`：只给姓名，检查 person→fact 编码；
- `analyze`：统计 expert load、entropy 和属性/expert NMI。

Probe 冻结主干并创建自己的小 optimizer，因此只需 `model.pt`，不读取主训练 Adam。

## 9. 读代码的实际顺序

```text
scripts/train.py
runtime/config.py → runtime/factory.py
data/dataloader.py
model/transformer.py → model/blocks.py → model/ops.py
train/trainer.py → train/runner.py → train/checkpoint.py
distributed/{ddp,fsdp}.py
```

如果目标是 SynBioS，再按以下顺序追加：

```text
experiments/synbios_moe/data.py
scripts/synbios_moe.py
experiments/synbios_moe/{evaluation,probes,router_analysis}.py
```

下一步阅读：[数据流水线](../data/data_pipeline.md) 或 [架构与代码边界](architecture.md)。
