# 架构与代码边界

## 一张图看懂依赖方向

```text
scripts/train.py
  ├─ runtime/config.py        YAML → typed ExperimentConfig
  ├─ runtime/factory.py       构造 backend / model / parallel strategy
  ├─ data/dataloader.py       构造 dataset + sampler + DataLoader
  ├─ train/optim.py           构造 AdamW
  ├─ train/trainer.py         完成一个 optimizer update
  └─ train/runner.py          epoch、日志、停止条件、checkpoint
          └─ train/checkpoint.py  PyTorch DCP 保存/恢复

MiniTransformer
  ├─ model/blocks.py          Attention、Dense FFN、MoE FFN、TransformerBlock
  ├─ model/moe_router.py      Top-k 路由
  └─ model/ops.py             OpsBackend 协议
       ├─ kernels/torch_ops.py
       ├─ kernels/triton/
       └─ kernels/cuda_ext/

ParallelStrategy
  ├─ distributed/single.py
  ├─ distributed/ddp.py
  └─ distributed/fsdp.py
```

依赖只应从上往下。模型不知道 YAML、torchrun 或 checkpoint；Trainer 不知道 Dense
还是 MoE；DDP/FSDP 不进入 Transformer 的 forward。这些边界让同一个实验可以只替换
一个维度，而不复制整套训练代码。

## 六个核心抽象

### `ExperimentConfig`

`runtime/config.py` 把 YAML 转成 dataclass，并在启动前验证枚举、正数范围和互斥条件。
`load_yaml_dict()` 先递归处理 `extends`，后加载的父配置和子配置覆盖前面的字段。

### `MiniTransformer`

`model/transformer.py` 是唯一主干。`ModelConfig.ffn_type` 决定每个 block 使用
`DenseFeedForward` 还是 `MoEFeedForward`。`MiniMoETransformer` 只是兼容别名，不是
第二套模型。

### `OpsBackend`

`model/ops.py` 定义 RMSNorm、RoPE、SwiGLU、attention、CE、router 和 fused MoE 的
统一接口。Torch 是正确性基线；Triton/CUDA 在支持当前 shape/device/dtype 时走优化
kernel，否则回退到 Torch。调用方不为每个 kernel 写分支。

### `ParallelStrategy`

策略负责 process group、模型包装、barrier 和 teardown。Single 不通信；DDP 复制完整
模型并 all-reduce 梯度；FSDP 按 TransformerBlock 分片参数/梯度/Adam。

### `Trainer` 与 `TrainingRunner`

`Trainer.train_step()` 只完成一个 batch：H2D、forward、backward、clip、optimizer、
LR 和计数器。`TrainingRunner` 管理 epoch、sampler `set_epoch()`、日志、停止条件和
checkpoint。区分这两层能让单步 benchmark 不必启动完整实验。

### DCP checkpoint

`train/checkpoint.py` 使用 PyTorch Distributed Checkpoint。所有 rank 参加 collective，
rank 0 最后发布 `COMMITTED`。训练恢复读取模型、Adam、scheduler、scaler、计数器和
RNG；评估只读取可选的 `model.pt`。

### `EventLogger` 与 `ProgressReporter`

`runtime/logger.py` 只负责把同一个 event 写到终端、JSONL 和 TensorBoard；
`runtime/monitoring.py` 负责进度、ETA、吞吐和跨 rank 显存/标量/小型矩阵聚合。训练 Runner、SynBioS
评估和 probe 复用这层，因此实验代码不需要各写一套计时与显存统计。

模型通过 `last_training_metrics` 发布标量，通过 `last_training_visualizations` 发布与架构
有关的小型矩阵；旧的 `last_moe_*` 名称保留为兼容别名。`Trainer` 只复制本 step 产物，
`TrainingRunner` 在日志区间内累积并跨 rank 求平均，`TensorBoardLogger` 决定标量、
histogram 或固定色标 heatmap 的展示方式。Dense 模型不产生 MoE 矩阵，接口无需分支。

SynBioS 仍把论文特有的 data/probe/evaluation/router 放在 `experiments/synbios_moe/`，
`scripts/synbios_moe.py` 只做参数解析、模型装配和命令编排；probe 任务、运行 identity、
原子状态、心跳和结果归并由 `experiments/synbios_moe/probe_pipeline.py` 负责。这样 CLI 不再
自己维护一套线程状态机。端到端调用由
`tests/synbios_moe_end_to_end.ipynb` 分阶段验证。

## 数据层为什么单独设计

`data/documents.py` 处理人类文本边界；`data/preprocess.py` 处理 tokenizer 和物理 shard；
`data/dataloader.py` 处理训练 block 和 rank 分片。逻辑 document/chunk 与物理 shard
不是同一个概念，拆开后才能既保留文档边界，又控制文件容量和随机 I/O。

## 扩展一个功能应该改哪里

| 需求 | 首选位置 |
|---|---|
| 新模型尺寸 | 新增 model YAML |
| Dense/MoE 新结构 | `model/blocks.py` 与 `ModelConfig` |
| 新算子实现 | 新 backend method/kernel，并对 Torch 基线测试 |
| 新并行策略 | `distributed/` 实现 `ParallelStrategy` |
| 新数据格式 | `data/documents.py` reader |
| 新采样/packing | `data/dataloader.py` Dataset/Sampler |
| 新训练调度 | `train/runner.py`，不要塞进模型 |
| 新实验 | `experiments/<name>/` + 薄 CLI |

## 当前明确没有实现的部分

- 通用 `scripts/eval.py` 与 `scripts/sample.py` 仍是 scaffold；
- 没有 tensor/pipeline/expert parallel；当前 MoE experts 在每个 DDP rank 内完整存在；
- 没有梯度累计；
- FSDP 是 FSDP1 API，不是 composable FSDP2；
- 单机 1/4/8 卡是当前生产目标，多节点尚未做端到端验收。

这份边界清单用于区分“代码已有能力”和“未来可扩展方向”。
