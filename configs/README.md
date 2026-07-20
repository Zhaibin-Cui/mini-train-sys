# 配置系统说明

## 先理解两类 YAML

- `model_*.yaml` 或 `synbios_moe/model.yaml`：只描述 Transformer 结构。
- run YAML：描述数据、训练、优化器、并行、日志和 checkpoint。

`scripts/train.py` 同时接收两者。不要把模型尺寸复制进每个 run，否则对比 DDP/FSDP
时很容易出现模型不一致。

## `extends` 如何合并

run YAML 支持一个或多个相对路径父配置：

```yaml
extends:
  - ../base.yaml
  - ../strategies/fsdp.yaml
  - ../topologies/8gpu.yaml
run:
  name: my_fsdp_8gpu
```

按列表顺序递归合并，后者覆盖前者，最后由当前文件覆盖。相对路径始终相对于当前
YAML 所在目录。循环继承会报错。

## 顶层配置段

| 段 | 负责什么 |
|---|---|
| `run` | 名称与随机 seed |
| `backend` | `torch`、`triton`、`cuda` 算子 backend |
| `parallel` | single/DDP/FSDP 与固定 world size |
| `optimizer` | AdamW 参数 |
| `lr_scheduler` | constant/cosine、warmup、decay |
| `train` | 每 rank batch、epoch/step、精度、clip |
| `checkpoint` | 保存频率、保留数量、恢复、模型导出 |
| `data` | 数据源、packing、worker、prefetch |
| `logging` | console、JSONL、TensorBoard |

所有字段最终由 `minitrain/runtime/config.py` 的 dataclass 校验；拼错字段不会被默默
接受，而会在构造配置时报错。

## Batch、epoch 与学习率

```yaml
train:
  batch_size: 8                 # 每 rank；不是 global batch
  epochs: 540
  max_steps: null
  reference_global_batch_size: 96
  batch_size_scaling: linear
  log_interval: 10              # 每多少 optimizer step 记录一次完整状态
  check_finite: true            # loss 为 NaN/Inf 时立即失败
  grad_clip_norm: 5.0           # 全局 L2 norm 安全阈值；null 表示关闭
```

本项目没有梯度累计：

```text
actual_global_batch = batch_size × WORLD_SIZE
```

`linear` 时，运行时按 `actual/reference` 缩放 LR，并反向缩放 warmup/显式 decay step。
epoch 不随卡数改变，所以文档/人物曝光次数保持不变。普通实验不需要线性缩放时使用
`batch_size_scaling: none`。

## 数据与 worker

```yaml
data:
  source: token_shards           # random | tokens | token_shards
  path: path/to/manifest.json
  packing: randomized_documents  # contiguous | randomized_documents
  num_workers: null              # null=自动；整数=每 rank 固定值
  worker_budget: 32              # 单机所有 rank 的总预算
  max_workers_per_rank: 4
  worker_cpu_affinity: true
  prefetch_factor: 2
  pin_memory: true
  persistent_workers: true
  drop_last: true
```

自动模式先为每个 trainer rank 保留 CPU，再在节点预算内分配 worker，并限制每 rank
上限。Linux worker 会单线程并尝试 CPU affinity。CPU/debug 配置应显式设
`num_workers: 0`、`persistent_workers: false`。

`randomized_documents` 只适用于带 `documents.idx` 的 token-shard manifest；它每 epoch
重排完整文档后再打包固定长度 block。`contiguous` 直接把整个 token 流按位置切块。

## Single、DDP 与 FSDP

```yaml
parallel:
  strategy: fsdp
  process_group_backend: nccl
  expected_world_size: 8
  fsdp:
    sharding_strategy: full_shard
    auto_wrap_policy: transformer_block
    backward_prefetch: backward_pre
    forward_prefetch: false
    limit_all_gathers: true
    use_orig_params: true
    sync_module_states: true
    cpu_offload: false
    activation_checkpointing: false
```

`expected_world_size` 是固定服务器 preset 的防误用检查。FSDP 默认每个
`TransformerBlock` 一个 unit；`auto_wrap_policy: none` 只用于诊断。各字段的执行语义
见 [distributed_training.md](../docs/training/distributed_training.md)。

## Checkpoint

```yaml
checkpoint:
  directory: checkpoints
  every_epochs: 1
  keep_last: 2
  keep_safety: 1               # 额外保留较老的完整DCP+Adam锚点
  safety_every_epochs: 10      # 安全锚点只从10的倍数epoch选择
  keep_model_exports: 1       # 只有最新目录保留重复的model.pt
  save_final: true
  resume_from: null              # null | latest | safety | 显式目录
  export_model: true             # 为 evaluate/probe 写 model.pt
  export_model_every_epochs: 50  # 可选：减少昂贵的完整模型聚合；最终保存仍强制导出
  cpu_offload: true
```

`export_model` 不等于训练 checkpoint：训练恢复读取 DCP 模型和 Adam；`model.pt` 只是
完整权重导出。大模型不做 probe 时可关闭导出，降低 rank-0 CPU 峰值和磁盘占用。
`keep_last + keep_safety` 是 checkpoint 目录数的硬上限；安全目录仍可完整恢复训练，只是
没有 `model.pt`，因此不能直接交给 probe。`--resume safety` 会选择带 `SAFETY` 标记的锚点。

## 当前 preset

| 路径 | 用途 |
|---|---|
| `train_debug.yaml` | CPU 最小训练 |
| `train_single.yaml` | 24 GB RTX 4090 单卡兼容入口 |
| `train_ddp.yaml` / `train_fsdp.yaml` | 默认 8 卡兼容入口 |
| `server/rtx4090_24gb/runs/` | 显式 single/DDP/FSDP × 1/4/8 卡矩阵 |
| `smoke/` | DDP/FSDP/checkpoint 极小验证 |
| `synbios_moe/` | base、variant、strategy 和具体 run 分层 |

服务器上优先使用显式 `server/.../runs/*.yaml`，而不是依赖兼容别名。

`synbios_moe/probe_pipeline.yaml` 是正式 `smoke → pilot → formal` probe 预算；
`synbios_moe/probe_pipeline_notebook_smoke.yaml` 仅供端到端 notebook 用 3 steps 验证
调度、独立 validation、监控和汇总链路，不能用于论文结果。
