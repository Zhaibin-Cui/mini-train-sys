# Single、DDP 与 FSDP 训练手册

这三个策略共用同一个模型、DataLoader、Trainer、epoch 预算和 checkpoint
接口。区别只发生在 `minitrain/distributed/`：进程组如何建立、模型如何包裹、
参数/梯度如何跨卡同步。

第一次阅读建议按第 2 节策略对比 → 第 4 节 batch/epoch → 第 5 节 checkpoint →
第 6 节命令阅读；理解主流程后再看第 1/3 节 FSDP 细节。

## 1. `auto_wrap_policy` 到底是什么

FSDP 不只是“是否分片”，还必须决定“以哪个子模块为一个分片执行单元”。
本项目默认：

```yaml
parallel:
  strategy: fsdp
  fsdp:
    auto_wrap_policy: transformer_block
```

`minitrain/distributed/fsdp.py::build_auto_wrap_policy` 返回
`ModuleWrapPolicy({TransformerBlock})`。若模型有 12 个 block，结构近似：

```text
root FSDP
├── embeddings / final norm / lm_head
├── block 0 FSDP
├── block 1 FSDP
├── ...
└── block 11 FSDP
```

执行 block 0 前收集它的参数，计算后释放；接着处理 block 1。反向过程按相反
顺序进行，并可通过 `backward_prefetch` 提前收集下一层。这样参数显存峰值约束在
少数 block，而不是整个模型，同时通信可以与计算重叠。

若设置 `auto_wrap_policy: none`，只有 root FSDP。参数仍被 FULL_SHARD 分片，
但一次需要收集整个模型，不能形成逐层通信流水，显存峰值和吞吐通常都更差。

共享的 embedding/lm-head 留在 root，而不会被两个独立 FSDP 单元重复管理，适合
`tie_word_embeddings: true`。

## 2. 三种策略的语义

| 策略 | 每张卡保存的参数 | 梯度同步 | 适用场景 |
|---|---|---|---|
| single | 完整参数与 Adam | 无 | 单卡、调试 |
| DDP | 完整参数与 Adam | backward 时 all-reduce | 模型单卡放得下，优先吞吐 |
| FSDP FULL_SHARD | 参数、梯度、Adam 都分片 | 按 block all-gather/reduce-scatter | 模型或 Adam 单卡放不下 |
| FSDP SHARD_GRAD_OP | forward 后保留参数，梯度/Adam 分片 | 较少参数重建 | 显存更宽裕、换吞吐 |

DDP/FSDP 都是一进程一卡。Linux/NCCL 是正式多卡路径；Gloo 只用于 CPU/DDP
烟雾测试。FSDP 明确拒绝 CPU，因为当前 PyTorch FSDP 需要 accelerator。

## 3. FSDP 配置说明

- `sharding_strategy`: 默认 `full_shard`，显存最省；`shard_grad_op` 通信较少。
- `auto_wrap_policy`: 默认 `transformer_block`，生产训练不建议改为 `none`。
- `backward_prefetch`: 默认 `backward_pre`，用更多瞬时显存换通信重叠；OOM 时可用
  `backward_post` 或 `none`。
- `forward_prefetch`: 静态执行图可能获益；默认关闭以避免额外峰值。
- `limit_all_gathers`: 默认开启，限制 CPU 线程过早发起 all-gather，抑制显存峰值。
- `use_orig_params`: 默认开启，保留原参数视图，优化器和冻结策略更易理解。
- `sync_module_states`: 默认开启，从 rank 0 广播初始化状态。
- `cpu_offload`: 默认关闭；开启会显著增加 PCIe 流量，仅在显存确实不足时使用。
- `activation_checkpointing`: 默认关闭；开启后对每个 `TransformerBlock` 使用
  non-reentrant activation checkpoint，以额外重算换激活显存。
- FSDP mixed precision：bf16/fp16 参数计算和 buffer 使用低精度，reduce 使用 fp32；
  fp16 仍由 GradScaler 管理。

24 GB RTX 4090 运行当前 SynBio MoE 时，默认从每卡 batch 8 开始，并以容量
benchmark 测得的稳定上限为准。多卡且模型单卡
放得下时优先 DDP；FSDP 主要用于验证扩展性或后续更大模型。OOM 调整顺序建议：
减小每卡 batch → 开 activation checkpointing → 改 prefetch → 最后才 CPU offload。

## 4. Epoch、公平预算与 batch

本项目没有梯度累计，一批数据对应一次 optimizer step：

```text
global_batch = per_rank_batch × WORLD_SIZE
```

SynBio 的 single 与 multi5+permute 分别固定 540 和 108 epochs。切换并行策略不会
改变 epoch，因此每个人和每条 biography 的曝光次数不变。不同卡数会改变 optimizer
step 数和梯度噪声，这是“不用梯度累计”的必然结果。

共享 base config 把 Allen-Zhu 参考点写成 global batch 96、peak LR 1e-3、warmup
1000 reference steps。`batch_size_scaling: linear` 在启动时计算：

```text
lr = 1e-3 × actual_global_batch / 96
warmup_updates = round(1000 × 96 / actual_global_batch)
```

例如 single batch 8 得到 LR 8.3333e-5、warmup 12000 updates；8 卡、每卡 batch
8 得到 global batch 64、LR 6.6667e-4、warmup 1500。日志同时记录 reference 和
resolved 数值，避免 YAML 与真实运行参数混淆。

## 5. 分布式 checkpoint

旧实现由 rank 0 调用普通 `model.state_dict()` 写一个 `.pt`，对真正的 FSDP 不安全：
它可能只得到局部分片，也可能在 rank 0 聚合时造成内存峰值，并且 Adam 状态无法
自然重分片。现在所有 rank 都参加 PyTorch Distributed Checkpoint (DCP)。

目录结构：

```text
checkpoints/<run>/epoch_000001_step_000012345/
├── distributed/              # DCP 模型 + Adam 分片和 metadata
├── runtime.pt                # trainer/scheduler/scaler/config/precision
├── rng_rank_00000.pt         # 每个 rank 独立 RNG
├── rng_rank_00001.pt
├── model.pt                  # 可选完整模型，仅供 evaluate/probe
└── COMMITTED                 # 最后写入；没有它就不是有效 checkpoint
```

写入先发生在同级临时目录；所有 DCP 分片、sidecar 和可选模型导出成功后，rank 0
写 `COMMITTED` 并原子重命名。`latest` 只选择 committed 目录，不会恢复半写文件。

训练恢复调用 DCP 的 canonical model/optimizer state-dict API，因此可在 single、DDP、
FSDP 之间以及不同 world size 间重新分片。world size 改变时，epoch-boundary resume
会把 LR 进度映射为 `completed_epoch × new_steps_per_epoch`，而 Adam 内部 update counter
保持真实历史值。world size 相同时还会精确恢复每个 rank 的 Python/Torch/CUDA RNG；
改变 world size 后，新出现的 rank 没有旧 RNG sidecar，模型和 Adam 仍可恢复，但不能
声称 bitwise continuation。

`checkpoint.export_model: true` 会额外生成 CPU `model.pt`。evaluate、P/Q probe 和
router analysis 只 mmap/load 这个文件，不读取 DCP 中的 Adam tensor。完整模型导出
需要一次 collective，并占用 rank-0 CPU 内存；不做 probe 的大模型可关闭它。

正式服务器 preset 每个 epoch 保存，但只保留“最新2个 + 较老安全锚点1个”。只有最新
目录保留 `model.pt`；另两个仍有完整 DCP+Adam，可用 `--resume safety` 回退。安全锚点
带 `SAFETY` 标记，默认从每10 epoch milestone 中轮换，因此不会累计保存所有 milestone。

旧 `.pt` checkpoint 仍可恢复，作为向后兼容路径。

## 6. 启动命令

通用：

```bash
bash scripts/bash/train.sh
NPROC=4 bash scripts/bash/distributed.sh ddp
NPROC=8 bash scripts/bash/distributed.sh ddp
NPROC=4 bash scripts/bash/distributed.sh fsdp
NPROC=8 bash scripts/bash/distributed.sh fsdp
```

SynBio 每次只启动一个“数据变体 × 并行策略”，输出 run name 相互隔离：

```bash
bash scripts/bash/synbios_moe.sh single single
NPROC=8 bash scripts/bash/synbios_moe.sh single ddp
NPROC=8 bash scripts/bash/synbios_moe.sh single fsdp
NPROC=8 bash scripts/bash/synbios_moe.sh multi5_permute fsdp
```

脚本发现 committed checkpoint 时默认 `--resume latest`；设置 `AUTO_RESUME=0` 从头
开始。`FORCE_PREPARE=1` 会重新生成数据，已有 checkpoint 时应先人工归档，避免把
旧模型/Adam 状态恢复到不同语料。

## 7. 服务器验收

代码合入前的本地门槛：Python 静态检查、配置/auto-wrap/batch-scaling 单测、单进程
训练、单进程 DCP save/load。Linux CI 还会执行二进程 Gloo DDP → single reshard 测试。

真实多卡服务器必须执行：

```bash
# 两步 FSDP + DCP 保存（固定服务器 launcher 支持 1/4/8 卡）
NPROC=4 CONFIG=configs/smoke/fsdp_cuda.yaml \
  MODEL_CONFIG=configs/model_debug_dense.yaml bash scripts/bash/distributed.sh fsdp

# 从刚才的完整 checkpoint 恢复，再跑两步
NPROC=4 CONFIG=configs/smoke/fsdp_cuda.yaml RESUME=latest \
  MODEL_CONFIG=configs/model_debug_dense.yaml bash scripts/bash/distributed.sh fsdp
```

验收项：所有 rank 均完成；模型中每个 `TransformerBlock` 都被独立 FSDP 包裹；
checkpoint 有 `COMMITTED`、DCP metadata、每-rank RNG 和 `model.pt`；恢复后的 step/
Adam/LR 连续；无 NCCL timeout、OOM 或 rank-0-only collective hang。生产长跑前还应在
目标 GPU 上记录峰值显存与 tokens/s，再决定 prefetch 和 activation checkpointing。

实现基于 PyTorch 官方 FSDP 与 Distributed Checkpoint 接口：

- https://docs.pytorch.org/docs/stable/fsdp.html
- https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
- https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
