# 训练监控、checkpoint 与恢复

这份文档回答三个实际问题：训练时应该看什么、中断后怎样继续、多卡训练出的模型怎样交给单卡评估和 probe。

## 1. 一条训练进度代表什么

`TrainingRunner` 每隔 `train.log_interval` 个 optimizer step 输出一行，例如：

```text
[train] batch 100/10000 | epoch 2/5 | loss 2.10431 | lr 3.000e-04 | tok/s 82,410.0 | gpu(max) 18.20/24.00 GiB | 21.0% | ETA 0:42:18
```

这里一个 batch 就是一个 optimizer update；本项目没有梯度累计。主要字段如下：

| 字段 | 含义 |
|---|---|
| `batch/batches_total` | 当前 epoch 内 DataLoader 进度 |
| `step/step_total` | 整个实验的 optimizer update 进度 |
| `loss` | 当前日志区间、所有 rank 的平均总 loss |
| `lr` | 当前实际学习率；多卡 batch scaling 后的值也反映在这里 |
| `tokens_seen/tokens_goal` | 全局已训练 token 与计划 token |
| `tokens_per_sec` | 这段日志区间内所有 GPU 合计吞吐 |
| `avg_tokens_per_sec` | 本次进程启动以来的平均总吞吐 |
| `data_wait_ms` | 每 step 等待 DataLoader 的平均时间 |
| `step_time_ms` | 包括取数、forward、backward、optimizer 和必要同步的平均时间 |
| `gpu_memory_allocated_mb_max` | 所有 rank 中最高的实际 tensor 显存 |
| `gpu_memory_allocated_mb_total` | 当前作业所有 rank 的 tensor 显存之和 |
| `gpu_memory_reserved_mb_max/total` | PyTorch allocator 保留显存的最大值/总和 |
| `gpu_peak_memory_allocated_mb_max` | 从本轮运行开始的 rank 峰值中的最大值 |
| `grad_norm` | clip 前的全局梯度范数；可用于发现爆炸 |
| `grad_clip_threshold` | 配置的全局梯度裁剪阈值；正式 preset 默认为 5.0 |
| `grad_clip_coefficient` | 本 step 梯度实际乘数；1.0 表示没有裁剪 |
| `grad_was_clipped` | 本 step 是否触发裁剪，0 或 1 |
| `grad_clip_fraction` | 当前日志区间内触发裁剪的 step 比例 |

`loss`、学习率、MoE router 指标和梯度指标在分布式训练中会先跨 rank 聚合。显存同时保留“单个最坏 rank”和“整个作业总和”，因此不会把 8 卡总显存误报成一张卡的使用量。

训练指标不是只取打印时的最后一个 batch。`TrainingRunner` 会先在设备上累计整个
`log_interval`，到日志点再求区间平均并跨 rank 聚合；这样不会每一步同步 CUDA，也避免
MoE 专家分布被一个偶然 batch 误导。

`grad_clip_norm: 5.0` 被定位为防止异常尖峰的安全阀，而不是常态正则化。若
`grad_clip_fraction` 长期超过 5%，应先检查学习率、数据和数值稳定性；超过 20% 时不建议
直接继续正式实验。框架不会动态改变阈值，以保证恢复训练和论文对照可复现。

## 2. 日志存在哪里

配置中的 `logging` 同时控制三种输出：

```yaml
logging:
  console: true
  tensorboard: true
  jsonl: true
  log_dir: artifacts/runs
  flush_secs: 10
```

每次启动会创建 `log_dir/<run-name>/<timestamp>/`：

- `events.jsonl` 是完整、便于脚本审计的结构化事件；
- `events.out.tfevents.*` 是 TensorBoard 数据；
- 终端只显示最常用的一行摘要。

启动 TensorBoard：

```bash
tensorboard --logdir artifacts/runs
```

常看的曲线是 `train/loss`、`train/lr`、`train/tokens_per_sec`、`train/data_wait_percent`、`train/gpu_memory_allocated_mb_max`、`train/gpu_memory_allocated_mb_total` 和 `train/grad_norm`。`train/check_finite: true` 会在 loss 变成 NaN/Inf 时立即终止，避免继续写坏 checkpoint。

### Dense 与 MoE loss

Dense 和 MoE 都记录：

| TensorBoard tag | 含义 |
|---|---|
| `train/loss/lm_cross_entropy` | 不含 router 正则项的纯 next-token CE |
| `train/loss/total` | optimizer 实际反向传播的总 loss |

MoE 额外记录：

| TensorBoard tag | 含义 |
|---|---|
| `train/moe/aux_loss` | 未乘系数的负载均衡 loss |
| `train/moe/z_loss` | 未乘系数的 router z-loss |
| `train/loss/moe_aux_weighted` | `aux_loss × router_aux_loss_coef` |
| `train/loss/moe_z_weighted` | `z_loss × router_z_loss_coef` |
| `train/loss/moe_regularization_total` | 两项加权 router loss 之和 |

因此应满足：

```text
loss/total = loss/lm_cross_entropy + loss/moe_regularization_total
```

不要只看总 loss 判断模型是否学会 biography；论文对照首先看纯 CE，再分别判断 router
正则是否异常放大。

### 专家分布可视化

MoE 训练会生成两组 `layer × expert` 矩阵：实际 Top-k 选中比例，以及完整 router softmax
概率。TensorBoard 对每组数据提供：

- 每个 expert 的独立标量曲线，保留 expert 身份；
- 相对均匀负载的 ratio histogram，观察整体离散程度；
- 固定色标 heatmap：行是 layer，列是 expert，蓝色表示低于均匀负载，白色表示均衡，
  红色表示高于均匀负载；颜色范围固定为 `0×～2×` 均匀负载，跨 step 可以直接比较。

主要 tag：

```text
train/moe/expert_load_fraction/expert_00
train/moe/expert_probability/expert_00
train/moe/expert_load_fraction_by_layer/balance_heatmap
train/moe/expert_load_fraction_by_layer/ratio_histogram
train/moe/expert_probability_by_layer/balance_heatmap
train/moe/expert_probability_by_layer/ratio_histogram
```

同时关注 `expert_load_cv`、`max_to_mean_load`、`dead_expert_count`、标准化 router entropy。
热力图接近白色只表示总负载均衡，不表示 experts 学到相同内容；跨独立 run 的 expert 编号
存在置换对称，不能直接把 run A 的 expert 0 与 run B 的 expert 0 对齐。JSONL 中保留原始
二维数值，便于后续绘制定制图。

SynBioS 的 `prepare`、`evaluate`、`probe`、`probe-pipeline` 和 `analyze` 使用同一事件协议。它们默认启用 TensorBoard，可用 `--log-dir` 指定目录、`--log-interval` 控制频率、`--no-tensorboard` 关闭、`--quiet` 关闭终端行。probe 的优化器只属于 probe，不会加载主训练 Adam。probe 额外记录区间/逐位置准确率、全局梯度范数、数据等待占比和 step 时间；这里的梯度范数仅监控，不执行 clipping。pipeline 的详细任务状态和日志路径见 [SynBioS probe pipeline](../experiments/synbios_moe_probe_pipeline.md#9-probe-训练监控)。

## 3. checkpoint 里保存了什么

每个完成的 checkpoint 是一个目录：

```text
epoch_000005_step_000012340/
  COMMITTED
  distributed/          # DCP：模型 + Adam，可跨 world size 重分片
  runtime.pt            # step/epoch/token、LR scheduler、GradScaler、完整配置
  rng_rank_00000.pt     # Python/Torch/CUDA RNG；每个训练 rank 一份
  rng_rank_00001.pt
  model.pt              # 仅最新目录保留；供单卡推理/probe
  SAFETY                # 仅较老安全锚点存在；仍含完整DCP+Adam
```

只有出现 `COMMITTED` 才表示写入完成。`resolve_resume_checkpoint(..., "latest")` 会忽略临时目录和没有该标记的目录。

训练恢复读取 `distributed/` 和 `runtime.pt`，恢复模型、Adam、scheduler、scaler、计数器和可匹配 rank 的 RNG。评估、生成和 probe 调用 `load_model_state_dict_from_checkpoint()`，只读取 `model.pt`，所以不会把 Adam 搬进显存。

生产配置应保持：

```yaml
checkpoint:
  every_epochs: 1
  keep_last: 2
  keep_safety: 1
  safety_every_epochs: 10
  keep_model_exports: 1
  save_final: true
  export_model: true
```

该策略最多保留3个可恢复目录：最新2个和1个较老安全锚点。每次保存仍先完整提交新目录，
再清理旧目录，因此写入中断不会先删掉已有恢复点。为了节省空间，只有最新目录保留
重复的 `model.pt`；另外两个目录的 DCP 模型、Adam、scheduler、RNG 都完整保留。

`export_model: true` 是“多卡训练、单卡推理”的必要条件，但不是恢复训练的必要条件。

## 4. 怎样恢复

单卡、DDP、FSDP 使用同一个入口：

```bash
python scripts/train.py --config CONFIG.yaml --model-config MODEL.yaml --resume latest
```

如果最新两个 epoch 损坏、数值发散或不值得继续，直接恢复安全锚点：

```bash
python scripts/train.py --config CONFIG.yaml --model-config MODEL.yaml --resume safety
```

`safety_every_epochs: 10` 表示优先保留最近一个已经离开滚动窗口的10倍数 epoch；训练
初期还没有这样的 milestone 时，会临时保留最早的 committed checkpoint，避免没有后路。

或者传入明确目录：

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --config CONFIG.yaml --model-config MODEL.yaml \
  --resume artifacts/checkpoints/RUN/epoch_000005_step_000012340
```

所有 rank 必须传相同路径并共同参加 DCP load。world size 改变时，DCP 会把模型和 Adam 重新分片；epoch/token/LR 进度按新 DataLoader 长度重新表达。相同 world size 且每 epoch checkpoint 的恢复可以延续 RNG；新增 rank 没有旧 RNG 文件时会使用确定性派生 seed，模型与 Adam 仍可恢复，但不承诺逐 bit 重放。

推荐只从 epoch checkpoint 改变 world size，因为它位于 sampler 的完整边界。若从一次中途 `save_final` 恢复，当前实现从该 epoch 的开头重新建立 sampler，而不是保存 batch 游标。

## 5. 验证入口

- `tests/test_checkpoint_contract.py`：单进程完整状态恢复，以及纯模型读取不碰 Adam；
- `tests/test_distributed_checkpoint.py`：DDP checkpoint 重分片到单进程，并用 `model.pt` 单卡加载；
- `tests/test_runtime_logger.py`：终端格式、JSONL、TensorBoard 标量和 probe/pipeline 摘要字段；
- `tests/test_synbios_moe.py`：probe 健康指标、任务心跳和 notebook 调用链契约；
- `tests/synbios_moe_end_to_end.ipynb`：数据准备、主训练、评估、单 probe、独立 validation、P/Q pipeline、事件落盘、router、结果汇总和继续训练。

本地没有 4/8 张 GPU 时，这些测试证明格式与 CPU/Gloo 路径；服务器上的真实 NCCL/FSDP 吞吐和显存门禁仍应运行 `tests/distributed_server_benchmark.ipynb`。
