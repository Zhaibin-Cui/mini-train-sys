# 单机多卡 benchmark 说明

目标机器是一台主机、同一 CPU、1/4/8 张 24 GB RTX 4090。入口 notebook 是
[`tests/distributed_server_benchmark.ipynb`](../../tests/distributed_server_benchmark.ipynb)，
高层编排与可视化位于 `tests/distributed_bench_utils.py`，测量 worker 位于
`scripts/run_dist_bench.py`。Notebook 不再拼接长命令，而是使用与当前 kernel 相同的
`sys.executable` 创建一个 `DistributedBenchmark` 会话。

## Notebook 运行方式

服务器上从项目根目录启动 Jupyter：

```bash
jupyter notebook tests/distributed_server_benchmark.ipynb
```

Notebook 只有四个实际阶段：

```text
BenchmarkSettings + DistributedBenchmark
  → bench.preflight() + bench.inventory()
  → bench.run_weak().show()
  → bench.run_capacity().show()
```

8 卡机器使用 `world_sizes=(1, 4, 8)`，4 卡机器改为 `(1, 4)`。每个 raw case 都绑定
模型、展开后的运行配置、训练/kernel 源码指纹、PyTorch/CUDA/驱动/GPU 拓扑、batch、
步数和 repeat identity；中断后重跑会复用 identity 一致的成功 case，只重做缺失、失败、
代码/配置或服务器环境已经变化的 case。显式传入
`rerun_existing=True` 才会强制重跑。

## 两类实验

Weak scaling 固定每卡 local batch，卡数从 1 增至 4、8。global batch 和每 step
处理的 token 数随卡数等比例增加。理想情况下 step time 不变、总 tokens/s 线性增加：

```text
efficiency(N) = throughput(N) / (N × throughput(1))
```

报告以 80% 作为需要调查的门槛，而不是承诺值。PCIe 拓扑、NCCL 通信、FSDP
all-gather/reduce-scatter 和负载不均衡都会带来损耗。DDP 在每卡复制模型、梯度和
Adam；它扩展 global batch 容量，但不会提供 N 倍的单卡模型容量。FSDP 分片这些
状态，才可能随卡数提高 local batch 或模型容量。

Capacity sweep 将每个 batch case 放在独立 `torchrun` 子进程中，默认从 1 扫到 128。
OOM 被记录为结果而不是让整套实验停止，前一个 case 的 allocator 状态也不会污染
后一个 case。它覆盖真正的 single 1 卡、DDP 1/4/8 卡和 FSDP 1/4/8 卡（按机器实际
可见拓扑选择），并报告各策略/卡数下最大的成功 local/global batch。正式训练应选择
成功且 peak allocated 显存不超过 90% 的最大 local batch，而不是盲目使用最后一个
未 OOM 的点。

## 记录内容

- 10 个 warmup step、30 个测量 step、3 次独立重复（notebook 可调整）；
- 跨 rank 最大 step time，并给出 mean/P50/P95 和全局 tokens/s；
- DataLoader 等待 mean/P95、data stall 比例、每 rank/每节点 worker 数；
- 最忙单卡和全系统汇总的 allocated/reserved 峰值、单卡/全系统显存使用比例；
- GPU、CUDA、PyTorch、CPU 数、Git commit/dirty 状态和 `nvidia-smi topo -m`；
- 每个 case 的完整 stdout/stderr、raw JSON，以及聚合 summary JSON。

`BenchmarkReport.show()` 还会展示并落盘：

- repeats 聚合后的结果表；
- weak scaling 的吞吐、效率、延迟、显存和数据等待图；
- capacity 的最大成功 local batch 与显存曲线；
- weak quality gates 或 capacity frontier；
- OOM、超时和其他失败明细。

每个 suite 的可移植展示产物位于：

```text
<output>/<weak|capacity>/presentation/
├── results_aggregated.csv
├── failures.csv
├── quality_gates.json 或 capacity_frontier.json
└── weak_overview.png 或 capacity_overview.png
```

raw case、逐 case 日志、硬件清单和 summary JSON 保存在同一 suite 目录。重启 Notebook
后可用 `bench.load("weak").show()` 或 `bench.load("capacity").show()` 直接重建展示，无需
重新训练。

建议验收门槛是 weak scaling efficiency ≥80%、data stall ≤5%、稳定训练时 allocated
显存 ≤90%。门槛失败并不自动说明 DDP/FSDP 错误：先看 topology、降频、DataLoader
等待和专家负载，再区分通信瓶颈与计算瓶颈。

配置在 `configs/server/rtx4090_24gb/`。每个 1/4/8 卡 preset 都包含
`expected_world_size`，进程数不匹配会立即失败。默认 local batch=4 是安全 weak
scaling 起点；SynBio 正式训练的 24 GB preset 从每卡 batch=8 开始，最终仍应以目标
模型、目标 kernel 的 capacity sweep 为准。
