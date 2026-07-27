# 分布式策略目录

这个目录只负责“模型如何在进程间包装和同步”，不负责 epoch、数据或 checkpoint。

| 文件 | 当前作用 |
|---|---|
| `strategy.py` | `ParallelStrategy` 协议 |
| `single.py` | 无通信基线 |
| `ddp.py` | 一进程一卡、梯度 all-reduce |
| `fsdp.py` | FULL_SHARD/SHARD_GRAD_OP、block auto-wrap、prefetch |
| `custom_allreduce.py` | 教学用 all-reduce，不是正式训练策略 |
| `comm_hooks.py` | 通信 hook 实验位置 |

调用顺序是 `setup()` → `wrap_model()` → 训练中的 `barrier()` → `teardown()`。optimizer
在 `wrap_model()` 之后创建。DDP/FSDP 数据分片不在这里做，而在
`minitrain/data/dataloader.py` 的 sampler 中完成。

当前生产目标是 Linux/NCCL 单机 1/4/8 卡。完整说明见
[`docs/training/distributed_training.md`](../../docs/training/distributed_training.md)。
