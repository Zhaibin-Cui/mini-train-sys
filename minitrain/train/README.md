# Train 模块

```text
trainer.py       一个 batch / 一个 optimizer update
runner.py        epoch、日志、停止条件、checkpoint cadence
optim.py         AdamW decay/no-decay 参数组
lr_scheduler.py  constant/cosine + warmup
precision.py     FP32/BF16/FP16 与 GradScaler
checkpoint.py    DCP 保存/恢复、RNG、model.pt 导出
```

`runner.py` 的日志同时记录全局 token 预算、吞吐、DataLoader 等待、梯度范数，以及多 rank
显存的最大值和总和。loss 与 MoE 专家矩阵先在日志区间内平均，再跨 rank 聚合，避免只
展示最后一个 batch。完整字段与恢复命令见
[`docs/training/monitoring_and_recovery.md`](../../docs/training/monitoring_and_recovery.md)。

这里没有梯度累计。`Trainer` 不知道 epoch，`TrainingRunner` 不实现 forward。optimizer
必须在 DDP/FSDP 包装模型之后创建。恢复顺序是构造模型/optimizer/scheduler/scaler，再
一次性加载 checkpoint，最后进入 runner。

详细时序见 [`docs/guides/project_walkthrough.md`](../../docs/guides/project_walkthrough.md)。
