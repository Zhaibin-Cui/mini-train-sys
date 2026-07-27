# 混合精度运行手册

> 文件名保留 `plan` 以兼容旧链接；本文描述的是当前实现，不是待办清单。

## 配置选择

`train.precision` 支持 `auto | fp32 | bf16 | fp16`：

- CPU 的 `auto` → FP32；
- 支持 BF16 的 CUDA（包括 RTX 4090）→ BF16；
- 其他 CUDA → FP16。

服务器固定配置显式写 BF16，避免驱动或设备变化造成实验间 dtype 漂移。

## 哪些张量是什么精度

| 对象 | BF16/FP16 训练时 |
|---|---|
| token id、target | int64 |
| 普通 single/DDP 参数 | FP32 master parameter |
| Adam state | FP32 |
| residual/activation | BF16 或 FP16 |
| autocast GEMM | BF16 或 FP16 |
| norm/softmax/loss 归约 | FP32 累积 |
| loss 标量 | FP32 |
| FSDP reduce | FP32 |

embedding 后显式转为 activation dtype，因此 residual stream 从第一个 block 开始保持
低精度。RoPE cache 用目标 activation dtype 保存切片，但角度计算保持稳定精度。

## GradScaler 和 clipping

BF16 与 FP32 不启用 GradScaler。FP16 的顺序是：

```text
scaled loss backward → unscale optimizer gradients
→ clip true gradients → optimizer step → scaler update
```

在 unscale 前 clipping 会裁剪错误的数值尺度。FSDP 使用 wrapper 的
`clip_grad_norm_()` 计算跨 shard 全局 norm；single/DDP 使用标准 PyTorch utility。

## 自定义算子的 AMP 边界

`kernels/amp.py` 与各 public backend method 把浮点 activation 对齐到当前 CUDA
autocast dtype，RMSNorm weight 等应保留 FP32 的参数不做同样转换。自定义
`autograd.Function` 使用 `torch.amp.custom_fwd/custom_bwd`，保证 backward 看到一致的
autocast 状态。

Triton backend 对每个算子先做 capability check；不支持当前 dtype/shape/功能时回退
Torch。当前覆盖范围见 [`minitrain/kernels/README.md`](../../minitrain/kernels/README.md)。

## FSDP mixed precision

FSDP 的低精度策略由 `FSDPStrategy` 根据 resolved precision 构造：计算参数和 buffer
使用 BF16/FP16，梯度 reduce 使用 FP32。`use_orig_params: true` 保持 optimizer 和参数
视图更可理解。CPU offload 与 mixed precision 是两件不同的事，24 GB 4090 默认关闭
offload，避免 PCIe 成为瓶颈。

## Checkpoint

DCP 保存可恢复的模型/Adam 状态；FP16 额外保存 GradScaler。恢复时必须在 optimizer、
scheduler 和 scaler 都构造完成后一次性加载。`model.pt` 是评估用完整权重，不包含
Adam/Scaler。

## 验证清单

- 日志 `init` 事件记录 resolved precision 和 activation dtype；
- BF16 loss 为 FP32 且无 scaler；FP16 scaler 启用；
- 自定义 kernel 与 Torch 基线 forward/backward 对齐；
- 没有意外产生全尺寸 FP32 logits；
- DDP/FSDP 各 rank 无 dtype mismatch；
- checkpoint 恢复后的 step、LR、Adam 和 scaler 连续。

性能比较必须固定 dtype；把 FP32 Torch 与 BF16 Triton 直接比较不能归因于 kernel。
