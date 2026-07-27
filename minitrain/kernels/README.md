# Kernel backend 导读

模型只依赖 `minitrain/model/ops.py::OpsBackend`，目前有三种实现：

- `torch_ops.py`：可移植正确性基线；
- `triton/`：RMSNorm、RoPE、SwiGLU、FlashAttention、CE、fused linear CE、router 和
  fused MoE；
- `cuda_ext/`：需要更细控制时使用的 CUDA C++ 扩展，当前重点是 FlashAttention。

Triton/CUDA backend 不是“全有或全无”。每个 method 先检查 device、dtype、shape 和
功能约束；支持则运行自定义 kernel，不支持就回退到 Torch。CPU smoke、dropout 或
确定性算法模式因此仍可运行，但性能报告必须确认实际走了哪个分支。

## Dtype 合同

- token id/target：`torch.long`；
- residual/activation：FP32、BF16 或 FP16；
- softmax、norm、loss 和必要的归约累积：FP32；
- kernel 输出恢复 activation dtype；
- BF16 不使用 GradScaler，FP16 使用动态 scaling。

公共 AMP 边界在 `kernels/amp.py`。新增 kernel 时不能只测 forward 数值，还要验证
autocast、backward dtype 和非连续输入。

## MoE 路径

router logits 用 FP32 后处理；`router.py` 支持最多 1024 experts、top-k 最多 8。
当前模型使用 dropless `T×K` routing，不启用 capacity mask。fused MoE 负责 token
scatter、grouped expert SwiGLU 和加权聚合；expert parallel 尚未实现。

环境变量：

- `MINITRAIN_FUSED_CE_WORKSPACE_MB`：fused linear CE 的 logits chunk 预算；
- `MINITRAIN_MOE_AUTOTUNE=1`：启用更大的 MoE Triton tuning 搜索。

## 一个新 kernel 的完成标准

1. 与 `TorchOpsBackend` 对齐 forward；
2. 对齐 backward/gradient；
3. 覆盖 FP32/BF16/FP16 支持矩阵和 fallback；
4. 测 contiguous/strided 与边界 shape；
5. warmup、CUDA synchronize 后报告 P50/P95 和峰值显存；
6. 保存机器可读 raw JSON，不只保留截图。

Notebook 和结果路径见 [`tests/README.md`](../../tests/README.md)，CUDA 扩展编译见
[`docs/kernels/cuda_ext_run_commands.md`](../../docs/kernels/cuda_ext_run_commands.md)。
