# MoE 数据流与边界

## 一次 token 如何经过 MoE

```text
token hidden x
  → FP32 router projection
  → softmax / top-k / 权重归一化 / 路由统计
  → 把 token 发送到选中的 K 个 expert
  → 每个 expert 做 SwiGLU
  → 按 router weight 加权求和
  → 回到 residual stream
```

总 expert 数 `E` 决定参数容量，top-k `K` 近似决定每 token 激活的 expert 计算量。
当前 SynBioS 配置为 8 experts、top-2。

## 代码对应

| 阶段 | 文件 |
|---|---|
| Router projection 与 loss | `model/moe_router.py` |
| Expert 参数和 FFN dispatch | `model/blocks.py` |
| Torch 正确性基线 | `kernels/torch_ops.py` |
| Triton router 后处理 | `kernels/triton/router.py` |
| Triton grouped experts | `kernels/triton/fused_moe*.py` |
| 跨 layer 汇总 metrics/loss | `model/transformer.py` |

Router 只调用 `OpsBackend.router_postprocess()`，不会直接 import Triton。不支持的环境
自动回退 Torch。

## 为什么 projection 不一起融合

Router projection 是规则 GEMM，交给 PyTorch/CUDA library。Triton kernel 接收
`[tokens, experts]` FP32 logits，融合 softmax、top-k、选中权重归一化、概率统计、z-loss
和 entropy，避免额外保存完整 probability matrix。Top-k index、route count 和 entropy
是非可微统计；选中权重、负载均衡 loss 和 z-loss 的梯度由 custom backward 合并。

## 稳定性

- Router 数学保持 FP32；
- 选中 K 个权重归一化为和 1；
- auxiliary loss 抑制 expert collapse；
- z-loss 抑制 logits 无限制增长；
- 可选 jitter 用于探索；
- 当前默认 dropless，不静默丢 token；
- route load 使用 `bincount`，避免巨大 one-hot。

训练目标为：

```text
total = lm_cross_entropy
      + router_aux_loss_coef * auxiliary_loss
      + router_z_loss_coef * z_loss
```

其中 `auxiliary_loss = E × Σ(f_i × p_i)`：`f_i` 是实际 Top-k route 比例，`p_i` 是完整
softmax 平均概率；均匀分配时该项约为 1。z-loss 是
`mean(logsumexp(router_logits)^2)`，用于限制 logits 尺度。所有层的 router loss 取平均，
因此系数不会随层数线性放大。

日志将纯 CE、raw aux/z、加权 aux/z 和总 loss 分开。专家分布还记录每层实际选中比例与
softmax 概率，并在 TensorBoard 中显示固定色标热力图、相对均匀负载直方图及每 expert
曲线。应同时关注 entropy、load CV、max/mean、dead expert、min/max load 和 dropped
fraction。长期只有少数 experts 接收绝大多数 route，通常意味着 collapse。

指标先在 `log_interval` 内平均，再跨 DDP/FSDP ranks 平均。当前没有 expert parallel，
所以这是所有复制 router 的平均行为，而不是某个 rank 的偶然快照。

## 当前范围

这是一条生产导向的本地 MoE 计算路径，但不是 expert-parallel 系统。DDP 每个 rank
都有全部 experts；FSDP 可以分片参数状态，但没有 all-to-all token dispatch、expert
placement 或 expert-parallel process group。不要把 FSDP 与 expert parallel 混为一谈。
