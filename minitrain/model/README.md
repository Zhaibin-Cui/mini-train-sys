# Model 模块

`MiniTransformer` 是唯一主干；`ModelConfig.ffn_type` 在每个 `TransformerBlock` 中选择
Dense 或 MoE FFN。attention、RMSNorm、RoPE、residual 和 loss 路径共用。

```text
config.py       模型尺寸和 MoE 参数
transformer.py  embedding → blocks → norm → LM head/loss
blocks.py       Attention、Dense/MoE FFN、TransformerBlock
moe_router.py   FP32 top-k router 和统计
ops.py          Torch/Triton/CUDA backend 协议
rotary.py       模型级 RoPE cache
```

模型不读取 YAML、不初始化 process group、不保存 checkpoint。组装工作在 `runtime/`，
训练工作在 `train/`。详见 [`docs/guides/architecture.md`](../../docs/guides/architecture.md)。
