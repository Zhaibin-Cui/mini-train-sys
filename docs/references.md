# References

- `../nanogpt`: minimal GPT training code reference.
- `../nanochat`: modern small LLM training harness reference.
- `../Liger-Kernel`: Triton kernels for LLM training, especially RMSNorm, RoPE, SwiGLU, CrossEntropy, and FusedLinearCrossEntropy.
- Triton official fused attention tutorial: FlashAttention v2-style Triton implementation and the preferred starting point for `flash_attention_pretraining_plan.md`.
- `../torchtitan`: PyTorch-native distributed training system reference.
- `../Megatron-LM`: large-scale tensor, pipeline, data, expert, and context parallelism reference.
- `../DeepSpeed`: runtime engine, ZeRO-style optimizer state partitioning, and CUDA extension reference.
- NCCL docs: collective communication semantics and tuning reference.

For the exact influence on this project, read `reference_map.md`.
