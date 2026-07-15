# References

- `../nanogpt`: minimal GPT training code reference.
- `../nanochat`: modern small LLM training harness reference.
- External Triton kernel references for RMSNorm, RoPE, SwiGLU, cross entropy, and fused loss.
- Triton official fused attention tutorial: FlashAttention v2-style Triton implementation and the preferred starting point for `flash_attention_pretraining_plan.md`.
- `../torchtitan`: PyTorch-native distributed training system reference.
- `../Megatron-LM`: large-scale tensor, pipeline, data, expert, and context parallelism reference.
- `../DeepSpeed`: runtime engine, ZeRO-style optimizer state partitioning, and CUDA extension reference.
- NCCL docs: collective communication semantics and tuning reference.
- Megatron-Core MoE/router docs: router precision, auxiliary losses, dropless
  routing, capacity, and inference/training separation.
- vLLM fused MoE and SGLang Expert Parallelism: Top-K as a distinct stage from
  token dispatch, grouped expert execution, and combine.
- Triton fused-softmax tutorial: row-wise reduction structure used by router
  postprocessing.

For the exact influence on this project, read `reference_map.md`.
