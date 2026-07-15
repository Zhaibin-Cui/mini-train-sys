# MoE Architecture

## Data flow

For every token vector `x`, the router projects to fp32 logits and delegates
their postprocessing to the selected operator backend:

```text
x -> fp32 GEMM -> fused router postprocess -> top-k experts + statistics
                                      |
                                      v
                         grouped expert SwiGLU GEMM
                                      |
                                      v
                         weighted sum -> residual stream
```

Only the selected experts run for a token. With `E` total experts and top-k `K`,
the parameter count grows roughly with `E`, while token compute grows roughly
with `K`. This is the core MoE tradeoff: more parameter capacity without running
every parameter for every token.

## Code boundaries

- `model/moe_router.py`: fp32 projection, auxiliary-loss
  composition, and routing metrics.
- `model/blocks.py`: expert parameters and backend dispatch.
- `kernels/torch_ops.py`: readable router and expert correctness references.
- `kernels/triton/router.py`: fused softmax, Top-K, selected-weight
  normalization, probability statistics, z-loss, entropy, and their backward.
- `kernels/triton/fused_moe*.py`: grouped expert forward/backward kernels.
- `model/transformer.py`: aggregate router losses and metrics across layers.

The router knows only `OpsBackend.router_postprocess`; it does not import
Triton. Unsupported shapes, deterministic mode, CPU, and environments without
Triton use the Torch oracle automatically.

## Fusion boundary

The router projection stays in `F.linear`, allowing PyTorch to select its
library GEMM. The postprocess kernel operates on `[tokens, experts]` fp32
logits and avoids materializing the equally large probability matrix. It emits
only `[tokens, top_k]` routes, `[experts]` probability means, and scalar
statistics.

Its custom backward combines all differentiable paths:

- selected expert weights from the language-model loss;
- mean expert probability from the load-balancing auxiliary loss;
- log-normalizer from router z-loss.

Top-K indices, route counts, and entropy metrics are non-differentiable. The
active model path is dropless so every backend evaluates the same fixed `T*K`
routing graph. The retained capacity helper is experimental and is not called
by `TopKRouter.forward`.

## Stability controls

- Router math is fp32 even when activations use bf16/fp16.
- Selected weights are normalized to sum to one.
- Auxiliary load-balancing loss discourages expert collapse.
- Router z-loss limits logit growth.
- Optional input jitter provides routing exploration during training.
- Dropless routing is the default, avoiding silent token loss.
- Capacity masking is disabled until dispatch compaction can remove the same
  routes consistently from every backend's forward and backward computation.
- Route loads use `bincount`, avoiding a `[tokens, top_k, experts]` one-hot
  allocation.

## Observability

Training logs expose router entropy, auxiliary loss, z-loss, dropped-route
fraction, and min/max expert load. A healthy run should not have persistently
collapsed entropy, one expert receiving almost every route, or unexpected drops.

## Scope

This is a production-oriented single-process MoE path. True large-cluster MoE
also needs expert parallel process groups, all-to-all token dispatch, topology-
aware placement, communication/computation overlap, and distributed checkpoint
resharding. Those are distributed-system features and should remain outside the
router and expert math implemented here.

## Design references

- [Megatron-Core MoE](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md): router precision, load balancing, dropless routing, and fusion controls.
- [Megatron-Core router API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.router.html): separation of gating, Top-K, loss, and token dropping.
- [vLLM fused MoE](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py): compiled Top-K selection followed by expert execution.
- [SGLang Expert Parallelism](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/expert_parallelism.md): distinct Top-K, dispatch, permutation, expert, and combine stages.
- [Triton fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html): row-wise reduction and backend fallback principles.
