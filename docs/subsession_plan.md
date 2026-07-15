# Subsession Plan

Use one focused sub-session per module. Each sub-session should leave behind
tests, benchmark notes, or a short report entry.

## Session 1: Baseline Training Path

<!-- Files:
- `scripts/train.py`
- `minitrain/runtime/`
- `minitrain/train/`
- `minitrain/data/`

Deliverables:
- load YAML configs;
- build model, ops backend, optimizer, dataloader, and strategy;
- run a few CPU/CUDA smoke-test steps;
- log loss, tokens/sec, and peak memory.

References:
- `../nanogpt/train.py`
- `../nanochat/nanochat/engine.py`
- `../nanochat/nanochat/checkpoint_manager.py` -->

<!-- ## Session 2: Operator Benchmark Harness

Files:
- `tests/operator_bench.ipynb`
- `tests/operator_bench_utils.py`
- `tests/test_backend_contract.py`
- `reports/operator_bench.md`

Deliverables:
- benchmark RMSNorm, RoPE, SwiGLU, CrossEntropy, FusedLinearCrossEntropy;
- compare correctness against `TorchOpsBackend`;
- report p50/p95 latency, memory, and speedup.

References:
- upstream kernel benchmarks
- upstream RMSNorm implementation
- upstream fused linear cross entropy implementation -->

<!-- ## Session 3: Triton Kernel 1 - RMSNorm

Files:
- `minitrain/kernels/triton/rmsnorm.py`
- `minitrain/kernels/triton/__init__.py`
- `tests/test_backend_contract.py`

Deliverables:
- Triton forward implementation;
- backward implementation or custom autograd wrapper;
- shape sweep and error analysis;
- decide whether CUDA C++ is worth trying.

References:
- upstream RMSNorm implementation

## Session 4: Triton Kernel 2 - RoPE and SwiGLU

Files:
- `minitrain/kernels/triton/rope.py`
- `minitrain/kernels/triton/swiglu.py`
- `reports/operator_bench.md`

Deliverables:
- fused Q/K RoPE path;
- SwiGLU elementwise kernel;
- benchmark launch overhead and memory bandwidth impact.

References:
- upstream RoPE implementation
- upstream SwiGLU implementation -->

## Session 5: Triton FlashAttention For Pretraining

Files:
- `docs/flash_attention_pretraining_plan.md`
- `minitrain/kernels/triton/flash_attention.py`
- `minitrain/kernels/triton_ops.py`
- `tests/test_triton_flash_attention.py`
- `tests/operator_bench_utils.py`
- `reports/operator_bench.md`

Deliverables:
- pretraining-only FlashAttention-style Triton forward and backward;
- causal bf16/fp16 support for `[B, H, S, D]`, starting with `D in {64, 128}`;
- fallback to PyTorch SDPA for unsupported shapes;
- correctness tests against SDPA for output and `dq/dk/dv`;
- benchmark latency, memory, and selected SDPA backend.

References:
- Triton official fused attention tutorial:
  `https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html`
- `../DeepSpeed/deepspeed/ops/transformer/inference/triton/attention.py`
- `../nanochat/nanochat/flash_attention.py`

Notes:
- This session is for pretraining only. Do not implement KV cache, paged
  attention, decode kernels, or serving-specific features here.
- Do not use the upstream fused neighborhood attention implementation
  as the algorithmic base because it materializes `[seq, seq]` intermediates.
  It is useful only for Triton project structure and benchmark style.

## Session 6: Fused Linear Cross Entropy

Files:
- `minitrain/kernels/triton/fused_linear_ce.py`
- `minitrain/model/transformer.py`
- `reports/operator_bench.md`

Deliverables:
- memory-saving fused loss path;
- benchmark large-vocab shapes;
- explain numerical stability and avoided logits materialization.

References:
- upstream fused linear cross entropy implementation
- upstream chunked-loss implementation

## Session 7: DDP Benchmark

Files:
- `minitrain/distributed/ddp.py`
- `reports/distributed_bench.md`

Deliverables:
- torchrun launch path;
- fixed-global-batch and fixed-per-GPU-batch benchmarks;
- bucket-size sweep and communication-overlap notes.

References:
- `../torchtitan/torchtitan/distributed/`
- `../nanochat/nanochat/execution.py`

## Session 8: FSDP and ZeRO-Style Memory Story

Files:
- `minitrain/distributed/fsdp.py`
- `minitrain/train/optim.py`
- `reports/distributed_bench.md`

Deliverables:
- FSDP benchmark against DDP;
- peak VRAM comparison;
- written explanation of parameters, gradients, and optimizer-state sharding.

References:
- `../torchtitan/torchtitan/distributed/fsdp.py`
- `../DeepSpeed/deepspeed/runtime/zero/`

## Session 9: Custom AllReduce

Files:
- `minitrain/distributed/custom_allreduce.py`
- `reports/distributed_bench.md`

Deliverables:
- teaching ring allreduce implementation;
- correctness test against `dist.all_reduce`;
- latency/bandwidth comparison with NCCL.

References:
- `../Megatron-LM/megatron/core/parallel_state.py`
- `../DeepSpeed/deepspeed/comm/`

## Session 10: CUDA C++ Extension Candidate

Files:
- `minitrain/kernels/cuda_ext/`
- `pyproject.toml`
- `tests/`

Deliverables:
- choose one kernel where Triton leaves performance on the table;
- implement a PyTorch custom op;
- compare PyTorch, Triton, and CUDA C++.

References:
- `../DeepSpeed/csrc/`
- `../DeepSpeed/op_builder/`
