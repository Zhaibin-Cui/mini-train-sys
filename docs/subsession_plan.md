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
- `../Liger-Kernel/benchmark/`
- `../Liger-Kernel/src/liger_kernel/ops/rms_norm.py`
- `../Liger-Kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py` -->

## Session 3: Triton Kernel 1 - RMSNorm

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
- `../Liger-Kernel/src/liger_kernel/ops/rms_norm.py`

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
- `../Liger-Kernel/src/liger_kernel/ops/rope.py`
- `../Liger-Kernel/src/liger_kernel/ops/swiglu.py`

## Session 5: Fused Linear Cross Entropy

Files:
- `minitrain/kernels/triton/fused_linear_ce.py`
- `minitrain/model/transformer.py`
- `reports/operator_bench.md`

Deliverables:
- memory-saving fused loss path;
- benchmark large-vocab shapes;
- explain numerical stability and avoided logits materialization.

References:
- `../Liger-Kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py`
- `../Liger-Kernel/src/liger_kernel/chunked_loss/`

## Session 6: DDP Benchmark

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

## Session 7: FSDP and ZeRO-Style Memory Story

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

## Session 8: Custom AllReduce

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

## Session 9: CUDA C++ Extension Candidate

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
