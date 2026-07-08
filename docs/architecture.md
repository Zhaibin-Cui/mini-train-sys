# Architecture

## Model Layer

`MiniTransformer` depends only on an `OpsBackend`. It should not import Triton,
CUDA extensions, or distributed code directly.

Reference influence:
- `nanoGPT` for keeping the base transformer readable.
- `Megatron-LM` for the long-term rule that model-parallel details should be
  isolated from the ordinary transformer definition.

## Kernel Layer

Each backend must preserve the same operator contract. This lets benchmarks
switch backends without changing model or trainer code.

Reference influence:
- `Liger-Kernel` for one optimized LLM op per implementation file.
- `DeepSpeed` for keeping CUDA/C++ extension work out of the Python model layer.

## Distributed Layer

`ParallelStrategy` owns process-group setup and model wrapping. Training code
should not branch directly on DDP/FSDP/custom choices.

Reference influence:
- `TorchTitan` for the explicit distributed module boundary.
- `Megatron-LM` for future process-group and model-parallel extensions.
- `DeepSpeed` for later ZeRO-style optimizer and runtime experiments.

## Runtime Layer

`runtime/` owns config loading, device selection, and factories that connect the
model, backend, and distributed strategy. This avoids turning `scripts/train.py`
into the only place where the system is understandable.

Reference influence:
- `TorchTitan` for typed config sections and explicit factories.
- `DeepSpeed` for separating runtime orchestration from lower-level ops.

## Benchmark Layer

Benchmarks should write machine-readable raw results and short Markdown summaries.
Every benchmark needs environment metadata, warmup, synchronization, and correctness
checks before speed numbers are trusted.

Reference influence:
- `Liger-Kernel/benchmark` for treating benchmark methodology as part of the
  product.
- `nanochat` for end-to-end metrics such as tokens/sec and memory.
