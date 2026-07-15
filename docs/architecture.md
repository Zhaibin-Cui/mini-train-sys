# Architecture

## Model Layer

`MiniTransformer` depends only on an `OpsBackend`. It should not import Triton,
CUDA extensions, or distributed code directly.

`MiniTransformer` and `TransformerBlock` are shared by dense and MoE runs.
`ModelConfig.ffn_type` selects the feed-forward implementation, and
`build_feed_forward()` contains that single architecture branch. The attention,
normalization, residual, loss, trainer, and checkpoint paths therefore stay
identical. `MiniMoETransformer` remains only as a compatibility alias for older
callers.

Reference influence:
- `nanoGPT` for keeping the base transformer readable.
- `Megatron-LM` for the long-term rule that model-parallel details should be
  isolated from the ordinary transformer definition.

## Kernel Layer

Each backend must preserve the same operator contract. This lets benchmarks
switch backends without changing model or trainer code.

Reference influence:
- Existing Triton projects for one optimized LLM op per implementation file.
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

## Training Layer

`Trainer` owns one optimizer update, precision, clipping, and LR stepping.
`TrainingRunner` owns epochs, metrics, limits, and checkpoint cadence. The CLI
entry point only builds these components and restores state. AdamW parameter
grouping and LR policy live in `optim.py` and `lr_scheduler.py`, so schedule
choices do not add branches to the model or runner.

The MoE data flow and its local-versus-expert-parallel boundary are documented
in [`moe.md`](moe.md).

## Benchmark Layer

Benchmarks should write machine-readable raw results and short Markdown summaries.
Every benchmark needs environment metadata, warmup, synchronization, and correctness
checks before speed numbers are trusted.

Reference influence:
- Established kernel benchmarks for treating benchmark methodology as part of the
  product.
- `nanochat` for end-to-end metrics such as tokens/sec and memory.
