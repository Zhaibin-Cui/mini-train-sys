# Reference Map

This project intentionally borrows structure from the reference repositories in
the workspace, but keeps the implementation small enough to finish in a week.

## nanoGPT

Location: `../nanogpt`

What it does:
- compact GPT model and training script;
- easy-to-read baseline for causal LM training;
- useful examples for sampling, config overrides, and quick benchmarking.

What MiniTrainSys borrows:
- simple script-first workflow under `scripts/`;
- a small transformer baseline that can be understood quickly;
- the idea that the first version should run before it becomes elegant.

## nanochat

Location: `../nanochat`

What it does:
- modern small-LLM training harness;
- tokenizer, dataloader, checkpoint, eval, and engine code are still readable;
- useful reference for reporting tokens/sec, memory, and loss during training.

What MiniTrainSys borrows:
- `train/` and `data/` boundaries;
- checkpoint and optimizer components as independent modules;
- the preference for a small but complete training path.

## Liger-Kernel

Location: `../Liger-Kernel`

What it does:
- Triton kernels for LLM training ops;
- benchmark guidelines and scripts;
- Hugging Face and Megatron integration examples.

What MiniTrainSys borrows:
- `OpsBackend` exists so Liger-like kernels can be swapped into the model;
- `kernels/triton/*.py` mirrors the one-op-per-file implementation style;
- `bench/` and `reports/` are first-class, not afterthoughts.

## TorchTitan

Location: `../torchtitan`

What it does:
- PyTorch-native distributed LLM training system;
- clear split between model, config, components, distributed, and observability;
- references for FSDP, tensor parallel, pipeline parallel, context parallel, and metrics.

What MiniTrainSys borrows:
- `distributed/` owns distributed setup and wrapping;
- `runtime/` owns config/device/factory glue;
- future observability should follow a component style instead of being buried in scripts.

## Megatron-LM

Location: `../Megatron-LM`

What it does:
- industrial model-parallel training stack;
- process-group management, tensor parallel, pipeline parallel, sequence/context parallel;
- optimized transformer/fusion modules and large-scale training scripts.

What MiniTrainSys borrows:
- the long-term split between `core` algorithms and `training` orchestration;
- future `distributed/tensor_parallel.py` and `pipeline_parallel.py` should be modeled after its boundaries;
- custom kernels should not leak parallelism details into the base model.

## DeepSpeed

Location: `../DeepSpeed`

What it does:
- runtime engine around distributed training;
- ZeRO optimizer stages, offload, checkpointing, communication, CUDA extensions;
- strong separation between Python runtime and low-level `csrc`/op builders.

What MiniTrainSys borrows:
- `runtime/` as the place for glue code;
- future ZeRO-like optimizer experiments should live below `distributed/` and `train/optim.py`;
- CUDA extension work should stay under `kernels/cuda_ext/`.

