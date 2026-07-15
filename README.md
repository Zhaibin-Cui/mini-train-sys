<div align="center">

<img src="assets/readme-banner.svg" alt="MiniTrainSys banner" width="100%" />

**A compact, readable LLM training and kernel experimentation system.**

</div>

## Features

- One Transformer architecture with configurable Dense or MoE FFN.
- PyTorch reference ops plus Triton RMSNorm, RoPE, SwiGLU, attention, CE, fused CE, and MoE paths.
- Single-GPU, DDP, and FSDP strategies.
- AdamW LLM parameter groups, optional warmup, constant LR, and cosine decay.
- Mixed precision, gradient clipping, TensorBoard/console metrics, and retained epoch checkpoints.
- Full-state resume through the same training entry point.
- Correctness tests and notebook-based operator benchmarks.

## Install

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[triton,dev]"
```

Windows activation:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[triton,dev]"
```

CUDA extension experiments additionally use:

```bash
pip install -e ".[cuda]"
```

## Train

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_train.ps1

powershell -ExecutionPolicy Bypass -File scripts/run_train.ps1 `
  -Config configs/train_single.yaml `
  -ModelConfig configs/model_125m_moe.yaml `
  -Device cuda `
  -Resume latest
```

Linux single GPU:

```bash
bash scripts/run_train.sh
MODEL_CONFIG=configs/model_125m_moe.yaml RESUME=latest bash scripts/run_train.sh
```

Linux distributed:

```bash
NPROC=8 bash scripts/run_distributed.sh ddp
NPROC=8 bash scripts/run_distributed.sh fsdp
```

Direct entry point:

```bash
python scripts/train.py \
  --config configs/train_debug.yaml \
  --model-config configs/model_debug_dense.yaml \
  --device cpu
```

## Configuration

Model and training configs are intentionally separate:

| Preset | Purpose |
| --- | --- |
| `model_default.yaml` | Default 32K-vocab, 2K-context LLaMA-style dense model |
| `model_debug_dense.yaml`, `model_debug_moe.yaml` | Small architecture checks |
| `model_15m_dense.yaml` | Fast end-to-end experiments |
| `model_125m_dense.yaml`, `model_125m_moe.yaml` | Representative training |
| `train_debug.yaml` | Fixed-LR CPU/debug run |
| `train_single.yaml` | LLM single-GPU defaults |
| `train_ddp.yaml`, `train_fsdp.yaml` | Distributed training |

See [configs/README.md](configs/README.md) for every supported field and LR recipe.

## Checkpoints

```yaml
train:
  epochs: 20
  max_steps: null
  checkpoint_every_epochs: 2
  checkpoint_dir: checkpoints
  save_final_checkpoint: true
  resume_from: latest
```

Each checkpoint retains model, optimizer, LR scheduler, scaler, trainer counters, RNG, precision,
and the full configuration snapshot. Files use unique epoch/step names and are not overwritten.

## Test and benchmark

```bash
jupyter notebook tests/example_training.ipynb
jupyter notebook tests/operator_bench.ipynb
jupyter notebook tests/moe_operator_bench.ipynb
```

## Repository layout

```text
configs/       model and experiment presets
docs/          architecture and kernel notes
minitrain/     data, model, kernels, distributed, runtime, and training modules
reports/       benchmark reports and selected figures
scripts/       training, distributed launch, benchmark, and cleanup entry points
tests/         formal regression tests and benchmark notebooks
```

Generated caches, native build products, checkpoints, runs, and raw benchmark output are ignored.
Clean them explicitly with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/clean.ps1
```

```bash
bash scripts/clean.sh
```

## License

Project code is MIT licensed. Third-party notices are shipped with the relevant kernel package.
