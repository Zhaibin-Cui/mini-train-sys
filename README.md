<div align="center">

<img src="https://raw.githubusercontent.com/Zhaibin-Cui/mini-train-sys/main/assets/readme-banner.svg" alt="MiniTrainSys banner" width="100%" />

<br />

**A tiny, readable, benchmark-first LLM pretraining system.**

MiniTrainSys is a runnable guide for understanding LLM pretraining systems from
the inside out: start with a PyTorch baseline, then progressively explore
Triton kernels, fused operators, DDP/FSDP, communication experiments, and
performance reports.

<br />

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Training-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-Kernels-7C3AED?style=for-the-badge)](https://triton-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-FACC15?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Learning%20System-22C55E?style=for-the-badge)](#current-status)

</div>

---

## 🌱 What is MiniTrainSys?

MiniTrainSys is a compact training framework for learning and experimenting
with **LLM pretraining systems**. It does not try to train the largest possible
model. Instead, it breaks the pretraining stack into small, inspectable pieces
that can be read, replaced, tested, and benchmarked independently.

<table>
  <tr>
    <td align="center"><b>🧠 Model</b><br />LLaMA-style mini Transformer</td>
    <td align="center"><b>⚙️ Ops</b><br />PyTorch, Triton, future CUDA</td>
    <td align="center"><b>🚂 Training</b><br />small loop, clear metrics</td>
  </tr>
  <tr>
    <td align="center"><b>📦 Data</b><br />random and token files</td>
    <td align="center"><b>🌐 Distributed</b><br />single, DDP, FSDP</td>
    <td align="center"><b>📊 Reports</b><br />latency, memory, scaling</td>
  </tr>
</table>

The project is meant to be a **cute but serious pretraining systems guide**:
small enough to understand, structured enough to measure real tradeoffs.

---

## 🎯 Goals

- Build a clear causal language model pretraining path.
- Keep PyTorch as the correctness reference.
- Swap in Triton and future CUDA kernels one operator at a time.
- Decouple kernel experiments from distributed training strategy.
- Track latency, throughput, memory, tokens/sec, and scaling efficiency.
- Keep the code small enough for reading, teaching, reproduction, and hacking.

---

## 🧩 Current Status

| Area | Status | Notes |
| --- | --- | --- |
| 🧠 Model | ✅ Available | LLaMA-style mini Transformer with RMSNorm, RoPE, SwiGLU, and causal attention |
| 🚂 Training loop | ✅ Available | optimizer step, token accounting, memory metrics, console/TensorBoard logging |
| 📦 Data | ✅ Available | random token stream and pre-tokenized `.pt` / `.npy` / `.bin` inputs |
| ⚙️ Ops backend | ✅ Available | PyTorch backend; Triton backend with RMSNorm replacement and fallback path |
| 🌐 Distributed | 🚧 Scaffolded | single-device, DDP, and FSDP strategy interfaces |
| 📊 Benchmarks | 🚧 In progress | operator benchmark notebook and report templates |
| 🔧 CUDA extension | 📝 Planned | extension layout exists; kernels are not implemented yet |

---

## 💡 Why This Exists

Large training frameworks such as Megatron-LM, DeepSpeed, and TorchTitan are
powerful, but production-scale code paths are not always the easiest place to
learn the fundamentals.

MiniTrainSys keeps the same important questions while reducing the surface area:

- What exactly happens in one pretraining step?
- Which operators dominate latency and memory?
- Why do fused kernels reduce memory pressure?
- Where are the boundaries between single-GPU, DDP, and FSDP training?
- What makes a benchmark trustworthy?
- Which production-system ideas are worth learning first?

MiniTrainSys is meant to produce more than code. It is a learning path with
measurements.

---

## 🏗️ Architecture

```text
mini-train-sys/
  configs/                 model sizes and experiment configs
  minitrain/
    data/                  token streams and dataloaders
    distributed/           single-device, DDP, FSDP and communication scaffolds
    kernels/               PyTorch reference ops, Triton kernels, CUDA extension slot
    model/                 Transformer blocks and backend-agnostic op calls
    runtime/               config loading, device selection, backend factories, logging
    train/                 trainer, optimizer, checkpoint and metric utilities
    utils/                 seed and profiler helpers
  reports/                 benchmark notes, figures and result summaries
  scripts/                 CLI entry points for training, eval, sampling and benchmark runs
  tests/                   correctness tests and notebook-first operator benchmark
```

MiniTrainSys is organized around two small interfaces:

| Interface | Purpose |
| --- | --- |
| `OpsBackend` | Switch model operators between `torch`, `triton`, and future CUDA C++ ops |
| `ParallelStrategy` | Switch execution between single-device, DDP, FSDP, and future custom communication strategies |

This makes it natural to compare combinations such as:

```text
torch  + single
triton + single
torch  + ddp
triton + ddp
```

without rewriting the model or trainer.

---

## 🚀 Installation

Python 3.10+ is recommended. CPU is enough for smoke tests. Triton and
distributed GPU experiments are best run in a CUDA-enabled Linux environment.

```bash
git clone https://github.com/Zhaibin-Cui/mini-train-sys.git
cd mini-train-sys

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

For Triton kernel experiments:

```bash
pip install -e ".[triton,dev]"
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## ⚡ Quick Start

Run a tiny CPU smoke test:

```bash
python scripts/train.py \
  --config configs/train_single.yaml \
  --model-config configs/model_tiny_smoke.yaml \
  --smoke-steps 3 \
  --device cpu
```

Run a single-GPU baseline:

```bash
python scripts/train.py \
  --config configs/train_single.yaml \
  --model-config configs/model_15m.yaml \
  --device cuda
```

Run DDP on two local GPUs:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/train_ddp.yaml \
  --model-config configs/model_15m.yaml \
  --device cuda
```

Run tests:

```bash
pytest
```

Open the operator benchmark notebook:

```bash
jupyter notebook tests/operator_bench.ipynb
```

---

## 📝 Configuration

Model configs and run configs are intentionally separated.

| Config | Usage |
| --- | --- |
| `configs/model_tiny_smoke.yaml` | tiny CPU/GPU sanity-check model |
| `configs/model_15m.yaml` | default small benchmark model |
| `configs/model_50m.yaml` | medium scaling target |
| `configs/model_125m.yaml` | larger scaling target |
| `configs/train_single.yaml` | single-device baseline |
| `configs/train_ddp.yaml` | DDP experiment |
| `configs/train_fsdp.yaml` | FSDP experiment |

Backend selection:

```yaml
backend:
  ops: torch       # torch | triton | cuda
  parallel: single # single | ddp | fsdp
```

Token data:

```yaml
data:
  source: tokens
  path: data/tokens.npy
  shuffle: true
```

Supported token file formats:

```text
.pt / .pth / .npy / .bin
```

---

## 📊 Benchmark Philosophy

MiniTrainSys treats performance numbers as first-class project artifacts.

A benchmark is only useful when it records:

- 🖥️ hardware and software environment
- 🧮 model size, sequence length, and batch size
- ⚙️ backend and distributed strategy
- ✅ correctness checks against a reference path
- ⏱️ p50/p95 latency or end-to-end throughput
- 💾 peak memory usage
- 🧭 known caveats and next actions

Reports live under `reports/`:

| Report | Purpose |
| --- | --- |
| `reports/operator_bench.md` | operator-level benchmark methodology |
| `reports/training_bench.md` | end-to-end training benchmark notes |
| `reports/distributed_bench.md` | distributed scaling benchmark notes |
| `reports/figures/` | generated plots and summaries |

---

## 🗺️ Roadmap

- [ ] Add Triton RoPE, SwiGLU, CrossEntropy, and FusedLinearCrossEntropy replacements
- [ ] Add a pretraining-only Triton FlashAttention-style causal attention kernel
- [ ] Add CUDA C++ extension examples for teaching-oriented kernels
- [ ] Expand benchmark reports with reproducible GPU results
- [ ] Improve DDP/FSDP benchmark scripts and scaling summaries
- [ ] Add a teaching implementation of ring all-reduce and compare it with NCCL
- [ ] Add tokenizer and small real-data pretraining examples
- [ ] Add checkpoint resume and sampling examples suitable for demos

---

## 🌟 Reference Projects

MiniTrainSys is a study-oriented bridge between small educational repositories
and production training systems.

| Project | What to Learn From It |
| --- | --- |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | compact GPT training loop, simple data pipeline, readable baseline structure |
| [nanochat](https://github.com/karpathy/nanochat) | end-to-end small LLM pipeline: tokenizer, dataloader, checkpoint, eval, and chat flow |
| [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | Triton kernel design, correctness tests, benchmark organization, and fused training ops |
| [TorchTitan](https://github.com/pytorch/torchtitan) | PyTorch-native distributed training architecture, config discipline, and observability |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | tensor/pipeline/context parallelism, process groups, and large-model training patterns |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | ZeRO optimizer state partitioning, runtime engine design, and CUDA extension layout |

MiniTrainSys does not try to reimplement these systems wholesale. It extracts
the smallest useful version of each idea, makes it readable, and measures it in
isolation.

---

## 🤝 Contributing

Contributions should keep the project small and benchmark-driven.

A good PR usually includes:

- correctness tests against the PyTorch backend
- benchmark or report updates for performance-sensitive changes
- a small config or script showing how to reproduce the behavior
- minimal abstraction unless it removes real duplication

---

## 📄 License

MiniTrainSys is released under the [MIT License](LICENSE).

---

<div align="center">

<sub>Font note: the banner prefers Quicksand, Nunito, Comic Neue, and rounded UI fonts when available.</sub>

<br />

**Small code. Clear measurements. Real pretraining systems intuition.**

</div>
