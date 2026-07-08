# Operator Benchmark

The operator benchmark is notebook-first:

- Notebook: `tests/operator_bench.ipynb`
- Reusable timing/plot helpers: `tests/operator_bench_utils.py`
- Generated figures: `reports/figures/`

The notebook compares three backend slots:

- `torch`: correctness and performance baseline.
- `triton`: optimized Triton slot, currently allowed to fall back to torch.
- `cuda`: CUDA extension slot, marked unavailable until implemented.

Each kernel has its own section and its own problem-size sweep:

- RMSNorm: sweep `rows`, x-axis is `rows * hidden`.
- RoPE: sweep `seq`, x-axis is Q+K elements.
- SwiGLU: sweep `rows`, x-axis is gate+up elements.
- CrossEntropy: sweep `tokens`, x-axis is logits elements.
- FusedLinearCrossEntropy: sweep `vocab`, x-axis is logical logits elements.

Each section reports correctness versus `TorchOpsBackend`, p50/p95 latency,
temporary peak CUDA memory, and speedup versus torch. Use the generated plots
for project presentations.
