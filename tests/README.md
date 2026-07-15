# Notebook workflow

- `example_training.ipynb`: configs, LR, Dense/MoE, checkpoint, and optional Triton smoke validation.
- `operator_bench.ipynb`: general operator correctness and performance sweeps.
- `moe_operator_bench.ipynb`: fused-router gradient checks, capacity and
  deterministic fallback, BF16/FP16 end-to-end smoke coverage, plus router and
  grouped-expert benchmarks.
- `operator_bench_utils.py`: shared notebook benchmark utilities.
- `operator_nsight.py`: Nsight-oriented operator entry point.

Run notebooks from the repository root so local imports and config paths resolve consistently.
