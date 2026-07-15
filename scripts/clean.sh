#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rm -rf -- \
  "$ROOT/.pytest_cache" \
  "$ROOT/.triton_cache" \
  "$ROOT/build" \
  "$ROOT/dist" \
  "$ROOT/mini_train_sys.egg-info" \
  "$ROOT/minitrain/kernels/cuda_ext/build" \
  "$ROOT/tests/benchmark_results" \
  "$ROOT/checkpoints" \
  "$ROOT/runs" \
  "$ROOT/logs" \
  "$ROOT/outputs" \
  "$ROOT/profiles"
find "$ROOT" -maxdepth 1 -type d -name '.pytest_tmp*' -prune -exec rm -rf -- {} +
find "$ROOT" -type d -name '__pycache__' -prune -exec rm -rf -- {} +
find "$ROOT" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
find "$ROOT" -maxdepth 1 -type f \( -name '*.obj' -o -name '*.lib' -o -name '*.exp' \) -delete
echo "Generated build, test, cache, and run outputs removed."
