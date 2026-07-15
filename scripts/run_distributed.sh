#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-ddp}"
if [[ "$MODE" != "ddp" && "$MODE" != "fsdp" ]]; then
  echo "usage: $0 [ddp|fsdp]" >&2
  exit 2
fi
CONFIG="${CONFIG:-configs/train_${MODE}.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_default.yaml}"
NPROC="${NPROC:-$(nvidia-smi --list-gpus | wc -l)}"
ARGS=(--standalone --nproc_per_node "$NPROC" scripts/train.py --config "$CONFIG" --model-config "$MODEL_CONFIG" --device cuda)
if [[ -n "${RESUME:-}" ]]; then ARGS+=(--resume "$RESUME"); fi
torchrun "${ARGS[@]}"
