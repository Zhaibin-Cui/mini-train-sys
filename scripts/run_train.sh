#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${CONFIG:-configs/train_single.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_default.yaml}"
DEVICE="${DEVICE:-auto}"
PYTHON="${PYTHON:-python}"
ARGS=(scripts/train.py --config "$CONFIG" --model-config "$MODEL_CONFIG" --device "$DEVICE")
if [[ -n "${RESUME:-}" ]]; then ARGS+=(--resume "$RESUME"); fi
"$PYTHON" "${ARGS[@]}"
