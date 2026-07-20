#!/usr/bin/env bash
set -euo pipefail

# 切换到项目根目录。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# 读取单机训练配置并组装启动参数。
CONFIG="${CONFIG:-configs/train_single.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_default.yaml}"
DEVICE="${DEVICE:-auto}"
ARGS=(scripts/train.py --config "$CONFIG" --model-config "$MODEL_CONFIG" --device "$DEVICE")

# 按需追加断点恢复参数。
if [[ -n "${RESUME:-}" ]]; then ARGS+=(--resume "$RESUME"); fi

# 启动单进程训练。
"${PYTHON:-python}" "${ARGS[@]}"
