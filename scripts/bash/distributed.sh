#!/usr/bin/env bash
set -euo pipefail

# 切换到项目根目录。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# 自动加载挂载盘上的缓存和实验存储配置。
[[ -f "$ROOT/.minitrain-storage.env" ]] && source "$ROOT/.minitrain-storage.env"

# 读取并校验分布式策略。
MODE="${1:-ddp}"
if [[ "$MODE" != "ddp" && "$MODE" != "fsdp" ]]; then
  echo "usage: $0 [ddp|fsdp]" >&2
  exit 2
fi

# 自动获取 GPU 数量并校验服务器预设。
NPROC="${NPROC:-$(nvidia-smi --list-gpus | wc -l)}"
if [[ "$NPROC" != "1" && "$NPROC" != "4" && "$NPROC" != "8" ]]; then
  echo "RTX 4090 server presets support NPROC=1, 4, or 8" >&2
  exit 2
fi

# 选择对应配置并组装训练参数。
CONFIG="${CONFIG:-configs/server/rtx4090_24gb/runs/${MODE}_${NPROC}gpu.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_default.yaml}"
ARGS=(--standalone --nproc_per_node "$NPROC" scripts/train.py --device cuda)
ARGS+=(--config "$CONFIG" --model-config "$MODEL_CONFIG")

# 按需追加断点恢复参数。
if [[ -n "${RESUME:-}" ]]; then ARGS+=(--resume "$RESUME"); fi

# 使用 torchrun 启动多进程训练。
torchrun "${ARGS[@]}"
