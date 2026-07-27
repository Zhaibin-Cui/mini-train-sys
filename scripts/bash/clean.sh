#!/usr/bin/env bash
set -euo pipefail

# 定位项目根目录。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 自动加载挂载盘配置，但不删除其中的数据和昂贵缓存。
[[ -f "$ROOT/.minitrain-storage.env" ]] && source "$ROOT/.minitrain-storage.env"

# 只清理 Python 临时缓存，保留数据、检查点和 CUDA/Triton 缓存。
find "$ROOT" -type d -name '__pycache__' -prune -exec rm -rf -- {} +
find "$ROOT" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

# 输出清理结果。
echo "Python temporary caches removed; CUDA/Triton caches preserved."
