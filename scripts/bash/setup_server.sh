#!/usr/bin/env bash
set -euo pipefail

# 切换到项目根目录，确保所有相对路径稳定。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# 默认要求先配置挂载盘，防止正式产物误写入系统盘。
if [[ -f "$ROOT/.minitrain-storage.env" ]]; then
  source "$ROOT/.minitrain-storage.env"
elif [[ "${ALLOW_SYSTEM_DISK_STORAGE:-0}" != "1" ]]; then
  echo "Run 'bash scripts/bash/setup_storage.sh /mounted/disk' before server setup." >&2
  echo "For temporary smoke only, set ALLOW_SYSTEM_DISK_STORAGE=1." >&2
  exit 2
fi

# 读取可覆盖的安装参数。
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.5.1}"
PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-cu124}"

# 检查 Python 版本并创建或复用虚拟环境。
"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), sys.version'
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# 安装 PyTorch、服务器依赖并检查依赖冲突。
VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install \
  "torch==$PYTORCH_VERSION" \
  --index-url "https://download.pytorch.org/whl/$PYTORCH_CUDA_INDEX"
"$VENV_PYTHON" -m pip install -e ".[server]"
"$VENV_PYTHON" -m pip check

# 检查服务器 GPU/训练环境，并把环境信息保存到 artifacts。
if [[ "${SKIP_SERVER_CHECK:-0}" != "1" ]]; then
  EXPECTED_GPUS="${EXPECTED_GPUS:-$(nvidia-smi --list-gpus | wc -l)}"
  CHECK_ARGS=(--expected-gpus "$EXPECTED_GPUS" --output artifacts/server_environment.json)
  [[ "${REQUIRE_NVCC:-0}" == "1" ]] && CHECK_ARGS+=(--require-nvcc)
  "$VENV_DIR/bin/minitrain-check-server" "${CHECK_ARGS[@]}"
fi

# 输出虚拟环境激活命令。
echo "Server environment is ready. Activate it with: source $VENV_DIR/bin/activate"
