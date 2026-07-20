#!/usr/bin/env bash
set -euo pipefail

# 切换到项目根目录。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# 读取并校验数据变体和训练策略。
VARIANT="${1:-single}"
STRATEGY="${2:-single}"
if [[ "$VARIANT" != "single" && "$VARIANT" != "multi5_permute" ]]; then
  echo "usage: $0 [single|multi5_permute] [single|ddp|fsdp]" >&2
  exit 2
fi
if [[ "$STRATEGY" != "single" && "$STRATEGY" != "ddp" && "$STRATEGY" != "fsdp" ]]; then
  echo "usage: $0 [single|multi5_permute] [single|ddp|fsdp]" >&2
  exit 2
fi

# 确定数据目录和数据生成变体。
DATA_ROOT="artifacts/synbios_moe/$VARIANT"
PREPARE_VARIANT="$VARIANT"
[[ "$VARIANT" == "multi5_permute" ]] && PREPARE_VARIANT="multi5+permute"

# 已有检查点时禁止强制重建数据，避免训练数据与检查点不一致。
if [[ "${FORCE_PREPARE:-0}" == "1" ]] &&
  compgen -G "artifacts/synbios_moe/checkpoints/synbios_moe_${VARIANT}_*/epoch_*_step_*/COMMITTED" >/dev/null; then
  echo "FORCE_PREPARE would invalidate existing checkpoints; archive them first" >&2
  exit 2
fi

# 检查数据产物，缺失或显式要求时重新准备数据。
NEEDS_PREPARE=0
for required_path in \
  "$DATA_ROOT/profiles.jsonl" \
  "$DATA_ROOT/manifest.json" \
  "$DATA_ROOT/token_shards/manifest.json"; do
  if [[ ! -f "$required_path" ]]; then
    NEEDS_PREPARE=1
    break
  fi
done
if [[ "${FORCE_PREPARE:-0}" == "1" || "$NEEDS_PREPARE" == "1" ]]; then
  python scripts/synbios_moe.py prepare \
    --output "$DATA_ROOT" \
    --variant "$PREPARE_VARIANT"
fi

# 校验两个语料变体使用完全相同的人物事实表。
OTHER="single"
[[ "$VARIANT" == "single" ]] && OTHER="multi5_permute"
if [[ -f "artifacts/synbios_moe/$OTHER/profiles.jsonl" ]] &&
  ! cmp -s "$DATA_ROOT/profiles.jsonl" "artifacts/synbios_moe/$OTHER/profiles.jsonl"; then
  echo "Profile tables differ between variants; refusing an invalid comparison" >&2
  exit 2
fi

# 根据训练策略和 GPU 数量选择配置。
if [[ "$STRATEGY" == "single" ]]; then
  CONFIG="configs/synbios_moe/runs/${VARIANT}_single.yaml"
else
  NPROC="${NPROC:-$(nvidia-smi --list-gpus | wc -l)}"
  if [[ "$NPROC" != "4" && "$NPROC" != "8" ]]; then
    echo "SynBio server presets support NPROC=4 or NPROC=8" >&2
    exit 2
  fi
  CONFIG="configs/synbios_moe/runs/${VARIANT}_${STRATEGY}_${NPROC}gpu.yaml"
fi
MODEL_CONFIG="configs/synbios_moe/model.yaml"

# 生成运行名称并组装训练参数。
RUN_NAME="synbios_moe_${VARIANT}_${STRATEGY}"
[[ "$STRATEGY" != "single" ]] && RUN_NAME="${RUN_NAME}_${NPROC}gpu"
ARGS=(scripts/train.py --device cuda --config "$CONFIG" --model-config "$MODEL_CONFIG")

# 存在已提交检查点时默认自动续训。
if [[ "${AUTO_RESUME:-1}" == "1" ]] &&
  compgen -G "artifacts/synbios_moe/checkpoints/$RUN_NAME/epoch_*_step_*/COMMITTED" >/dev/null; then
  ARGS+=(--resume "${RESUME:-latest}")
fi

# 单卡使用 Python，多卡使用 torchrun 启动训练。
if [[ "$STRATEGY" == "single" ]]; then
  python "${ARGS[@]}"
else
  torchrun --standalone --nproc_per_node "$NPROC" "${ARGS[@]}"
fi
