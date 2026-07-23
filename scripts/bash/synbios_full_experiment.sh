#!/usr/bin/env bash
set -euo pipefail

# 切换到项目根目录。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# 自动加载挂载盘上的缓存和实验存储配置。
[[ -f "$ROOT/.minitrain-storage.env" ]] && source "$ROOT/.minitrain-storage.env"

# 读取训练策略，并要求显式确认完整实验。
STRATEGY="${1:-single}"
if [[ "$STRATEGY" != "single" && "$STRATEGY" != "ddp" && "$STRATEGY" != "fsdp" ]]; then
  echo "usage: $0 [single|ddp|fsdp]" >&2
  exit 2
fi
if [[ "${CONFIRM_FULL_EXPERIMENT:-0}" != "1" ]]; then
  echo "This launches both full pretraining conditions and all probes." >&2
  echo "Set CONFIRM_FULL_EXPERIMENT=1 after checking configs and disk capacity." >&2
  exit 2
fi

# 根据策略确定预训练和探针使用的 GPU 数量。
if [[ "$STRATEGY" == "single" ]]; then
  NPROC=1
else
  NPROC="${NPROC:-$(nvidia-smi --list-gpus | wc -l)}"
  if [[ "$NPROC" != "4" && "$NPROC" != "8" ]]; then
    echo "SynBio distributed presets support NPROC=4 or NPROC=8" >&2
    exit 2
  fi
fi
export NPROC
export PROBE_GPUS="${PROBE_GPUS:-$NPROC}"

# 分布式完整实验在预训练前确认正式 Probe 的容量回归结果可用。
if [[ "$STRATEGY" != "single" && "${ALLOW_DEFAULT_PROBE_BATCHES:-0}" != "1" ]]; then
  [[ -n "${PROBE_BATCH_ENV:-}" && -f "$PROBE_BATCH_ENV" ]] || {
    echo "distributed full experiment requires PROBE_BATCH_ENV=<benchmark>/recommended.env" >&2
    exit 2
  }
fi

# 依次训练 single 和 multi5_permute 两种语料条件。
variants=(single multi5_permute)
for variant in "${variants[@]}"; do
  bash scripts/bash/synbios_moe.sh "$variant" "$STRATEGY"
done

# 两种条件分别按 smoke、pilot、formal 顺序运行探针。
for variant in "${variants[@]}"; do
  for stage in smoke pilot formal; do
    STAGE="$stage" bash scripts/bash/synbios_probes.sh \
      "$variant" "$STRATEGY" latest
  done
done

# 生成与训练产物目录一致的运行后缀。
run_suffix() {
  local variant="$1"
  if [[ "$STRATEGY" == "single" ]]; then
    printf '%s_single' "$variant"
  else
    printf '%s_%s_%sgpu' "$variant" "$STRATEGY" "$NPROC"
  fi
}

single_run="$(run_suffix single)"
augmented_run="$(run_suffix multi5_permute)"

# 汇总并比较两种语料条件的正式探针结果。
python scripts/synbios_moe.py summarize-probes \
  --run "single=artifacts/synbios_moe/results/$single_run/probe_pipeline/formal/validation" \
  --run "multi5_permute=artifacts/synbios_moe/results/$augmented_run/probe_pipeline/formal/validation" \
  --output "artifacts/synbios_moe/results/comparison_${STRATEGY}_${NPROC}gpu"

# 输出完整实验完成信息。
echo "Full SynBioS experiment completed: strategy=$STRATEGY world_size=$NPROC"
