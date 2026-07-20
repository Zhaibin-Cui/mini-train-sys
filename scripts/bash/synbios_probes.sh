#!/usr/bin/env bash
set -euo pipefail

# 切换到项目根目录。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# 自动加载挂载盘上的缓存和实验存储配置。
[[ -f "$ROOT/.minitrain-storage.env" ]] && source "$ROOT/.minitrain-storage.env"

# 读取实验变体、训练策略、探针阶段和进程数。
VARIANT="${1:-single}"
STRATEGY="${2:-single}"
STAGE="${STAGE:-smoke}"
if [[ -z "${NPROC+x}" ]]; then
  [[ "$STRATEGY" == "single" ]] && NPROC=1 || NPROC=8
fi

# 生成训练运行名并解析要读取的已提交检查点。
RUN_NAME="synbios_moe_${VARIANT}_${STRATEGY}"
[[ "$STRATEGY" != "single" ]] && RUN_NAME="${RUN_NAME}_${NPROC}gpu"
CHECKPOINT="${3:-latest}"
if [[ "$CHECKPOINT" == "latest" ]]; then
  CHECKPOINT="$(find "artifacts/synbios_moe/checkpoints/$RUN_NAME" \
    -mindepth 2 -maxdepth 2 -name COMMITTED -printf '%h\n' | sort | tail -n 1)"
fi
if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT/COMMITTED" ]]; then
  echo "No committed checkpoint found for $RUN_NAME" >&2
  exit 2
fi

# 定义数据、缓存、模型和探针输出目录。
DATA="artifacts/synbios_moe/$VARIANT"
CACHE="$DATA/probe_cache"
MODEL="configs/synbios_moe/model.yaml"
OUTPUT="artifacts/synbios_moe/results/${RUN_NAME#synbios_moe_}/probe_pipeline"

# 首次运行时生成探针缓存，并在使用前校验缓存。
if [[ ! -f "$CACHE/manifest.json" ]]; then
  cache_args=(cache-probes --data "$DATA" --output "$CACHE")
  [[ "${REQUIRE_COVERAGE:-1}" == "1" ]] && cache_args+=(--require-coverage)
  python scripts/synbios_moe.py "${cache_args[@]}"
fi
python scripts/synbios_moe.py validate-probe-cache --probe-cache "$CACHE"

# 选择探针任务使用的设备或 GPU 数量。
device_args=(--devices "${PROBE_DEVICES:-auto}")
if [[ "${PROBE_DEVICES:-auto}" == "auto" ]]; then
  device_args+=(--num-gpus "${PROBE_GPUS:-$NPROC}")
fi

# 组装 smoke、pilot 或 formal 阶段的探针流水线参数。
pipeline_args=(
  probe-pipeline
  --stage "$STAGE"
  --data "$DATA"
  --probe-cache "$CACHE"
  --model-config "$MODEL"
  --checkpoint "$CHECKPOINT"
  --output "$OUTPUT"
  "${device_args[@]}"
)
[[ "${SKIP_PRETRAIN_GATE:-0}" == "1" ]] && pipeline_args+=(--skip-gate)
[[ "${IGNORE_STAGE_PREREQUISITE:-0}" == "1" ]] && pipeline_args+=(--ignore-prerequisite)
[[ "${REQUIRE_COVERAGE:-1}" == "1" ]] && pipeline_args+=(--require-coverage)

# 启动 P/Q 探针训练与验证流水线。
python scripts/synbios_moe.py "${pipeline_args[@]}"

# formal 阶段完成后逐属性执行只读的路由分析。
if [[ "$STAGE" == "formal" && "${RUN_ROUTER_ANALYSIS:-1}" == "1" ]]; then
  analysis_device="${ANALYSIS_DEVICE:-cuda:0}"
  mkdir -p "$OUTPUT/formal/router"
  for attribute in birth_date birth_city university major company company_city; do
    router_output="$OUTPUT/formal/router/${attribute}.json"
    [[ -f "$router_output" ]] && continue
    python scripts/synbios_moe.py analyze \
      --data "$DATA" --probe-cache "$CACHE" --model-config "$MODEL" \
      --checkpoint "$CHECKPOINT" --attribute "$attribute" --target first \
      --device "$analysis_device" --output "$router_output"
  done
fi
