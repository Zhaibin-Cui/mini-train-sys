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

# 分布式服务器运行必须复用容量回归产出的四个 batch；本地单卡调试保留论文默认值。
if [[ -n "${PROBE_BATCH_ENV:-}" ]]; then
  [[ -f "$PROBE_BATCH_ENV" ]] || { echo "missing PROBE_BATCH_ENV: $PROBE_BATCH_ENV" >&2; exit 2; }
  source "$PROBE_BATCH_ENV"
fi
if [[ "$STRATEGY" != "single" && "${ALLOW_DEFAULT_PROBE_BATCHES:-0}" != "1" ]]; then
  for batch_variable in P_BATCH_SIZE Q_BATCH_SIZE P_VALIDATION_BATCH_SIZE Q_VALIDATION_BATCH_SIZE; do
    [[ -n "${!batch_variable:-}" ]] || {
      echo "distributed probe requires $batch_variable from a completed batch benchmark" >&2
      echo "set PROBE_BATCH_ENV=<benchmark>/recommended.env" >&2
      exit 2
    }
  done
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

# 缓存缺失、损坏或协议过期时自动重建，并在使用前再次校验。
if [[ ! -f "$CACHE/manifest.json" ]] || \
  ! python scripts/synbios_moe.py validate-probe-cache --probe-cache "$CACHE" --data "$DATA" >/dev/null 2>&1; then
  cache_args=(cache-probes --data "$DATA" --output "$CACHE")
  [[ -e "$CACHE" ]] && cache_args+=(--force)
  [[ "${REQUIRE_COVERAGE:-1}" == "1" ]] && cache_args+=(--require-coverage)
  python scripts/synbios_moe.py "${cache_args[@]}"
fi
python scripts/synbios_moe.py validate-probe-cache --probe-cache "$CACHE" --data "$DATA"

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
[[ -n "${P_BATCH_SIZE:-}" ]] && pipeline_args+=(--p-batch-size "$P_BATCH_SIZE")
[[ -n "${Q_BATCH_SIZE:-}" ]] && pipeline_args+=(--q-batch-size "$Q_BATCH_SIZE")
[[ -n "${P_VALIDATION_BATCH_SIZE:-}" ]] && pipeline_args+=(--p-validation-batch-size "$P_VALIDATION_BATCH_SIZE")
[[ -n "${Q_VALIDATION_BATCH_SIZE:-}" ]] && pipeline_args+=(--q-validation-batch-size "$Q_VALIDATION_BATCH_SIZE")
[[ -n "${PROBE_CHECKPOINT_INTERVAL:-}" ]] && pipeline_args+=(--checkpoint-interval-steps "$PROBE_CHECKPOINT_INTERVAL")
[[ -n "${PROBE_LOG_INTERVAL:-}" ]] && pipeline_args+=(--log-interval "$PROBE_LOG_INTERVAL")
[[ -n "${PROBE_HEARTBEAT_SECONDS:-}" ]] && pipeline_args+=(--heartbeat-seconds "$PROBE_HEARTBEAT_SECONDS")

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
