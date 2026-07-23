#!/usr/bin/env bash
set -euo pipefail

# 在四张空闲 GPU 上复测最坏任务，并只发布已找到安全上界的 batch。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/.minitrain-storage.env" ]] && source "$ROOT/.minitrain-storage.env"

VARIANT="${1:-multi5_permute}"
CHECKPOINT="${2:?usage: $0 VARIANT CHECKPOINT}"
[[ "$VARIANT" == "single" || "$VARIANT" == "multi5_permute" ]] || { echo "invalid variant: $VARIANT" >&2; exit 2; }
[[ -f "$CHECKPOINT/COMMITTED" && -f "$CHECKPOINT/model.pt" ]] || { echo "checkpoint must contain COMMITTED and model.pt: $CHECKPOINT" >&2; exit 2; }
DATA="artifacts/synbios_moe/$VARIANT"
CACHE="$DATA/probe_cache"
MODEL="configs/synbios_moe/model.yaml"
RUN_ID="${PROBE_BENCHMARK_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT="${PROBE_BENCHMARK_OUTPUT:-artifacts/synbios_moe/results/probe_batch_benchmark/$VARIANT/$RUN_ID}"
P_BATCHES="${P_BATCHES:-32,50,64,80,96}"
Q_BATCHES="${Q_BATCHES:-128,200,256,320,384}"
P_VALIDATION_BATCHES="${P_VALIDATION_BATCHES:-64,96,128,160,192}"
Q_VALIDATION_BATCHES="${Q_VALIDATION_BATCHES:-256,384,512,640,768}"
MEMORY_LIMIT_PERCENT="${PROBE_MEMORY_LIMIT_PERCENT:-92}"
WARMUP_STEPS="${PROBE_BENCHMARK_WARMUP_STEPS:-3}"
MEASURE_STEPS="${PROBE_BENCHMARK_MEASURE_STEPS:-10}"
mkdir -p "$OUTPUT/logs"
trap 'echo "probe benchmark directory: $OUTPUT"' EXIT

# 首次运行自动建立缓存，并确认缓存确实属于当前数据。
if [[ ! -f "$DATA/manifest.json" || ! -f "$DATA/profiles.jsonl" ]]; then
  echo "missing prepared SynBioS data: $DATA" >&2
  exit 2
fi
if [[ ! -f "$CACHE/manifest.json" ]] || \
  ! python scripts/synbios_moe.py validate-probe-cache --probe-cache "$CACHE" --data "$DATA" >/dev/null 2>&1; then
  cache_args=(cache-probes --data "$DATA" --output "$CACHE" --require-coverage)
  [[ -e "$CACHE" ]] && cache_args+=(--force)
  python scripts/synbios_moe.py "${cache_args[@]}"
fi
python scripts/synbios_moe.py validate-probe-cache --probe-cache "$CACHE" --data "$DATA"

# 有其他作业占卡会污染吞吐和安全阈值，默认超过 1 GiB 就拒绝压测。
MAX_IDLE_MEMORY_MB="${PROBE_BENCHMARK_MAX_IDLE_MEMORY_MB:-1024}"
mapfile -t gpu_memory < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[[ "${#gpu_memory[@]}" -ge 4 ]] || { echo "probe benchmark requires four visible GPUs" >&2; exit 2; }
for gpu in 0 1 2 3; do
  used="${gpu_memory[$gpu]//[[:space:]]/}"
  if (( used > MAX_IDLE_MEMORY_MB )); then
    echo "GPU $gpu is not idle: ${used} MiB used (limit ${MAX_IDLE_MEMORY_MB} MiB)" >&2
    exit 2
  fi
done

# 每轮在四张卡上复测 P/Q；训练和验证分开测，避免 forward-only batch 过于保守。
run_mode() {
  local mode="$1" p_candidates="$2" q_candidates="$3" status=0
  local specification kind gpu replica candidates result pid
  local pids=()
  for specification in "p:0:a:$p_candidates" "q:1:a:$q_candidates" "p:2:b:$p_candidates" "q:3:b:$q_candidates"; do
    IFS=: read -r kind gpu replica candidates <<<"$specification"
    result="$OUTPUT/${kind}_${replica}_${mode}.json"
    python scripts/synbios_moe.py benchmark-probe-batches \
      --data "$DATA" --probe-cache "$CACHE" --model-config "$MODEL" \
      --checkpoint "$CHECKPOINT" --kind "$kind" --mode "$mode" \
      --attribute university --target whole --batch-sizes "$candidates" \
      --warmup-steps "$WARMUP_STEPS" --measure-steps "$MEASURE_STEPS" \
      --memory-limit-percent "$MEMORY_LIMIT_PERCENT" \
      --device "cuda:$gpu" --output "$result" \
      >"$OUTPUT/logs/${kind}_${replica}_${mode}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  [[ "$status" == "0" ]] || return 1
}

run_mode training "$P_BATCHES" "$Q_BATCHES" || { echo "training batch benchmark failed; inspect $OUTPUT/logs" >&2; exit 1; }
run_mode validation "$P_VALIDATION_BATCHES" "$Q_VALIDATION_BATCHES" || { echo "validation batch benchmark failed; inspect $OUTPUT/logs" >&2; exit 1; }

python scripts/synbios_moe.py summarize-probe-benchmarks \
  --run "$OUTPUT/p_a_training.json" --run "$OUTPUT/p_b_training.json" \
  --run "$OUTPUT/q_a_training.json" --run "$OUTPUT/q_b_training.json" \
  --run "$OUTPUT/p_a_validation.json" --run "$OUTPUT/p_b_validation.json" \
  --run "$OUTPUT/q_a_validation.json" --run "$OUTPUT/q_b_validation.json" \
  --output "$OUTPUT/summary.json" --env-output "$OUTPUT/recommended.env" \
  --require-complete-search

echo "batch search complete; run: source $OUTPUT/recommended.env"
