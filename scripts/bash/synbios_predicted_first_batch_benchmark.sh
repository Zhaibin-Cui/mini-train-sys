#!/usr/bin/env bash
set -euo pipefail

# Enter the repository and load mounted-disk paths.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/.minitrain-storage.env" ]] && source "$ROOT/.minitrain-storage.env"

# Resolve the multi5+permute inputs used by the final pilot.
RUN_NAME="${1:-multi5_permute_fsdp_4gpu}"
CHECKPOINT="${2:-latest}"
DATA="artifacts/synbios_moe/multi5_permute"
CACHE="$DATA/probe_cache"
FORMAL="artifacts/synbios_moe/results/$RUN_NAME/probe_pipeline/formal"
PROBE_DIR="${FIRST_PROBE_DIR:-$FORMAL/training}"
MODEL="${MODEL_CONFIG:-configs/synbios_moe/model.yaml}"
if [[ "$CHECKPOINT" == "latest" ]]; then
  CHECKPOINT="$(find "artifacts/synbios_moe/checkpoints/synbios_moe_$RUN_NAME" \
    -mindepth 2 -maxdepth 2 -name COMMITTED -printf '%h\n' | sort | tail -n 1)"
fi
[[ -n "$CHECKPOINT" && -f "$CHECKPOINT/COMMITTED" ]] || {
  echo "missing committed checkpoint: $CHECKPOINT" >&2
  exit 2
}
[[ -f "$PROBE_DIR/q_university_first.pt" ]] || {
  echo "missing formal first probes: $PROBE_DIR" >&2
  exit 2
}

# Candidate lists are independent because P biographies and Q name queries have different lengths.
P_BATCHES="${P_BATCHES:-128,256,384,512,768,1024}"
Q_BATCHES="${Q_BATCHES:-1024,2048,4096,6144,8192,12288,16384}"
P_VALIDATION_BATCHES="${P_VALIDATION_BATCHES:-256,512,768,1024,1536,2048}"
Q_VALIDATION_BATCHES="${Q_VALIDATION_BATCHES:-2048,4096,6144,8192,12288,16384}"
MEMORY_LIMIT_PERCENT="${PROBE_MEMORY_LIMIT_PERCENT:-92}"
WARMUP_STEPS="${PROBE_BENCHMARK_WARMUP_STEPS:-3}"
MEASURE_STEPS="${PROBE_BENCHMARK_MEASURE_STEPS:-10}"
PREDICTION_BATCH_SIZE="${PREDICTION_BATCH_SIZE:-512}"
RUN_ID="${PREDICTED_PROBE_BENCHMARK_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT="${PREDICTED_PROBE_BENCHMARK_OUTPUT:-artifacts/synbios_moe/results/predicted_first_batch_benchmark/$RUN_ID}"
mkdir -p "$OUTPUT/logs"
cleanup() {
  while read -r pid; do kill "$pid" 2>/dev/null || true; done < <(jobs -pr)
}
trap cleanup EXIT INT TERM

# Reject a contaminated measurement instead of benchmarking beside another GPU job.
mapfile -t gpu_memory < <(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
[[ "${#gpu_memory[@]}" -ge 4 ]] || {
  echo "the replicated batch benchmark requires four visible GPUs" >&2
  exit 2
}
for gpu in 0 1 2 3; do
  used="${gpu_memory[$gpu]//[[:space:]]/}"
  (( used <= 1024 )) || {
    echo "GPU $gpu is not idle: ${used} MiB used" >&2
    exit 2
  }
done

# Run two independent GPU replicas for each kind and mode.
run_mode() {
  local mode="$1" p_candidates="$2" q_candidates="$3" status=0
  local specification kind gpu replica candidates result pid
  local pids=()
  for specification in \
    "p:0:a:$p_candidates" "q:1:a:$q_candidates" \
    "p:2:b:$p_candidates" "q:3:b:$q_candidates"; do
    IFS=: read -r kind gpu replica candidates <<<"$specification"
    result="$OUTPUT/${kind}_${replica}_${mode}.json"
    python scripts/synbios_moe.py benchmark-predicted-first-whole-batches \
      --data "$DATA" --probe-cache "$CACHE" --probe-dir "$PROBE_DIR" \
      --model-config "$MODEL" --checkpoint "$CHECKPOINT" \
      --kind "$kind" --attribute university --mode "$mode" \
      --batch-sizes "$candidates" --prediction-batch-size "$PREDICTION_BATCH_SIZE" \
      --warmup-steps "$WARMUP_STEPS" --measure-steps "$MEASURE_STEPS" \
      --memory-limit-percent "$MEMORY_LIMIT_PERCENT" \
      --device "cuda:$gpu" --output "$result" \
      >"$OUTPUT/logs/${kind}_${replica}_${mode}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  (( status == 0 ))
}

run_mode training "$P_BATCHES" "$Q_BATCHES"
run_mode validation "$P_VALIDATION_BATCHES" "$Q_VALIDATION_BATCHES"

# Publish settings only when every recommendation is reproduced and bracketed.
python scripts/synbios_moe.py summarize-probe-benchmarks \
  --run "$OUTPUT/p_a_training.json" --run "$OUTPUT/p_b_training.json" \
  --run "$OUTPUT/q_a_training.json" --run "$OUTPUT/q_b_training.json" \
  --run "$OUTPUT/p_a_validation.json" --run "$OUTPUT/p_b_validation.json" \
  --run "$OUTPUT/q_a_validation.json" --run "$OUTPUT/q_b_validation.json" \
  --output "$OUTPUT/summary.json" --env-output "$OUTPUT/recommended.env" \
  --require-complete-search

echo "batch search complete: source $OUTPUT/recommended.env"
