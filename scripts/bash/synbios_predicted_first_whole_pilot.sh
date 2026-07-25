#!/usr/bin/env bash
set -euo pipefail

# Enter the repository and load mounted-disk paths.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/.minitrain-storage.env" ]] && source "$ROOT/.minitrain-storage.env"

# Resolve the supported condition, formal first probes, and backbone.
VARIANT="${1:-multi5_permute}"
[[ "$VARIANT" == "multi5_permute" ]] || {
  echo "this pilot is intentionally restricted to multi5_permute" >&2
  exit 2
}
RUN_NAME="${2:-${VARIANT}_fsdp_4gpu}"
CHECKPOINT="${3:-latest}"
DATA="artifacts/synbios_moe/$VARIANT"
CACHE="$DATA/probe_cache"
FORMAL="artifacts/synbios_moe/results/$RUN_NAME/probe_pipeline/formal"
PROBE_DIR="${FIRST_PROBE_DIR:-$FORMAL/training}"
OUTPUT="${OUTPUT:-$FORMAL/diagnostics/predicted_first_whole_pilot}"
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
python scripts/synbios_moe.py validate-probe-cache --probe-cache "$CACHE" --data "$DATA"
mkdir -p "$OUTPUT"

# Tune P and Q independently; defaults retain the pilot-scale protocol.
STEPS="${STEPS:-3000}"
P_BATCH_SIZE="${P_BATCH_SIZE:-128}"
Q_BATCH_SIZE="${Q_BATCH_SIZE:-768}"
P_EVAL_BATCH_SIZE="${P_EVAL_BATCH_SIZE:-${P_VALIDATION_BATCH_SIZE:-512}}"
Q_EVAL_BATCH_SIZE="${Q_EVAL_BATCH_SIZE:-${Q_VALIDATION_BATCH_SIZE:-6144}}"
CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-1000}"
LOG_INTERVAL_STEPS="${LOG_INTERVAL_STEPS:-100}"
GPUS="${GPUS:-0 1 2 3}"
read -r -a gpu_ids <<<"$GPUS"
(( ${#gpu_ids[@]} > 0 )) || { echo "GPUS must not be empty" >&2; exit 2; }

# Build the fixed ten-task multi5+permute matrix.
tasks=()
for kind in p q; do
  for attribute in birth_city university major company company_city; do
    tasks+=("$kind:$attribute")
  done
done

# Run one classifier per GPU and wait for each wave to finish.
pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
for index in "${!tasks[@]}"; do
  IFS=: read -r kind attribute <<<"${tasks[$index]}"
  gpu="${gpu_ids[$((index % ${#gpu_ids[@]}))]}"
  if [[ "$kind" == "p" ]]; then
    batch_size="$P_BATCH_SIZE"
    evaluation_batch_size="$P_EVAL_BATCH_SIZE"
  else
    batch_size="$Q_BATCH_SIZE"
    evaluation_batch_size="$Q_EVAL_BATCH_SIZE"
  fi
  output="$OUTPUT/${kind}_${attribute}.json"
  recovery="$OUTPUT/${kind}_${attribute}.recovery.pt"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/synbios_moe.py train-predicted-first-whole \
    --data "$DATA" --probe-cache "$CACHE" --probe-dir "$PROBE_DIR" \
    --model-config "$MODEL" --checkpoint "$CHECKPOINT" \
    --kind "$kind" --attribute "$attribute" --steps "$STEPS" \
    --batch-size "$batch_size" --evaluation-batch-size "$evaluation_batch_size" \
    --checkpoint-interval-steps "$CHECKPOINT_INTERVAL_STEPS" \
    --log-interval "$LOG_INTERVAL_STEPS" \
    --device cuda --output "$output" --recovery-checkpoint "$recovery" &
  pids+=("$!")
  if (( ${#pids[@]} == ${#gpu_ids[@]} )); then
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

# Validate all tasks and render CSV, JSON, PNG, and PDF summaries.
python scripts/synbios_moe.py summarize-predicted-first-whole --run "$OUTPUT"
