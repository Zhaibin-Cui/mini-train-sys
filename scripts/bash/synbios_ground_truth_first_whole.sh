#!/usr/bin/env bash
set -euo pipefail

# Enter the repository and load mounted-disk paths.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/.minitrain-storage.env" ]] && source "$ROOT/.minitrain-storage.env"

# Resolve the requested condition, formal probes, and backbone.
VARIANT="${1:-multi5_permute}"
[[ "$VARIANT" == "single" || "$VARIANT" == "multi5_permute" ]] || {
  echo "VARIANT must be single or multi5_permute" >&2
  exit 2
}
RUN_NAME="${2:-${VARIANT}_fsdp_4gpu}"
CHECKPOINT="${3:-latest}"
DATA="artifacts/synbios_moe/$VARIANT"
CACHE="$DATA/probe_cache"
FORMAL="artifacts/synbios_moe/results/$RUN_NAME/probe_pipeline/formal"
PROBE_DIR="${FIRST_PROBE_DIR:-$FORMAL/training}"
OUTPUT="${OUTPUT:-$FORMAL/diagnostics/ground_truth_first_whole_p}"
MODEL="${MODEL_CONFIG:-configs/synbios_moe/model.yaml}"
if [[ "$CHECKPOINT" == "latest" ]]; then
  CHECKPOINT="$(find "artifacts/synbios_moe/checkpoints/synbios_moe_$RUN_NAME" \
    -mindepth 2 -maxdepth 2 -name COMMITTED -printf '%h\n' | sort | tail -n 1)"
fi
[[ -n "$CHECKPOINT" && -f "$CHECKPOINT/COMMITTED" ]] || {
  echo "missing committed checkpoint: $CHECKPOINT" >&2
  exit 2
}
[[ -f "$PROBE_DIR/p_university_whole.pt" ]] || {
  echo "missing formal whole probes used for rank/checkpoint binding: $PROBE_DIR" >&2
  exit 2
}
python scripts/synbios_moe.py validate-probe-cache --probe-cache "$CACHE" --data "$DATA"
mkdir -p "$OUTPUT"

# The accepted P-only run uses the prior pilot update budget and complete validation.
STEPS="${STEPS:-4000}"
P_BATCH_SIZE="${P_BATCH_SIZE:-128}"
P_EVAL_BATCH_SIZE="${P_EVAL_BATCH_SIZE:-${P_VALIDATION_BATCH_SIZE:-3072}}"
CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-1000}"
LOG_INTERVAL_STEPS="${LOG_INTERVAL_STEPS:-100}"
GPUS="${GPUS:-0 1 2 3}"
read -r -a gpu_ids <<<"$GPUS"
(( ${#gpu_ids[@]} > 0 )) || { echo "GPUS must not be empty" >&2; exit 2; }

# Build the fixed five-task P matrix.
tasks=()
for attribute in birth_city university major company company_city; do
  tasks+=("$attribute")
done

# Run one classifier per GPU. Refill each GPU as soon as its task finishes so
# completed P tasks immediately free a GPU for the next attribute.
pids=()
available_gpus=("${gpu_ids[@]}")
declare -A pid_gpu=()
cleanup() {
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
reap_one() {
  local finished_pid rc=0 pid
  local -a remaining=()
  if wait -n -p finished_pid "${pids[@]}"; then
    rc=0
  else
    rc=$?
  fi
  available_gpus+=("${pid_gpu[$finished_pid]}")
  unset "pid_gpu[$finished_pid]"
  for pid in "${pids[@]}"; do
    [[ "$pid" == "$finished_pid" ]] || remaining+=("$pid")
  done
  pids=("${remaining[@]}")
  return "$rc"
}
for index in "${!tasks[@]}"; do
  attribute="${tasks[$index]}"
  output="$OUTPUT/p_${attribute}.json"
  probe_output="${output%.json}.pt"
  if [[ -f "$output" && -f "$probe_output" ]]; then
    echo "skip completed task: p/$attribute"
    continue
  fi
  while (( ${#available_gpus[@]} == 0 )); do
    if ! reap_one; then
      echo "ground-truth-first worker failed" >&2
      exit 1
    fi
  done
  gpu="${available_gpus[0]}"
  available_gpus=("${available_gpus[@]:1}")
  batch_size="$P_BATCH_SIZE"
  evaluation_batch_size="$P_EVAL_BATCH_SIZE"
  recovery="$OUTPUT/p_${attribute}.recovery.pt"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/synbios_moe.py train-ground-truth-first-whole \
    --data "$DATA" --probe-cache "$CACHE" --probe-dir "$PROBE_DIR" \
    --model-config "$MODEL" --checkpoint "$CHECKPOINT" \
    --attribute "$attribute" --steps "$STEPS" \
    --batch-size "$batch_size" --evaluation-batch-size "$evaluation_batch_size" \
    --checkpoint-interval-steps "$CHECKPOINT_INTERVAL_STEPS" \
    --log-interval "$LOG_INTERVAL_STEPS" \
    --device cuda:0 --output "$output" --recovery-checkpoint "$recovery" &
  pid="$!"
  pids+=("$pid")
  pid_gpu["$pid"]="$gpu"
done
while (( ${#pids[@]} > 0 )); do
  if ! reap_one; then
    echo "ground-truth-first worker failed" >&2
    exit 1
  fi
done

# Validate all tasks and render CSV, JSON, PNG, and PDF summaries.
python scripts/synbios_moe.py summarize-ground-truth-first-whole --run "$OUTPUT"
