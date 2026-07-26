#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/.minitrain-storage.env" ]] && source "$ROOT/.minitrain-storage.env"

RUN_ID="${SYNBIOS_BACKEND_BENCHMARK_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
FIXED_OUTPUT="${FIXED_OUTPUT:-artifacts/distributed_benchmark/synbios_backend_fixed/$RUN_ID}"
CAPACITY_OUTPUT="${CAPACITY_OUTPUT:-artifacts/distributed_benchmark/synbios_backend_capacity/$RUN_ID}"
CONFIG="${CASE_CONFIG:-configs/synbios_moe/runs/multi5_permute_fsdp_4gpu.yaml}"
MODEL="${MODEL_CONFIG:-configs/synbios_moe/model.yaml}"
mkdir -p "$FIXED_OUTPUT" "$CAPACITY_OUTPUT"

RECOVERY_ARGS=()
if [[ "${SYNBIOS_BENCHMARK_REUSE_INTERRUPTED:-0}" == "1" ]]; then
  RECOVERY_ARGS+=(--reuse-failures --reuse-stale-results)
fi

mapfile -t gpu_memory < <(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
[[ "${#gpu_memory[@]}" -eq 4 ]] || {
  echo "the formal backend benchmark requires exactly four visible GPUs" >&2
  exit 2
}
for gpu in 0 1 2 3; do
  used="${gpu_memory[$gpu]//[[:space:]]/}"
  (( used <= 1024 )) || {
    echo "GPU $gpu is not idle: ${used} MiB used" >&2
    exit 2
  }
done

python scripts/run_dist_bench.py validate-backend \
  --device cuda:0 --batch-size 2 --sequence-length 512 \
  --output "$FIXED_OUTPUT/backend_validation.json"

python scripts/run_dist_bench.py run \
  --suite capacity --strategies fsdp --world-sizes 4 \
  --ops-backends torch triton cuda \
  --batch-sizes 1 2 4 8 16 24 32 48 64 80 96 112 120 128 \
  --warmup-steps 5 --measure-steps 20 --repeats 2 \
  "${RECOVERY_ARGS[@]}" \
  --case-config "$CONFIG" --model-config "$MODEL" \
  --output "$CAPACITY_OUTPUT"

python scripts/run_dist_bench.py present-capacity \
  --input "$CAPACITY_OUTPUT/capacity_summary.json" \
  --memory-limit-percent 92 --min-repeats 2 --require-batch-one \
  --output "$CAPACITY_OUTPUT/presentation"

COMMON_BATCH="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["common_fixed_batch"])' \
    "$CAPACITY_OUTPUT/presentation/capacity_backend_comparison.json"
)"

python scripts/run_dist_bench.py run \
  --suite backend --strategies fsdp --world-sizes 4 \
  --ops-backends torch triton cuda --local-batch "$COMMON_BATCH" \
  --warmup-steps 10 --measure-steps 30 --repeats 3 \
  --case-config "$CONFIG" --model-config "$MODEL" \
  --output "$FIXED_OUTPUT"

python scripts/run_dist_bench.py present-backend \
  --input "$FIXED_OUTPUT/backend_summary.json" \
  --validation "$FIXED_OUTPUT/backend_validation.json" \
  --output "$FIXED_OUTPUT/presentation"

bash scripts/bash/export_test_results.sh
echo "common_fixed_batch=$COMMON_BATCH"
echo "fixed_workload=$FIXED_OUTPUT"
echo "fixed_space=$CAPACITY_OUTPUT"
