#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/.minitrain-storage.env" ]] && source "$ROOT/.minitrain-storage.env"

RUN_NAME="${1:-multi5_permute_fsdp_4gpu}"
CHECKPOINT="${2:-latest}"
OUTPUT="${OUTPUT:?set OUTPUT to a new smoke-result directory}"
DATA="artifacts/synbios_moe/multi5_permute"
CACHE="$DATA/probe_cache"
FORMAL="artifacts/synbios_moe/results/$RUN_NAME/probe_pipeline/formal"
PROBE_DIR="$FORMAL/training"
MODEL="${MODEL_CONFIG:-configs/synbios_moe/model.yaml}"
if [[ "$CHECKPOINT" == "latest" ]]; then
  CHECKPOINT="$(find "artifacts/synbios_moe/checkpoints/synbios_moe_$RUN_NAME" \
    -mindepth 2 -maxdepth 2 -name COMMITTED -printf '%h\n' | sort | tail -n 1)"
fi
[[ -f "$CHECKPOINT/COMMITTED" ]] || {
  echo "missing committed checkpoint: $CHECKPOINT" >&2
  exit 2
}
mkdir -p "$OUTPUT/logs"

run_pass() {
  local pass="$1" status=0
  CUDA_VISIBLE_DEVICES=0 python scripts/synbios_moe.py train-ground-truth-first-whole \
    --data "$DATA" --probe-cache "$CACHE" --probe-dir "$PROBE_DIR" \
    --model-config "$MODEL" --checkpoint "$CHECKPOINT" \
    --attribute university --steps 20 --batch-size 128 \
    --evaluation-batch-size 3072 --checkpoint-interval-steps 10 \
    --max-validation-examples 12288 \
    --log-interval 10 --device cuda:0 \
    --output "$OUTPUT/p_university.json" \
    --recovery-checkpoint "$OUTPUT/p_university.recovery.pt" \
    >"$OUTPUT/logs/p_${pass}.log" 2>&1 &
  local p_pid="$!"
  wait "$p_pid" || status=1
  (( status == 0 ))
}

# Pass one creates an intermediate recovery checkpoint; pass two must consume it exactly.
run_pass initial
run_pass resumed

python - "$OUTPUT" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
path = root / "p_university.json"
payload = json.loads(path.read_text(encoding="utf-8"))
assert payload["protocol"] == "ground_truth_first_whole_rank_matched_v1"
assert payload["ground_truth_first_token"] is True
assert payload["rank"] == 2
assert payload["architecture_match"]["low_rank_embedding_delta"] is True
assert payload["resumed_from_step"] == 10
assert payload["backbone_parameters_updated"] is False
assert payload["validation_examples_evaluated"] == 12288
assert payload["validation_examples_total"] > payload["validation_examples_evaluated"]
assert sum(payload["whole_total_validation_by_source_position"]) == 12288
assert len(payload["whole_correct_validation_by_source_position"]) == 6
assert all(payload["alignment_checks"].values())
assert all(math.isfinite(row["loss"]) for row in payload["loss_curve"])
assert (root / "p_university.pt").is_file()
print("ground-truth-t1 P smoke and exact recovery gate passed")
PY
