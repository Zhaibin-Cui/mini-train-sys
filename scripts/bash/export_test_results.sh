#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/.minitrain-storage.env" ]] && source "$ROOT/.minitrain-storage.env"

DEST="$ROOT/results"
mkdir -p \
  "$DEST/benchmarks" \
  "$DEST/datasets" \
  "$DEST/environment" \
  "$DEST/formal_runs" \
  "$DEST/logs/benchmarks" \
  "$DEST/logs/experiments" \
  "$DEST/logs/maintenance" \
  "$DEST/logs/validation" \
  "$DEST/notebooks" \
  "$DEST/smoke" \
  "$DEST/tensorboard" \
  "$DEST/validation"

copy_tree() {
  local source="$1"
  local target="$2"
  if [[ -d "$source" ]]; then
    mkdir -p "$target"
    # Results are an append-only provenance archive. Active artifacts may be
    # pruned or reset between runs; never mirror those deletions into Git.
    rsync -a "$source/" "$target/"
  fi
}

copy_tree "$ROOT/artifacts/distributed_benchmark" "$DEST/benchmarks"
copy_tree "$ROOT/artifacts/operator_benchmark" "$DEST/benchmarks/operator_benchmark"
log_category() {
  local lowered
  lowered="$(basename "$1" | tr '[:upper:]' '[:lower:]')"
  case "$lowered" in
    *kernel*|*backend_benchmark*|*distributed_server_benchmark*|*capacity*|*b112_stability*|*weak_scal*|*cuda_build*)
      echo "benchmarks"
      ;;
    *probe*|*cloze*|*synbios*|*ground_truth*|*tensorboard*)
      echo "experiments"
      ;;
    *test*|*regression*|*preflight*|*prepush*|*fidelity*|*validation*|*quality*)
      echo "validation"
      ;;
    *)
      echo "maintenance"
      ;;
  esac
}

place_log() {
  local source="$1"
  local mode="$2"
  local category target
  category="$(log_category "$source")"
  target="$DEST/logs/$category/$(basename "$source")"
  mkdir -p "$(dirname "$target")"
  if [[ "$source" == "$target" ]]; then
    return
  fi
  if [[ -e "$target" ]]; then
    if [[ "$mode" == "copy" ]]; then
      # The mounted server log is authoritative and may have grown since the
      # previous export.
      cp -p "$source" "$target"
      return
    fi
    local source_size target_size common_size
    source_size="$(stat -c '%s' "$source")"
    target_size="$(stat -c '%s' "$target")"
    common_size="$source_size"
    if (( target_size < common_size )); then
      common_size="$target_size"
    fi
    if (( common_size > 0 )); then
      cmp -n "$common_size" "$source" "$target"
    fi
    if (( source_size > target_size )); then
      mv "$source" "$target"
    else
      rm "$source"
    fi
  elif [[ "$mode" == "move" ]]; then
    mv "$source" "$target"
  else
    cp -p "$source" "$target"
  fi
}

# Migrate earlier flat exports without losing evidence, then place all current
# server logs into purpose-specific directories.
while IFS= read -r -d '' existing; do
  [[ "$(basename "$existing")" == "README.md" ]] && continue
  place_log "$existing" move
done < <(find "$DEST/logs" -maxdepth 1 -type f -print0)
if [[ -d "$ROOT/artifacts/logs" ]]; then
  while IFS= read -r -d '' source; do
    place_log "$source" copy
  done < <(find "$ROOT/artifacts/logs" -type f -print0)
fi

# Executed notebooks are compact, reproducible server evidence. Source
# Source notebooks remain under benchmarks/ or examples/; only executed copies and logs land here.
copy_tree "$ROOT/artifacts/notebooks" "$DEST/notebooks"
# Preserve validation reports, event logs, runtime/RNG metadata, and COMMITTED
# markers in Git. Multi-gigabyte DCP shards and model exports remain on the
# mounted artifact volume and are intentionally not duplicated into Git.
if [[ -d "$ROOT/artifacts/validation" ]]; then
  rsync -a \
    --exclude='distributed/*.distcp' \
    --exclude='model.pt' \
    "$ROOT/artifacts/validation/" "$DEST/validation/"
fi
copy_tree "$ROOT/artifacts/smoke" "$DEST/smoke"

if [[ -f "$ROOT/artifacts/server_environment.json" ]]; then
  cp "$ROOT/artifacts/server_environment.json" "$DEST/environment/server_environment.json"
fi

# Data payloads stay on the mounted disk. Their generation and tokenizer
# manifests are small, sufficient to bind every formal run to exact bytes.
for variant in single multi5_permute; do
  source_root="$ROOT/artifacts/synbios_moe/$variant"
  target_root="$DEST/datasets/synbios_moe/$variant"
  if [[ -f "$source_root/manifest.json" ]]; then
    mkdir -p "$target_root"
    cp "$source_root/manifest.json" "$target_root/manifest.json"
  fi
  if [[ -f "$source_root/lineage.json" ]]; then
    mkdir -p "$target_root"
    cp "$source_root/lineage.json" "$target_root/lineage.json"
  fi
  if [[ -f "$source_root/token_shards/manifest.json" ]]; then
    mkdir -p "$target_root/token_shards"
    cp "$source_root/token_shards/manifest.json" "$target_root/token_shards/manifest.json"
  fi
  if [[ -f "$source_root/token_shards/lineage.json" ]]; then
    mkdir -p "$target_root/token_shards"
    cp "$source_root/token_shards/lineage.json" "$target_root/token_shards/lineage.json"
  fi
  if [[ -f "$source_root/probe_cache/manifest.json" ]]; then
    mkdir -p "$target_root/probe_cache"
    cp "$source_root/probe_cache/manifest.json" "$target_root/probe_cache/manifest.json"
  fi
  if [[ -f "$source_root/probe_cache/lineage.json" ]]; then
    mkdir -p "$target_root/probe_cache"
    cp "$source_root/probe_cache/lineage.json" "$target_root/probe_cache/lineage.json"
  fi
done

# Probe capacity sweeps are benchmarks, not formal probe conclusions.
copy_tree \
  "$ROOT/artifacts/synbios_moe/results/probe_batch_benchmark" \
  "$DEST/benchmarks/synbios_moe/probe_batch_benchmark"
copy_tree \
  "$ROOT/artifacts/synbios_moe/results/ground_truth_first_batch_benchmark" \
  "$DEST/benchmarks/synbios_moe/ground_truth_first_batch_benchmark"

# Persist formal metrics and recovery metadata, never multi-gigabyte tensor
# payloads. COMMITTED/runtime/RNG files prove a checkpoint was publishable.
copy_tree "$ROOT/artifacts/synbios_moe/runs" "$DEST/formal_runs/synbios_moe/runs"
if [[ -d "$ROOT/artifacts/synbios_moe/results" ]]; then
  mkdir -p "$DEST/formal_runs/synbios_moe/results"
  rsync -a \
    --exclude='probe_batch_benchmark/' \
    --exclude='ground_truth_first_batch_benchmark/' \
    --exclude='*.pt' \
    --exclude='*/diagnostics/*/records.csv' \
    --exclude='*/diagnostics/*/bad_cases.csv' \
    --exclude='*/diagnostics/*/route_records.csv' \
    "$ROOT/artifacts/synbios_moe/results/" "$DEST/formal_runs/synbios_moe/results/"
fi
# Older exports predated the weight-exclusion rule and may still contain probe
# head tensors because this archive intentionally does not mirror deletions.
# Keep their JSON identities/hashes, but remove only these Git-inappropriate
# tensor copies; the authoritative heads remain under artifacts/ on /data.
find "$DEST/formal_runs/synbios_moe/results" -type f \
  \( -path '*/probe_pipeline/*/training/*.pt' \
     -o -path '*/probe_pipeline/*/recovery/*.pt' \) \
  -delete
copy_tree \
  "$ROOT/artifacts/synbios_moe/operation_logs" \
  "$DEST/formal_runs/synbios_moe/operation_logs"
if [[ -d "$ROOT/artifacts/synbios_moe/checkpoints" ]]; then
  mkdir -p "$DEST/formal_runs/synbios_moe/checkpoints"
  rsync -a \
    --exclude='distributed/*.distcp' \
    --exclude='model.pt' \
    "$ROOT/artifacts/synbios_moe/checkpoints/" \
    "$DEST/formal_runs/synbios_moe/checkpoints/"
fi

# Preserve generic/local smoke runs created before the server-specific artifact
# roots are selected. Export only recovery metadata from checkpoints.
copy_tree "$ROOT/runs" "$DEST/smoke/local_runs"
if [[ -d "$ROOT/checkpoints/rtx4090_single_1gpu" ]]; then
  mkdir -p "$DEST/smoke/checkpoints/rtx4090_single_1gpu"
  rsync -a \
    --exclude='distributed/*.distcp' \
    --exclude='model.pt' \
    "$ROOT/checkpoints/rtx4090_single_1gpu/" \
    "$DEST/smoke/checkpoints/rtx4090_single_1gpu/"
fi

# Prove that every file covered by the export policy has an identical
# destination before building the human/machine catalog.
python "$ROOT/scripts/audit_results_export.py" \
  --repo-root "$ROOT" \
  --artifacts "$ROOT/artifacts" \
  --results "$DEST" \
  --output "$DEST/catalog/export_audit.json"

# Generate a human/machine catalog, a central TensorBoard event index, and a
# retention inventory for large payloads that intentionally remain on /data.
python "$ROOT/scripts/build_results_catalog.py" \
  --results "$DEST" \
  --artifacts "$ROOT/artifacts"

# Hash every exported file so a Git snapshot can be checked independently of
# the mounted artifact volume.
python "$ROOT/scripts/build_results_manifest.py" --results "$DEST"

echo "Exported Git-trackable test results to $DEST"
