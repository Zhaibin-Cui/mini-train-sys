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
  "$DEST/logs" \
  "$DEST/smoke" \
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
copy_tree "$ROOT/artifacts/logs" "$DEST/logs"
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
  if [[ -f "$source_root/token_shards/manifest.json" ]]; then
    mkdir -p "$target_root/token_shards"
    cp "$source_root/token_shards/manifest.json" "$target_root/token_shards/manifest.json"
  fi
done

# Persist formal metrics and recovery metadata, never multi-gigabyte tensor
# payloads. COMMITTED/runtime/RNG files prove a checkpoint was publishable.
copy_tree "$ROOT/artifacts/synbios_moe/runs" "$DEST/formal_runs/synbios_moe/runs"
copy_tree "$ROOT/artifacts/synbios_moe/results" "$DEST/formal_runs/synbios_moe/results"
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

# Hash every exported file so a Git snapshot can be checked independently of
# the mounted artifact volume. Exclude the manifest itself to avoid recursion.
find "$DEST" -type f ! -name MANIFEST.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  | sed "s#$ROOT/##" \
  > "$DEST/MANIFEST.sha256"

echo "Exported Git-trackable test results to $DEST"
