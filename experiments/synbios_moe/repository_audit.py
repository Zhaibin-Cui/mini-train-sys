"""End-to-end path, lineage, and retained-evidence audit for SynBioS."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import yaml

from experiments.synbios_moe.formal_report import load_formal_run, validate_matched_runs
from experiments.synbios_moe.probe_data import validate_probe_cache
from minitrain.runtime.config import load_yaml_dict


EXPECTED_PROBE_RUNTIME = {
    "training_batch_sizes": {"p": 128, "q": 768},
    "validation_batch_sizes": {"p": 512, "q": 6144},
    "log_interval_steps": 100,
    "heartbeat_seconds": 10,
    "checkpoint_interval_steps": 1000,
    "evaluate_train": True,
}


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty catalog: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _file_record(path: Path, base: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(base).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _verify_manifest_file(root: Path, record: dict[str, object]) -> None:
    path = root / str(record["path"])
    if path.stat().st_size != int(record["num_bytes"]):
        raise ValueError(f"token-shard size mismatch: {path}")
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"token-shard SHA256 mismatch: {path}")


def _dataset_lineage(variant: str, data_root: Path) -> dict[str, object]:
    data_manifest_path = data_root / "manifest.json"
    token_root = data_root / "token_shards"
    token_manifest_path = token_root / "manifest.json"
    cache_root = data_root / "probe_cache"
    cache_manifest_path = cache_root / "manifest.json"
    data_manifest = _read_json(data_manifest_path)
    token_manifest = _read_json(token_manifest_path)
    cache_manifest = _read_json(cache_manifest_path)
    expected_variant = "multi5+permute" if variant == "multi5_permute" else "single"
    if data_manifest.get("variant") != expected_variant:
        raise ValueError(f"{variant} dataset manifest has the wrong variant")
    for name, record in data_manifest["files"].items():
        path = data_root / name
        if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise ValueError(f"{variant} raw dataset file mismatch: {path}")
    train = token_manifest["splits"]["train"]
    if int(train["documents"]) != int(data_manifest["biographies"]):
        raise ValueError(f"{variant} token-shard document count differs from dataset")
    if token_manifest["splits"]["validation"]["documents"] != 0:
        raise ValueError(f"{variant} pretraining token shards unexpectedly have a held-out split")
    for shard in train["shards"]:
        _verify_manifest_file(token_root, shard)
    _verify_manifest_file(token_root, train["document_index"])
    _verify_manifest_file(token_root, token_manifest["splits"]["validation"]["document_index"])
    if cache_manifest.get("source_manifest") != data_manifest:
        raise ValueError(f"{variant} probe-cache parent manifest mismatch")
    cache_validation = validate_probe_cache(cache_root, data_root, include_missing_classes=False)
    if not cache_validation["coverage_complete"]:
        raise ValueError(f"{variant} probe cache has incomplete validation-class coverage")
    cache_files = sorted(
        path
        for path in cache_root.iterdir()
        if path.is_file() and path.name not in {"lineage.json"}
    )
    cache_records = [_file_record(path, cache_root) for path in cache_files]
    token_records = [
        _file_record(path, token_root)
        for path in sorted(token_root.rglob("*"))
        if path.is_file() and path.name != "lineage.json"
    ]
    parent_sha = _sha256(data_manifest_path)
    token_sha = _sha256(token_manifest_path)
    cache_sha = _sha256(cache_manifest_path)
    lineage = {
        "format_version": 1,
        "variant": variant,
        "source_dataset": {
            "path": str(data_root.resolve()),
            "manifest_sha256": parent_sha,
            "profiles_sha256": data_manifest["files"]["profiles.jsonl"]["sha256"],
            "people": int(data_manifest["num_people"]),
            "biographies": int(data_manifest["biographies"]),
        },
        "token_shards": {
            "path": str(token_root.resolve()),
            "manifest_sha256": token_sha,
            "parent_manifest_sha256": parent_sha,
            "documents": int(train["documents"]),
            "tokens": int(train["tokens"]),
            "split_semantics": "all generated biographies are pretraining train documents",
            "files": token_records,
        },
        "probe_cache": {
            "path": str(cache_root.resolve()),
            "manifest_sha256": cache_sha,
            "parent_manifest_sha256": parent_sha,
            "profiles": int(cache_manifest["profiles"]),
            "p_examples": int(cache_manifest["p_examples"]),
            "q_examples": int(cache_manifest["q_examples"]),
            "split_semantics": (
                "deterministic person-level probe train/validation split; "
                "both partitions were seen by backbone pretraining"
            ),
            "coverage_complete": True,
            "files": cache_records,
        },
    }
    _atomic_json(data_root / "lineage.json", lineage)
    _atomic_json(
        token_root / "lineage.json",
        {
            "format_version": 1,
            "variant": variant,
            **lineage["token_shards"],
        },
    )
    _atomic_json(
        cache_root / "lineage.json",
        {
            "format_version": 1,
            "variant": variant,
            **lineage["probe_cache"],
        },
    )
    return lineage


def _validate_config_paths(repo_root: Path, formal_runs) -> dict[str, object]:
    records = {}
    for variant, formal, config_name, expected_epochs in (
        ("single", formal_runs[0], "single_fsdp_4gpu.yaml", 540),
        ("multi5_permute", formal_runs[1], "multi5_permute_fsdp_4gpu.yaml", 108),
    ):
        config_path = repo_root / "configs/synbios_moe/runs" / config_name
        payload = load_yaml_dict(config_path)
        data_path = (repo_root / payload["data"]["path"]).resolve()
        expected_data_path = (
            repo_root / f"artifacts/synbios_moe/{variant}/token_shards/manifest.json"
        ).resolve()
        if data_path != expected_data_path or not data_path.is_file():
            raise ValueError(f"{variant} training config resolves the wrong token manifest")
        if int(payload["train"]["epochs"]) != expected_epochs:
            raise ValueError(f"{variant} training config has the wrong epoch budget")
        if int(payload["train"]["batch_size"]) != 112:
            raise ValueError(f"{variant} training config no longer uses benchmark-selected batch 112")
        if int(payload["parallel"]["expected_world_size"]) != 4:
            raise ValueError(f"{variant} formal config no longer requires four GPUs")
        records[variant] = {
            "config": str(config_path.relative_to(repo_root)),
            "resolved_data_manifest": str(data_path),
            "epochs": expected_epochs,
            "local_batch": 112,
            "world_size": 4,
            "checkpoint": formal.identity["checkpoint"],
        }
    probe_path = repo_root / "configs/synbios_moe/probe_pipeline.yaml"
    probe_payload = yaml.safe_load(probe_path.read_text(encoding="utf-8"))
    if probe_payload["runtime"] != EXPECTED_PROBE_RUNTIME:
        raise ValueError("probe pipeline defaults differ from the accepted formal runtime")
    for formal in formal_runs:
        if formal.identity["runtime"] != EXPECTED_PROBE_RUNTIME:
            raise ValueError(f"{formal.condition} formal identity differs from accepted runtime")
    return {
        "training": records,
        "probe_pipeline": {
            "config": str(probe_path.relative_to(repo_root)),
            "runtime": EXPECTED_PROBE_RUNTIME,
        },
    }


def classify_log(name: str) -> str:
    if name.startswith(("probe_", "formal_probe_")):
        return "probe"
    if "cloze" in name:
        return "pretraining_validation"
    if "benchmark" in name or "capacity" in name or "scaling" in name:
        return "benchmark"
    if "formal" in name and "synbios_moe" in name:
        return "pretraining"
    if "prepare" in name:
        return "dataset"
    if "tensorboard" in name or "export" in name:
        return "infrastructure"
    return "engineering_validation"


def _log_catalog(log_root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(log_root.glob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "category": classify_log(path.name),
                "filename": path.name,
                "bytes": path.stat().st_size,
                "modified_utc": datetime.fromtimestamp(
                    path.stat().st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
                "sha256": _sha256(path),
                "mounted_path": str(path.resolve()),
                "git_safe_path": f"results/logs/{path.name}",
            }
        )
    return rows


def build_repository_audit(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Verify the canonical server experiment graph and emit audit catalogs."""

    root = Path(repo_root).resolve()
    artifacts = root / "artifacts"
    if not artifacts.is_symlink():
        raise ValueError("artifacts must be the mounted-storage symlink")
    if artifacts.resolve() != Path("/data/mini-train-sys/artifacts"):
        raise ValueError(f"artifacts resolves to the unexpected storage root: {artifacts.resolve()}")
    single_root = artifacts / "synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal"
    multi_root = (
        artifacts / "synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal"
    )
    single = load_formal_run("single", single_root)
    multi = load_formal_run("multi5_permute", multi_root)
    validate_matched_runs(single, multi)
    lineages = {
        variant: _dataset_lineage(variant, artifacts / f"synbios_moe/{variant}")
        for variant in ("single", "multi5_permute")
    }
    if (
        lineages["single"]["source_dataset"]["profiles_sha256"]
        != lineages["multi5_permute"]["source_dataset"]["profiles_sha256"]
    ):
        raise ValueError("single and multi5_permute no longer use identical profiles")
    configs = _validate_config_paths(root, (single, multi))
    for formal in (single, multi):
        checkpoint = Path(str(formal.identity["checkpoint"]))
        if not (checkpoint / "COMMITTED").is_file():
            raise ValueError(f"formal checkpoint is not committed: {checkpoint}")
        if _sha256(checkpoint / "model.pt") != formal.identity["checkpoint_model_sha256"]:
            raise ValueError(f"formal checkpoint model hash mismatch: {checkpoint}")
    cloze_paths = {
        "single": artifacts / "synbios_moe/results/single_cloze_eval/full_100k/summary.json",
        "multi5_permute": (
            artifacts
            / "synbios_moe/results/multi5_permute_cloze_eval/full_500k/summary.json"
        ),
    }
    for formal, condition in ((single, "single"), (multi, "multi5_permute")):
        cloze = _read_json(cloze_paths[condition])
        if Path(str(cloze["checkpoint"])).resolve() != Path(
            str(formal.identity["checkpoint"])
        ).resolve():
            raise ValueError(f"{condition} cloze summary uses the wrong checkpoint")
    diagnostic_summary = _read_json(
        multi_root / "diagnostics/report/summary.json"
    )
    if diagnostic_summary.get("status") != "completed":
        raise ValueError("diagnostic report is not completed")
    if (
        diagnostic_summary["identity"]["checkpoint_model_sha256"]
        != multi.identity["checkpoint_model_sha256"]
    ):
        raise ValueError("diagnostic report uses the wrong multi checkpoint")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_rows = _log_catalog(artifacts / "logs")
    _atomic_json(output / "dataset_lineage.json", lineages)
    _atomic_json(output / "path_contract.json", configs)
    _atomic_csv(output / "log_catalog.csv", log_rows)
    checks = [
        "mounted_artifacts_symlink",
        "raw_dataset_file_sizes_and_hashes",
        "token_shard_file_sizes_and_hashes",
        "probe_cache_schema_shapes_offsets_and_class_coverage",
        "dataset_cache_parent_lineage",
        "matched_single_multi_formal_identity",
        "formal_checkpoint_committed_and_model_hash",
        "cloze_checkpoint_identity",
        "diagnostic_checkpoint_identity",
        "training_config_paths_epochs_batch_and_world_size",
        "probe_runtime_matches_completed_formal_runs",
    ]
    summary = {
        "status": "passed",
        "format_version": 1,
        "checks": checks,
        "storage": {
            "repository": str(root),
            "artifacts_symlink": str(artifacts),
            "artifacts_target": str(artifacts.resolve()),
        },
        "conditions": {
            formal.condition: {
                "checkpoint": formal.identity["checkpoint"],
                "checkpoint_model_sha256": formal.identity["checkpoint_model_sha256"],
                "data_manifest_sha256": formal.identity["data_manifest_sha256"],
                "probe_cache_manifest_sha256": formal.identity[
                    "probe_cache_manifest_sha256"
                ],
            }
            for formal in (single, multi)
        },
        "catalogs": {
            "dataset_lineage": "dataset_lineage.json",
            "path_contract": "path_contract.json",
            "logs": "log_catalog.csv",
            "log_count": len(log_rows),
        },
    }
    _atomic_json(output / "summary.json", summary)
    return summary
