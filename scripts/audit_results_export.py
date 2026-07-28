#!/usr/bin/env python3
"""Verify that every pushable server artifact has an identical Git-safe copy."""


import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

try:
    from scripts.build_results_catalog import log_destination
except ModuleNotFoundError:  # Direct `python scripts/audit_results_export.py`.
    from build_results_catalog import log_destination


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_mapping(
    relative: Path, destination: Path
) -> tuple[Path | None, str | None]:
    if relative.name == "model.pt":
        return None, "model_export"
    if relative.suffix == ".distcp":
        return None, "dcp_tensor_shard"
    return destination / relative, None


def artifact_mapping(
    relative: Path, results_root: Path
) -> tuple[Path | None, str | None]:
    """Map one path relative to artifacts/ to its Git-safe destination."""
    parts = relative.parts
    if not parts:
        return None, "directory"
    top = parts[0]
    tail = Path(*parts[1:]) if len(parts) > 1 else Path()

    if top == "distributed_benchmark":
        return results_root / "benchmarks" / tail, None
    if top == "operator_benchmark":
        return results_root / "benchmarks/operator_benchmark" / tail, None
    if top == "logs":
        destination = log_destination(relative.name)
        if destination is None:
            return None, "non_mainline_log"
        return results_root / destination / relative.name, None
    if top == "notebooks":
        return results_root / "benchmarks/notebooks" / tail, None
    if relative == Path("server_environment.json"):
        return results_root / "environment/server_environment.json", None
    if top == "archive":
        return None, "server_archive"
    if top != "synbios_moe":
        return None, "unknown_artifact_root"

    if len(parts) < 2:
        return None, "directory"
    section = parts[1]
    synbios_tail = Path(*parts[2:]) if len(parts) > 2 else Path()
    pretraining_root = results_root / "pretraining/synbios_moe"

    if section in {"single", "multi5_permute"}:
        allowed = {
            Path("manifest.json"),
            Path("lineage.json"),
            Path("token_shards/manifest.json"),
            Path("token_shards/lineage.json"),
            Path("probe_cache/manifest.json"),
            Path("probe_cache/lineage.json"),
        }
        if synbios_tail in allowed:
            if synbios_tail.parts[0] == "probe_cache":
                rest = Path(*synbios_tail.parts[1:])
                return results_root / "probes/synbios_moe/cache" / section / rest, None
            return pretraining_root / "datasets" / section / synbios_tail, None
        return None, "raw_or_derived_dataset_payload"
    if section == "runs":
        return pretraining_root / "runs" / synbios_tail, None
    if section == "operation_logs":
        return pretraining_root / "preparation_logs" / synbios_tail, None
    if section == "checkpoints":
        return _checkpoint_mapping(synbios_tail, pretraining_root / "checkpoints")
    if section != "results":
        return None, "unknown_synbios_section"

    if not synbios_tail.parts:
        return None, "directory"
    result_family = synbios_tail.parts[0]
    if result_family == "probe_batch_benchmark":
        rest = Path(*synbios_tail.parts[1:])
        return results_root / "benchmarks/synbios_moe/probe_batch_benchmark" / rest, None
    if result_family == "ground_truth_first_batch_benchmark":
        rest = Path(*synbios_tail.parts[1:])
        return (
            results_root / "benchmarks/synbios_moe/ground_truth_first_batch_benchmark" / rest,
            None,
        )
    if synbios_tail.suffix == ".pt":
        return None, "probe_or_recovery_weight"
    if synbios_tail.name in {"records.csv", "bad_cases.csv", "route_records.csv"}:
        return None, "large_per_example_diagnostic"
    if result_family == "single_cloze_eval":
        if synbios_tail.parts[1:2] != ("full_100k",):
            return None, "non_mainline_cloze_stage"
        rest = Path(*synbios_tail.parts[1:])
        return results_root / "cloze/synbios_moe/single" / rest, None
    if result_family == "multi5_permute_cloze_eval":
        if synbios_tail.parts[1:2] != ("full_500k",):
            return None, "non_mainline_cloze_stage"
        rest = Path(*synbios_tail.parts[1:])
        return results_root / "cloze/synbios_moe/multi5_permute" / rest, None
    if result_family == "single_fsdp_4gpu" and synbios_tail.parts[1:2] == (
        "probe_pipeline",
    ):
        if synbios_tail.parts[2:3] != ("formal",):
            return None, "non_mainline_probe_stage"
        rest = Path(*synbios_tail.parts[3:])
        return results_root / "probes/synbios_moe/single/formal" / rest, None
    if result_family == "multi5_permute_fsdp_4gpu" and synbios_tail.parts[1:2] == (
        "probe_pipeline",
    ):
        if synbios_tail.parts[2:3] != ("formal",):
            return None, "non_mainline_probe_stage"
        rest = Path(*synbios_tail.parts[3:])
        return results_root / "probes/synbios_moe/multi5_permute/formal" / rest, None
    if result_family == "formal_probe_comparison_20260724":
        rest = Path(*synbios_tail.parts[1:])
        return results_root / "probes/synbios_moe/comparisons/formal_20260724" / rest, None
    if result_family == "repository_audit_20260724":
        rest = Path(*synbios_tail.parts[1:])
        return results_root / "catalog/audits/repository_20260724" / rest, None
    return None, "unknown_synbios_result_family"


def audit_export(
    repo_root: Path, artifact_root: Path, results_root: Path
) -> dict[str, object]:
    expected: list[tuple[Path, Path]] = []
    excluded = Counter()
    excluded_bytes = Counter()

    for source in sorted(artifact_root.rglob("*")):
        if not source.is_file():
            continue
        target, reason = artifact_mapping(source.relative_to(artifact_root), results_root)
        if target is None:
            assert reason is not None
            excluded[reason] += 1
            excluded_bytes[reason] += source.stat().st_size
        else:
            expected.append((source, target))

    missing: list[str] = []
    mismatched: list[str] = []
    for source, target in expected:
        if not target.is_file():
            missing.append(target.relative_to(repo_root).as_posix())
            continue
        if source.stat().st_size != target.stat().st_size or sha256(source) != sha256(target):
            mismatched.append(target.relative_to(repo_root).as_posix())

    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "passed" if not missing and not mismatched else "failed",
        "artifact_root": artifact_root.as_posix(),
        "expected_exported_files": len(expected),
        "expected_exported_bytes": sum(source.stat().st_size for source, _ in expected),
        "verified_identical_files": len(expected) - len(missing) - len(mismatched),
        "missing": missing,
        "mismatched": mismatched,
        "excluded": [
            {
                "reason": reason,
                "files": excluded[reason],
                "size_bytes": excluded_bytes[reason],
            }
            for reason in sorted(excluded)
        ],
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/catalog/export_audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    payload = audit_export(
        repo_root,
        args.artifacts.resolve(),
        args.results.resolve(),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["status"] != "passed":
        raise SystemExit(
            f"export audit failed: {len(payload['missing'])} missing, "
            f"{len(payload['mismatched'])} mismatched"
        )


if __name__ == "__main__":
    main()
