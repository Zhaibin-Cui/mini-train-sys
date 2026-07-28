#!/usr/bin/env python3
"""Build deterministic indexes for the Git-safe server result snapshot."""


import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CATALOG_RELATIVE_PATHS = {
    Path("catalog/artifacts.json"),
    Path("catalog/retention.json"),
    Path("catalog/summary.md"),
    Path("catalog/tensorboard/index.csv"),
    Path("MANIFEST.sha256"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_utf8(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def _scope(relative: Path) -> tuple[str, str]:
    parts = relative.parts
    category = parts[0] if parts else "unknown"
    if len(parts) == 1:
        return "root", "index"
    if category == "benchmarks":
        name = parts[1] if len(parts) > 1 else "root"
        if name == "operator_benchmark":
            return category, "kernels"
        if name == "synbios_moe":
            return category, "probes"
        if name == "notebooks":
            return category, "executed_notebooks"
        if name == "logs":
            return category, "logs"
        return category, "distributed_training"
    if category in {"pretraining", "cloze", "probes"}:
        if len(parts) == 2 and parts[1] == "README.md":
            return category, "index"
        return category, parts[1] if len(parts) > 1 else "root"
    if category == "environment":
        return category, "inventory"
    if category == "catalog":
        return category, parts[1] if len(parts) > 1 else "index"
    return category, parts[1] if len(parts) > 1 else "root"


def log_destination(name: str) -> Path | None:
    """Return the destination for a mainline server log, if retained."""
    lowered = name.lower()
    if any(
        token in lowered
        for token in (
            "kernel",
            "backend_benchmark",
            "distributed_server_benchmark",
            "capacity",
            "b112_stability",
            "weak_scal",
            "cuda_build",
        )
    ):
        return Path("benchmarks/logs")
    if "cloze_full" in lowered:
        return Path("cloze/synbios_moe/logs")
    if any(
        token in lowered
        for token in (
            "prepare",
            "fsdp4_formal",
        )
    ):
        return Path("pretraining/synbios_moe/logs")
    return None


def _tensorboard_scope(relative: Path) -> tuple[str, str, str]:
    text = relative.as_posix()
    condition = "other"
    if "multi5_permute" in text:
        condition = "multi5_permute"
    elif "single" in text:
        condition = "single"

    if text.startswith("benchmarks/"):
        stage = "benchmark"
    elif text.startswith("probes/") and "/formal/" in text:
        stage = "probe_formal"
    elif text.startswith("probes/"):
        stage = "probe"
    elif text.startswith("pretraining/") and "/preparation_logs/" in text:
        stage = "preparation"
    elif text.startswith("pretraining/") and "/runs/" in text:
        stage = "pretraining"
    elif text.startswith("cloze/"):
        stage = "cloze"
    else:
        stage = "other"
    return "tensorboard", condition, stage


def _directory_stats(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    count = 0
    size = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            size += item.stat().st_size
    return count, size


def _selected_file_stats(path: Path, predicate) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    files = [item for item in path.rglob("*") if item.is_file() and predicate(item)]
    return len(files), sum(item.stat().st_size for item in files)


def _manifest_identity(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_retention_inventory(artifact_root: Path) -> dict[str, Any]:
    """Describe large server-only payloads without copying their contents."""
    groups: list[dict[str, Any]] = []
    candidates = [
        (
            "dataset_single",
            artifact_root / "synbios_moe/single",
            artifact_root / "synbios_moe/single/manifest.json",
            "Raw biographies, profiles, token shards, and derived probe cache stay on /data.",
        ),
        (
            "dataset_multi5_permute",
            artifact_root / "synbios_moe/multi5_permute",
            artifact_root / "synbios_moe/multi5_permute/manifest.json",
            "Raw biographies, profiles, token shards, and derived probe cache stay on /data.",
        ),
        (
            "formal_checkpoints_single",
            artifact_root
            / "synbios_moe/checkpoints/synbios_moe_single_fsdp_4gpu",
            None,
            "DCP/Adam shards and model.pt stay on /data; COMMITTED/runtime/RNG metadata is exported.",
        ),
        (
            "formal_checkpoints_multi5_permute",
            artifact_root
            / "synbios_moe/checkpoints/synbios_moe_multi5_permute_fsdp_4gpu",
            None,
            "DCP/Adam shards and model.pt stay on /data; COMMITTED/runtime/RNG metadata is exported.",
        ),
        (
            "probe_weights_and_records",
            artifact_root / "synbios_moe/results",
            None,
            "Probe .pt heads and large per-example diagnostic records stay on /data; aggregates are exported.",
        ),
    ]
    for name, path, manifest, policy in candidates:
        if name == "probe_weights_and_records":
            files, size = _selected_file_stats(
                path,
                lambda item: item.suffix == ".pt"
                or item.name in {"records.csv", "bad_cases.csv", "route_records.csv"},
            )
        else:
            files, size = _directory_stats(path)
        if not path.exists():
            continue
        entry: dict[str, Any] = {
            "name": name,
            "logical_path": path.as_posix(),
            "files": files,
            "size_bytes": size,
            "retention": "server_only",
            "policy": policy,
        }
        manifest_identity = _manifest_identity(manifest) if manifest else None
        if manifest_identity:
            entry["authoritative_manifest"] = manifest_identity
        groups.append(entry)
    return {
        "schema_version": 1,
        "artifact_root": artifact_root.as_posix(),
        "groups": groups,
    }


def retained_inventory(results_root: Path, artifact_root: Path) -> dict[str, Any]:
    """Refresh server retention when mounted, otherwise preserve the published inventory."""

    if artifact_root.is_dir():
        return build_retention_inventory(artifact_root)
    published = results_root / "catalog" / "retention.json"
    if published.is_file():
        payload = json.loads(published.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("groups"), list):
            raise ValueError(f"invalid published retention inventory: {published}")
        return payload
    return {
        "schema_version": 1,
        "artifact_root": artifact_root.as_posix(),
        "groups": [],
    }


def build_catalog(results_root: Path, artifact_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    tensorboard_rows: list[dict[str, Any]] = []

    for path in sorted(results_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(results_root)
        if relative in CATALOG_RELATIVE_PATHS:
            continue
        category, scope = _scope(relative)
        size = path.stat().st_size
        records.append(
            {
                "path": relative.as_posix(),
                "category": category,
                "scope": scope,
                "size_bytes": size,
            }
        )
        totals[(category, scope)][0] += 1
        totals[(category, scope)][1] += size
        if path.name.startswith("events.out.tfevents."):
            _, condition, stage = _tensorboard_scope(relative)
            tensorboard_rows.append(
                {
                    "path": relative.as_posix(),
                    "condition": condition,
                    "stage": stage,
                    "size_bytes": size,
                }
            )

    retention = retained_inventory(results_root, artifact_root)
    payload = {
        "schema_version": 1,
        "results_root": results_root.name,
        "files": records,
        "groups": [
            {
                "category": category,
                "scope": scope,
                "files": values[0],
                "size_bytes": values[1],
            }
            for (category, scope), values in sorted(totals.items())
        ],
        "tensorboard": {
            "events": len(tensorboard_rows),
            "size_bytes": sum(row["size_bytes"] for row in tensorboard_rows),
        },
    }

    catalog_dir = results_root / "catalog"
    tensorboard_dir = results_root / "catalog" / "tensorboard"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    _write_utf8(
        catalog_dir / "artifacts.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _write_utf8(
        catalog_dir / "retention.json",
        json.dumps(retention, indent=2, sort_keys=True) + "\n",
    )
    with (tensorboard_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "condition", "stage", "size_bytes"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(sorted(tensorboard_rows, key=lambda row: row["path"]))

    lines = [
        "# Result catalog",
        "",
        "This is the deterministic index of the Git-safe server snapshot. Paths are relative to",
        "`results/`; large server-only payloads are listed in `retention.json`.",
        "",
        "| Category | Scope | Files | Size |",
        "|---|---|---:|---:|",
    ]
    for (category, scope), values in sorted(totals.items()):
        lines.append(
            f"| `{category}` | `{scope}` | {values[0]:,} | {_human_bytes(values[1])} |"
        )
    lines.extend(
        [
            "",
            "## TensorBoard",
            "",
            f"- Event files: **{len(tensorboard_rows):,}**",
            f"- Total size: **{_human_bytes(sum(row['size_bytes'] for row in tensorboard_rows))}**",
            "- Machine index: [`tensorboard/index.csv`](tensorboard/index.csv)",
            "",
            "## Server-only retention",
            "",
            "| Group | Files | Size | Location |",
            "|---|---:|---:|---|",
        ]
    )
    for group in retention["groups"]:
        lines.append(
            f"| `{group['name']}` | {group['files']:,} | "
            f"{_human_bytes(group['size_bytes'])} | `{group['logical_path']}` |"
        )
    lines.append("")
    _write_utf8(catalog_dir / "summary.md", "\n".join(lines))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Mounted artifact root used only for server-only retention metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_catalog(args.results.resolve(), args.artifacts.resolve())


if __name__ == "__main__":
    main()
