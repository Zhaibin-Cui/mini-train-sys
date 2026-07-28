"""Validate and combine independent probe validation results."""

import csv
import json
from collections.abc import Iterable
from pathlib import Path

from experiments.synbios_moe.artifact_io import write_json_atomic
from experiments.synbios_moe.probes.spec import ProbeJob


def summarize_probe_results(
    named_directories: dict[str, Path],
    output_dir: str | Path,
    *,
    expected_jobs: Iterable[ProbeJob] | None = None,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not named_directories:
        raise ValueError("at least one probe result directory is required")
    expected_keys = {job.key for job in expected_jobs} if expected_jobs is not None else None
    rows: list[dict[str, object]] = []
    run_payloads: dict[str, dict[str, object]] = {}
    for run_name, directory in named_directories.items():
        task_payloads = {}
        profile_fingerprints = set()
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            required = {"kind", "attribute", "target", "validation_accuracy"}
            if not required.issubset(payload):
                continue
            key = f"{payload['kind']}_{payload['attribute']}_{payload['target']}"
            if key in task_payloads:
                raise ValueError(f"duplicate validation result for task {key!r}")
            expected_positions = 6 if payload["kind"] == "p" else 1
            if len(payload["validation_accuracy"]) != expected_positions:
                raise ValueError(
                    f"task {key!r} has {len(payload['validation_accuracy'])} positions; "
                    f"expected {expected_positions}"
                )
            if int(payload["classes"]) <= 0 or int(payload["examples"]) <= 0:
                raise ValueError(f"task {key!r} has empty classes or examples")
            task_payloads[key] = payload
            dataset_manifest = payload.get("dataset_manifest", {})
            profile_hash = dataset_manifest.get("files", {}).get("profiles.jsonl", {}).get("sha256")
            if profile_hash:
                profile_fingerprints.add(profile_hash)
            for position, accuracy in enumerate(payload["validation_accuracy"]):
                rows.append(
                    {
                        "run": run_name,
                        "task": key,
                        "kind": payload["kind"],
                        "attribute": payload["attribute"],
                        "target": payload["target"],
                        "position": position,
                        "accuracy": float(accuracy),
                        "classes": int(payload["classes"]),
                        "examples": int(payload["examples"]),
                    }
                )
        if len(profile_fingerprints) > 1:
            raise ValueError(f"validation files for run {run_name!r} use different profiles")
        task_keys = set(task_payloads)
        if not task_keys:
            raise ValueError(f"no probe validation results found for run {run_name!r}")
        if expected_keys is not None and task_keys != expected_keys:
            missing = sorted(expected_keys - task_keys)
            unexpected = sorted(task_keys - expected_keys)
            raise ValueError(
                f"incomplete probe results for run {run_name!r}; "
                f"missing={missing}, unexpected={unexpected}"
            )
        run_payloads[run_name] = {
            "directory": str(directory.resolve()),
            "tasks": task_payloads,
            "profiles_sha256": next(iter(profile_fingerprints), None),
        }
    comparable_fingerprints = {
        payload["profiles_sha256"]
        for payload in run_payloads.values()
        if payload["profiles_sha256"] is not None
    }
    if len(comparable_fingerprints) > 1:
        raise ValueError("probe runs cannot be compared: profiles.jsonl fingerprints differ")
    task_sets = {frozenset(payload["tasks"]) for payload in run_payloads.values()}
    if len(task_sets) > 1:
        raise ValueError("probe runs cannot be compared: task sets differ")
    position_sets = {
        run_name: {
            (str(row["task"]), int(row["position"])) for row in rows if row["run"] == run_name
        }
        for run_name in named_directories
    }
    if len({frozenset(value) for value in position_sets.values()}) > 1:
        raise ValueError("probe runs cannot be compared: observation positions differ")
    summary = {"runs": run_payloads, "rows": rows}
    fields = [
        "run",
        "task",
        "kind",
        "attribute",
        "target",
        "position",
        "accuracy",
        "classes",
        "examples",
    ]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    run_names = list(named_directories)
    comparison_path = output / "comparison.csv"
    if len(run_names) >= 2:
        baseline = run_names[0]
        by_key = {(row["run"], row["task"], row["position"]): row for row in rows}
        comparisons = []
        for candidate in run_names[1:]:
            for key, base_row in by_key.items():
                if key[0] != baseline:
                    continue
                candidate_row = by_key.get((candidate, key[1], key[2]))
                if candidate_row is None:
                    continue
                comparisons.append(
                    {
                        "baseline": baseline,
                        "candidate": candidate,
                        "task": key[1],
                        "position": key[2],
                        "baseline_accuracy": base_row["accuracy"],
                        "candidate_accuracy": candidate_row["accuracy"],
                        "delta": float(candidate_row["accuracy"]) - float(base_row["accuracy"]),
                    }
                )
        comparison_fields = [
            "baseline",
            "candidate",
            "task",
            "position",
            "baseline_accuracy",
            "candidate_accuracy",
            "delta",
        ]
        with comparison_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=comparison_fields)
            writer.writeheader()
            writer.writerows(comparisons)
        summary["comparisons"] = comparisons
    elif comparison_path.exists():
        comparison_path.unlink()
    write_json_atomic(output / "summary.json", summary)
    return summary
