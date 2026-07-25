"""Reporting for the predicted-first-token whole-probe pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence


KINDS = ("p", "q")
ATTRIBUTES = ("birth_city", "university", "major", "company", "company_city")
POSITION_LABELS = ("birth date", "birth city", "university", "major", "company", "company city")
PROTOCOL = "predicted_first_whole_pilot_v1"


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
        raise ValueError("cannot write an empty predicted-first-token summary")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_tasks(run_dir: Path) -> dict[tuple[str, str], dict[str, object]]:
    tasks: dict[tuple[str, str], dict[str, object]] = {}
    for kind in KINDS:
        for attribute in ATTRIBUTES:
            path = run_dir / f"{kind}_{attribute}.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing predicted-first-token result: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("protocol") != PROTOCOL:
                raise ValueError(f"unexpected protocol in {path}")
            if payload.get("kind") != kind or payload.get("attribute") != attribute:
                raise ValueError(f"task identity mismatch in {path}")
            tasks[(kind, attribute)] = payload

    identity_fields = (
        "data",
        "probe_cache",
        "probe_dir",
        "checkpoint",
        "probe_cache_manifest_sha256",
    )
    reference = next(iter(tasks.values()))
    for key, payload in tasks.items():
        for field in identity_fields:
            if payload.get(field) != reference.get(field):
                raise ValueError(f"{key} disagrees with the run identity in {field}")
    return tasks


def _rows(tasks: dict[tuple[str, str], dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for (kind, attribute), payload in tasks.items():
        first = payload["first_accuracy_validation_by_position"]
        whole = payload["whole_accuracy_validation_by_source_position"]
        if len(first) != len(whole) or len(first) != (6 if kind == "p" else 1):
            raise ValueError(f"invalid position metrics for {kind}/{attribute}")
        for position, (first_accuracy, whole_accuracy) in enumerate(zip(first, whole)):
            rows.append(
                {
                    "kind": kind,
                    "attribute": attribute,
                    "source_position": position,
                    "source_position_label": (
                        POSITION_LABELS[position] if kind == "p" else "name query"
                    ),
                    "first_accuracy": float(first_accuracy),
                    "whole_accuracy": float(whole_accuracy),
                    "recovery_gap": float(whole_accuracy) - float(first_accuracy),
                    "classes": int(payload["classes"]),
                    "steps": int(payload["steps"]),
                    "batch_size": int(payload["batch_size"]),
                }
            )
    return rows


def _plot(rows: Sequence[dict[str, object]], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    p_rows = [row for row in rows if row["kind"] == "p"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("first_accuracy", "whole_accuracy"),
        ("Frozen t1 classifier", "Whole classifier after predicted t1"),
    ):
        matrix = np.asarray(
            [
                [
                    next(
                        float(row[metric])
                        for row in p_rows
                        if row["attribute"] == attribute and row["source_position"] == position
                    )
                    for position in range(6)
                ]
                for attribute in ATTRIBUTES
            ]
        )
        image = axis.imshow(matrix * 100, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        axis.set_xticks(range(6))
        axis.set_xticklabels(POSITION_LABELS, rotation=28, ha="right")
        axis.set_yticks(range(5))
        axis.set_yticklabels([value.replace("_", " ") for value in ATTRIBUTES])
        axis.set_title(title)
        for row_index in range(5):
            for column in range(6):
                value = matrix[row_index, column] * 100
                axis.text(
                    column,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value >= 55 else "#1a202c",
                )
    figure.colorbar(image, ax=axes, label="Held-out accuracy (%)", fraction=0.026, pad=0.02)
    figure.suptitle("Predicted-t1 P-probe pilot · multi5+permute", fontsize=15)
    p_path = figures / "predicted_first_p_overview"
    figure.savefig(p_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    figure.savefig(p_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    q_rows = [row for row in rows if row["kind"] == "q"]
    x = np.arange(len(ATTRIBUTES))
    figure, axis = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    width = 0.36
    axis.bar(
        x - width / 2,
        [float(row["first_accuracy"]) * 100 for row in q_rows],
        width,
        label="Frozen t1 classifier",
        color="#718096",
    )
    axis.bar(
        x + width / 2,
        [float(row["whole_accuracy"]) * 100 for row in q_rows],
        width,
        label="Whole classifier after predicted t1",
        color="#2b6cb0",
    )
    axis.set_xticks(x)
    axis.set_xticklabels([value.replace("_", " ") for value in ATTRIBUTES], rotation=20)
    axis.set_ylabel("Held-out accuracy (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    axis.set_title("Predicted-t1 Q-probe pilot · multi5+permute")
    q_path = figures / "predicted_first_q_overview"
    figure.savefig(q_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    figure.savefig(q_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return [
        str(path.relative_to(output_dir))
        for stem in (p_path, q_path)
        for path in (stem.with_suffix(".png"), stem.with_suffix(".pdf"))
    ]


def summarize_predicted_first_whole(run_dir: str | Path) -> dict[str, object]:
    """Validate all ten pilot tasks and write standard tables and figures."""

    output = Path(run_dir)
    tasks = _load_tasks(output)
    rows = _rows(tasks)
    _atomic_csv(output / "summary.csv", rows)
    reference = next(iter(tasks.values()))
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "condition": "multi5_permute",
        "identity": {
            field: reference.get(field)
            for field in (
                "data",
                "probe_cache",
                "probe_dir",
                "checkpoint",
                "probe_cache_manifest_sha256",
            )
        },
        "tasks": len(tasks),
        "rows": rows,
        "figures": _plot(rows, output),
    }
    _atomic_json(output / "summary.json", summary)
    return summary
