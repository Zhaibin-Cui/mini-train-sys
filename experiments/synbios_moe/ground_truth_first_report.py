"""Reporting for the ground-truth-t1 rank-matched whole-probe run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence


KINDS = ("p", "q")
ATTRIBUTES = ("birth_city", "university", "major", "company", "company_city")
POSITION_LABELS = ("birth date", "birth city", "university", "major", "company", "company city")
PROTOCOL = "ground_truth_first_whole_rank_matched_v1"


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
        raise ValueError("cannot write an empty ground-truth-first-token summary")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_tasks(run_dir: Path) -> dict[tuple[str, str], dict[str, object]]:
    tasks: dict[tuple[str, str], dict[str, object]] = {}
    for kind in KINDS:
        for attribute in ATTRIBUTES:
            path = run_dir / f"{kind}_{attribute}.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing ground-truth-first-token result: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("protocol") != PROTOCOL:
                raise ValueError(f"unexpected protocol in {path}")
            if payload.get("kind") != kind or payload.get("attribute") != attribute:
                raise ValueError(f"task identity mismatch in {path}")
            expected_rank = 2 if kind == "p" else 16
            architecture = payload.get("architecture_match", {})
            if (
                payload.get("ground_truth_first_token") is not True
                or payload.get("rank") != expected_rank
                or architecture.get("rank") != expected_rank
                or architecture.get("reference_rank") != expected_rank
                or architecture.get("low_rank_embedding_delta") is not True
            ):
                raise ValueError(f"ground-truth/rank architecture gate failed in {path}")
            if not all(payload.get("alignment_checks", {}).values()):
                raise ValueError(f"dataset alignment gate failed in {path}")
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


def _baseline_metrics(
    tasks: dict[tuple[str, str], dict[str, object]],
) -> tuple[dict[tuple[str, str, int], float], Path | None]:
    reference = next(iter(tasks.values()))
    baseline_path = Path(str(reference["probe_dir"])).parent / "summary" / "summary.json"
    if not baseline_path.is_file():
        return {}, None
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    metrics = {
        (str(row["kind"]), str(row["attribute"]), int(row["position"])): float(row["accuracy"])
        for row in payload["rows"]
        if row.get("target") == "whole"
        and row.get("kind") in KINDS
        and row.get("attribute") in ATTRIBUTES
    }
    expected = {
        (kind, attribute, position)
        for kind in KINDS
        for attribute in ATTRIBUTES
        for position in range(6 if kind == "p" else 1)
    }
    missing = sorted(expected - metrics.keys())
    if missing:
        raise ValueError(f"original whole-probe summary is incomplete: {missing}")
    return metrics, baseline_path


def _rows(
    tasks: dict[tuple[str, str], dict[str, object]],
    baseline: dict[tuple[str, str, int], float],
) -> list[dict[str, object]]:
    rows = []
    for (kind, attribute), payload in tasks.items():
        first = payload["ground_truth_first_accuracy_by_position"]
        whole = payload["whole_accuracy_validation_by_source_position"]
        if len(first) != len(whole) or len(first) != (6 if kind == "p" else 1):
            raise ValueError(f"invalid position metrics for {kind}/{attribute}")
        for position, (first_accuracy, whole_accuracy) in enumerate(zip(first, whole)):
            baseline_accuracy = baseline.get((kind, attribute, position))
            rows.append(
                {
                    "kind": kind,
                    "attribute": attribute,
                    "source_position": position,
                    "source_position_label": (
                        POSITION_LABELS[position] if kind == "p" else "name query"
                    ),
                    "first_accuracy": float(first_accuracy),
                    "original_whole_accuracy": baseline_accuracy,
                    "whole_accuracy": float(whole_accuracy),
                    "delta_vs_original_whole": (
                        float(whole_accuracy) - baseline_accuracy
                        if baseline_accuracy is not None
                        else None
                    ),
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

    has_baseline = all(row["original_whole_accuracy"] is not None for row in rows)
    p_rows = [row for row in rows if row["kind"] == "p"]
    metrics = [
        ("first_accuracy", "Ground-truth t1 supplied"),
        *(([("original_whole_accuracy", "Original whole probe")]) if has_baseline else []),
        ("whole_accuracy", "Rank-matched whole probe after true t1"),
        *(([("delta_vs_original_whole", "True t1 − original")]) if has_baseline else []),
    ]
    figure, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(6.6 * len(metrics), 5.4),
        constrained_layout=True,
    )
    for axis, (metric, title) in zip(
        axes,
        metrics,
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
        is_delta = metric == "delta_vs_original_whole"
        image = axis.imshow(
            matrix * 100,
            vmin=-50 if is_delta else 0,
            vmax=50 if is_delta else 100,
            cmap="RdBu" if is_delta else "Blues",
            aspect="auto",
        )
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
                    color=(
                        "white"
                        if (not is_delta and value >= 55) or (is_delta and abs(value) >= 28)
                        else "#1a202c"
                    ),
                )
        figure.colorbar(
            image,
            ax=axis,
            label="Delta (percentage points)" if is_delta else "Held-out accuracy (%)",
            fraction=0.046,
            pad=0.02,
        )
    figure.suptitle("Ground-truth-t1 rank-matched P probe · multi5+permute", fontsize=15)
    p_path = figures / "ground_truth_first_p_overview"
    figure.savefig(p_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    figure.savefig(p_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    q_rows = [row for row in rows if row["kind"] == "q"]
    x = np.arange(len(ATTRIBUTES))
    figure, axis = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    width = 0.25 if has_baseline else 0.36
    axis.bar(
        x - width if has_baseline else x - width / 2,
        [float(row["first_accuracy"]) * 100 for row in q_rows],
        width,
        label="Frozen t1 classifier",
        color="#718096",
    )
    if has_baseline:
        axis.bar(
            x,
            [float(row["original_whole_accuracy"]) * 100 for row in q_rows],
            width,
            label="Original whole probe",
            color="#38a169",
        )
    axis.bar(
        x + width if has_baseline else x + width / 2,
        [float(row["whole_accuracy"]) * 100 for row in q_rows],
        width,
        label="Rank-matched whole probe after true t1",
        color="#2b6cb0",
    )
    axis.set_xticks(x)
    axis.set_xticklabels([value.replace("_", " ") for value in ATTRIBUTES], rotation=20)
    axis.set_ylabel("Held-out accuracy (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    axis.set_title("Ground-truth-t1 rank-matched Q probe · multi5+permute")
    q_path = figures / "ground_truth_first_q_overview"
    figure.savefig(q_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    figure.savefig(q_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return [
        str(path.relative_to(output_dir))
        for stem in (p_path, q_path)
        for path in (stem.with_suffix(".png"), stem.with_suffix(".pdf"))
    ]


def _mean(rows: Sequence[dict[str, object]], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return sum(values) / len(values) if values else None


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def _points(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:+.2f} pp"


def _write_report(
    output: Path,
    *,
    rows: Sequence[dict[str, object]],
    reference: dict[str, object],
    baseline_path: Path | None,
) -> None:
    p_rows = [row for row in rows if row["kind"] == "p"]
    p0_rows = [row for row in p_rows if row["source_position"] == 0]
    q_rows = [row for row in rows if row["kind"] == "q"]
    p_batch = next(row["batch_size"] for row in rows if row["kind"] == "p")
    q_batch = next(row["batch_size"] for row in rows if row["kind"] == "q")
    steps = sorted({int(row["steps"]) for row in rows})
    q_lines = []
    for attribute in ATTRIBUTES:
        row = next(
            value
            for value in q_rows
            if value["attribute"] == attribute
        )
        q_lines.append(
            f"| `{attribute}` | {_percent(row['original_whole_accuracy'])} | "
            f"{_percent(row['whole_accuracy'])} | "
            f"{_points(row['delta_vs_original_whole'])} |"
        )
    step_text = ", ".join(str(value) for value in steps)
    text = f"""# Ground-truth-`t1` rank-matched fresh whole probes

## Question or hypothesis

After the correct first attribute token is supplied, can a fresh rank-matched whole-value probe
read the remaining value more accurately from the frozen `multi5_permute` MoE than the original
formal whole probe without that token?

## Exact compared conditions

The baseline is the original formal whole probe. The intervention trains a fresh
`AttributeProbe` with the same low-rank input delta and rank: P rank 2 reads the inserted true
`t1` after each biography prefix; Q rank 16 reads the final EOS in
`[EOS, name, true t1, EOS]`. Both use the same frozen backbone, whole-value class mappings,
person split, seed, and cache. This run uses {step_text} optimizer steps per head, P batch {p_batch},
and Q batch {q_batch}.

## Run/checkpoint and dataset identity

- Backbone checkpoint: `{reference['checkpoint']}`
- Probe cache: `{reference['probe_cache']}`
- Probe-cache manifest SHA256: `{reference['probe_cache_manifest_sha256']}`
- Baseline summary: `{baseline_path if baseline_path is not None else 'unavailable'}`
- Evaluation: complete person-held-out probe validation; these people were seen during backbone
  pretraining, so this is representation readout rather than unseen-person generalization.

## Primary metrics

| Endpoint | Original whole probe | Fresh whole probe + true `t1` | Delta |
|---|---:|---:|---:|
| P, all six source positions × five attributes | {_percent(_mean(p_rows, 'original_whole_accuracy'))} | {_percent(_mean(p_rows, 'whole_accuracy'))} | {_points(_mean(p_rows, 'delta_vs_original_whole'))} |
| P0, five attributes | {_percent(_mean(p0_rows, 'original_whole_accuracy'))} | {_percent(_mean(p0_rows, 'whole_accuracy'))} | {_points(_mean(p0_rows, 'delta_vs_original_whole'))} |
| Q, five attributes | {_percent(_mean(q_rows, 'original_whole_accuracy'))} | {_percent(_mean(q_rows, 'whole_accuracy'))} | {_points(_mean(q_rows, 'delta_vs_original_whole'))} |

Q attribute detail:

| Attribute | Original Q whole | Q whole + true `t1` | Delta |
|---|---:|---:|---:|
{chr(10).join(q_lines)}

## Supporting artifacts

- Machine-readable aggregate: `summary.json`
- Position-level table: `summary.csv`
- P original-vs-intervention heatmap: `figures/ground_truth_first_p_overview.png`
- Q original-vs-intervention chart: `figures/ground_truth_first_q_overview.png`
- Individual task JSON/PT, loss curves, recovery checkpoints, and operation logs are retained in
  this run directory; lifecycle and failed/stopped predecessors are in repository-root
  `HISTORY.md`.

## Interpretation

An increase shows that the complete value is more linearly extractable after true `t1` changes
the context and a matched fresh head is trained. It does not show that the unchanged original
head was causally unlocked, and it does not by itself locate the value in a particular MoE
expert.

## Limitations and threats to validity

This is a user-selected 4,000-step pilot-budget full matrix, one third of the prior 12,000-step
whole-probe updates. P expands each biography into six separate sequences, so its label exposure
is also lower than an original P probe that reads six positions in one forward. Any visibly
unconverged P task is a budget limitation, not evidence that true `t1` contains no useful
information. The intervention also changes sequence length and readout coordinates.

## Next decision/action

Use these complete held-out curves and tables to decide whether the qualitative original-vs-true
`t1` contrast is stable. If only P remains optimization-limited, extend P alone with the same
protocol and report the extension separately; do not silently replace this pilot-budget result.
"""
    _atomic_text(output / "README.md", text)


def summarize_ground_truth_first_whole(run_dir: str | Path) -> dict[str, object]:
    """Validate all ten pilot tasks and write standard tables and figures."""

    output = Path(run_dir)
    tasks = _load_tasks(output)
    baseline, baseline_path = _baseline_metrics(tasks)
    rows = _rows(tasks, baseline)
    _atomic_csv(output / "summary.csv", rows)
    reference = next(iter(tasks.values()))
    summary = {
        "schema_version": 2,
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
        "baseline": (
            {
                "summary": str(baseline_path.resolve()),
                "sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                "protocol": "original_formal_whole_probe",
            }
            if baseline_path is not None
            else None
        ),
        "rows": rows,
        "figures": _plot(rows, output),
    }
    _atomic_json(output / "summary.json", summary)
    _write_report(
        output,
        rows=rows,
        reference=reference,
        baseline_path=baseline_path,
    )
    return summary
