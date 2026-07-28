"""Summarize the oracle-first-token whole-value intervention."""


import json
from pathlib import Path
from typing import Sequence

from experiments.synbios_moe.artifact_io import (
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)


KINDS = ("p",)
ATTRIBUTES = ("birth_city", "university", "major", "company", "company_city")
POSITION_LABELS = ("birth date", "birth city", "university", "major", "company", "company city")
PROTOCOL = "ground_truth_first_whole_rank_matched_v1"


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
            expected_rank = 2
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
        normalized = str(baseline_path).replace("\\", "/")
        marker = "/synbios_moe/"
        if marker in normalized:
            repository = Path(__file__).resolve().parents[3]
            suffix = normalized.split(marker, 1)[1]
            exported = repository / "results" / "formal_runs" / "synbios_moe" / suffix
            if exported.is_file():
                baseline_path = exported
    if not baseline_path.is_file():
        return {}, None
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    metrics = {
        (str(row["kind"]), str(row["attribute"]), int(row["position"])): float(row["accuracy"])
        for row in payload["rows"]
        if row.get("target") == "whole"
        and row.get("kind") == "p"
        and row.get("attribute") in ATTRIBUTES
    }
    expected = {
        (kind, attribute, position)
        for kind in KINDS
        for attribute in ATTRIBUTES
        for position in range(6)
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
        if len(first) != len(whole) or len(first) != 6:
            raise ValueError(f"invalid position metrics for {kind}/{attribute}")
        for position, (first_accuracy, whole_accuracy) in enumerate(zip(first, whole)):
            baseline_accuracy = baseline.get((kind, attribute, position))
            rows.append(
                {
                    "kind": kind,
                    "attribute": attribute,
                    "source_position": position,
                    "source_position_label": POSITION_LABELS[position],
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


def _condition(reference: dict[str, object]) -> str:
    data_root = Path(str(reference["data"]))
    manifest_path = data_root / "manifest.json"
    if manifest_path.is_file():
        variant = json.loads(manifest_path.read_text(encoding="utf-8")).get("variant")
        if variant in {"single", "multi5+permute"}:
            return str(variant)
    data_text = str(data_root)
    if "multi5_permute" in data_text or "multi5+permute" in data_text:
        return "multi5+permute"
    if "single" in data_text:
        return "single"
    raise ValueError(f"cannot identify dataset condition from {data_root}")


def _plot(rows: Sequence[dict[str, object]], output_dir: Path, condition: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    if not all(row["original_whole_accuracy"] is not None for row in rows):
        raise ValueError("ground-truth-first figure requires the formal no-t1 baseline")
    p_rows = [row for row in rows if row["kind"] == "p"]
    metrics = [
        ("original_whole_accuracy", "Formal no-t1 baseline"),
        ("whole_accuracy", "Fresh whole probe + P-first output"),
    ]
    cmap = LinearSegmentedColormap.from_list(
        "paper_teal",
        ["#f6f3ed", "#d9ece6", "#8bc7b8", "#2a968a", "#075c63"],
    )
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
        image = axis.imshow(
            matrix * 100,
            vmin=0,
            vmax=100,
            cmap=cmap,
            aspect="auto",
        )
        axis.set_xticks(range(6))
        axis.set_xticklabels(POSITION_LABELS, rotation=28, ha="right")
        axis.set_yticks(range(5))
        axis.set_yticklabels([value.replace("_", " ") for value in ATTRIBUTES])
        axis.tick_params(length=0, labelsize=12)
        marks_fixed_positions = condition == "single" and metric in {
            "original_whole_accuracy",
            "whole_accuracy",
        }
        axis.set_title(
            f"{title}\nfixed positions outlined" if marks_fixed_positions else title,
            loc="left",
            fontsize=16,
            fontweight="bold",
            pad=15,
        )
        if marks_fixed_positions:
            for row_index in range(len(ATTRIBUTES)):
                axis.add_patch(
                    Rectangle(
                        (row_index + 0.5, row_index - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#d29c62",
                        linewidth=4.5,
                    )
                )
        for row_index in range(5):
            for column in range(6):
                value = matrix[row_index, column] * 100
                axis.text(
                    column,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=("white" if value >= 55 else "#1a202c"),
                )
        colorbar = figure.colorbar(
            image,
            ax=axis,
            label="Held-out accuracy (%)",
            fraction=0.046,
            pad=0.02,
        )
        colorbar.ax.tick_params(labelsize=11)
        colorbar.set_label("Held-out accuracy (%)", fontsize=12)
    figure.suptitle(
        f"Whole-value P readout · {condition}",
        fontsize=21,
        fontweight="bold",
    )
    p_path = figures / "ground_truth_first_p_overview"
    figure.savefig(p_path.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(p_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    return [
        str(path.relative_to(output_dir))
        for stem in (p_path,)
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
    condition: str,
) -> None:
    p_rows = [row for row in rows if row["kind"] == "p"]
    p0_rows = [row for row in p_rows if row["source_position"] == 0]
    p_batch = next(row["batch_size"] for row in rows if row["kind"] == "p")
    steps = sorted({int(row["steps"]) for row in rows})
    step_text = ", ".join(str(value) for value in steps)
    text = f"""# Fresh whole probes with oracle `t1`

## Question or hypothesis

After the P-first classifier supplies the first attribute token, can a fresh rank-matched whole-value probe
read the remaining value more accurately from the frozen `{condition}` MoE than the original
formal whole probe without that token?

## Exact compared conditions

The baseline is the original formal whole probe. The intervention trains a fresh
P `AttributeProbe` with the same rank-2 low-rank input delta and reads the inserted ground-truth `t1`
after each biography prefix. It uses the same frozen backbone, whole-value class mappings,
person split, seed, and cache. This run uses {step_text} optimizer steps per head and P batch
{p_batch}.

`t1` comes from the aligned first-token label in the probe cache. The token ID is decoded and
round-trip checked before the frozen-backbone readout; neither the original P-first classifier
nor the fresh whole-value head generates it.

## Run/checkpoint and dataset identity

- Backbone checkpoint: `{reference["checkpoint"]}`
- Probe cache: `{reference["probe_cache"]}`
- Probe-cache manifest SHA256: `{reference["probe_cache_manifest_sha256"]}`
- Baseline summary: `{baseline_path if baseline_path is not None else "unavailable"}`
- Evaluation: complete person-held-out probe validation; these people were seen during backbone
  pretraining, so this is representation readout rather than unseen-person generalization.

## Primary metrics

| Endpoint | Formal no-`t1` baseline | Fresh whole probe + oracle `t1` | Delta |
|---|---:|---:|---:|
| P, all six source positions × five attributes | {_percent(_mean(p_rows, "original_whole_accuracy"))} | {_percent(_mean(p_rows, "whole_accuracy"))} | {_points(_mean(p_rows, "delta_vs_original_whole"))} |
| P0, five attributes | {_percent(_mean(p0_rows, "original_whole_accuracy"))} | {_percent(_mean(p0_rows, "whole_accuracy"))} | {_points(_mean(p0_rows, "delta_vs_original_whole"))} |

## Supporting artifacts

- Machine-readable aggregate: `summary.json`
- Position-level table: `summary.csv`
- P formal-baseline-vs-intervention heatmap: `figures/ground_truth_first_p_overview.png`
- Individual task JSON/PT, loss curves, recovery checkpoints, and operation logs are retained in
  this run directory.

## Interpretation

An increase shows that the complete value is more linearly extractable after oracle `t1` changes
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

Use these complete held-out curves and tables to decide whether the qualitative baseline-vs-oracle-`t1`
contrast is stable. If only P remains optimization-limited, extend P alone with the same
protocol and report the extension separately; do not silently replace this pilot-budget result.
"""
    write_text_atomic(output / "README.md", text)


def summarize_ground_truth_first_whole(run_dir: str | Path) -> dict[str, object]:
    """Validate all five P pilot tasks and write standard tables and figures."""

    output = Path(run_dir)
    tasks = _load_tasks(output)
    baseline, baseline_path = _baseline_metrics(tasks)
    rows = _rows(tasks, baseline)
    write_csv_atomic(output / "summary.csv", rows)
    reference = next(iter(tasks.values()))
    condition = _condition(reference)
    summary = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "condition": condition,
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
                "sha256": sha256_file(baseline_path),
                "protocol": "original_formal_whole_probe",
            }
            if baseline_path is not None
            else None
        ),
        "rows": rows,
        "figures": _plot(rows, output, condition),
    }
    write_json_atomic(output / "summary.json", summary)
    _write_report(
        output,
        rows=rows,
        reference=reference,
        baseline_path=baseline_path,
        condition=condition,
    )
    return summary
