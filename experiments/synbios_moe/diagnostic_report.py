"""Audited report artifacts for the two Q-whole MoE diagnostics.

This module is intentionally inference-free.  It validates the retained
oracle-intervention and route-branching outputs against the completed formal
pipeline, reconstructs their aggregate metrics, and renders deterministic
publication-quality tables and figures.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.synbios_moe.formal_report import (
    ALLEN_ZHU_Q_REFERENCE,
    ATTRIBUTE_LABELS,
    WHOLE_ATTRIBUTES,
    FormalRun,
    load_formal_run,
    tidy_rows,
    validate_matched_runs,
)


ORACLE_PROTOCOL = "q_whole_oracle_first_token_v1"
ROUTE_PROTOCOL = "q_bad_case_t1_t2_route_branching_v1"
EXPECTED_SPLIT = "person-held-out validation"


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _same_path(left: object, right: object) -> bool:
    return Path(str(left)).resolve() == Path(str(right)).resolve()


def _assert_close(left: float, right: float, label: str, tolerance: float = 1e-12) -> None:
    if abs(left - right) > tolerance:
        raise ValueError(f"{label} mismatch: {left} != {right}")


def _validate_raw_manifest(diagnostics_root: Path, manifest: dict[str, object]) -> None:
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported diagnostics raw-artifact manifest")
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise ValueError("diagnostics raw-artifact manifest has no artifacts")
    expected = {
        "oracle_first_token/records.csv",
        "bad_case_routes/bad_cases.csv",
        "bad_case_routes/route_records.csv",
    }
    if {str(record.get("logical_path")) for record in records} != expected:
        raise ValueError("diagnostics raw-artifact manifest is incomplete")
    for record in records:
        path = diagnostics_root / str(record["logical_path"])
        if not path.is_file():
            raise ValueError(f"missing retained raw diagnostic evidence: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"raw diagnostic size mismatch: {path}")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"raw diagnostic SHA256 mismatch: {path}")


def _validate_common_identity(
    formal: FormalRun,
    oracle: dict[str, object],
    routes: dict[str, object],
) -> None:
    identity = formal.identity
    if oracle.get("protocol") != ORACLE_PROTOCOL:
        raise ValueError("unexpected oracle diagnostic protocol")
    if routes.get("protocol") != ROUTE_PROTOCOL:
        raise ValueError("unexpected route diagnostic protocol")
    for label, payload in (("oracle", oracle), ("routes", routes)):
        if payload.get("parameters_updated") is not False:
            raise ValueError(f"{label} diagnostic was not inference-only")
        if payload.get("split") != EXPECTED_SPLIT:
            raise ValueError(f"{label} diagnostic uses the wrong split")
        for field, expected in (
            ("data", identity["data"]),
            ("probe_cache", identity["probe_cache"]),
            ("backbone_checkpoint", identity["checkpoint"]),
        ):
            if not _same_path(payload.get(field), expected):
                raise ValueError(f"{label} diagnostic identity mismatch in {field}")
        if not _same_path(payload.get("probe_dir"), formal.root / "training"):
            raise ValueError(f"{label} diagnostic uses the wrong formal probe directory")
    for field in ("data", "probe_cache", "probe_dir", "backbone_checkpoint"):
        if not _same_path(oracle[field], routes[field]):
            raise ValueError(f"diagnostic runs disagree in {field}")


def _oracle_rows(
    formal: FormalRun,
    summary: dict[str, object],
    csv_rows: Sequence[dict[str, str]],
) -> list[dict[str, object]]:
    attributes = summary.get("attributes")
    if not isinstance(attributes, list):
        raise ValueError("oracle summary has no attribute rows")
    by_attribute = {str(row["attribute"]): row for row in attributes}
    csv_by_attribute = {row["attribute"]: row for row in csv_rows}
    if set(by_attribute) != set(WHOLE_ATTRIBUTES) or set(csv_by_attribute) != set(
        WHOLE_ATTRIBUTES
    ):
        raise ValueError("oracle result does not cover the exact five whole attributes")

    rows: list[dict[str, object]] = []
    totals = defaultdict(int)
    for attribute in WHOLE_ATTRIBUTES:
        row = by_attribute[attribute]
        csv_row = csv_by_attribute[attribute]
        examples = int(row["examples"])
        baseline_correct = int(row["baseline_correct"])
        baseline_errors = int(row["baseline_errors"])
        recovered = int(row["recovered_errors"])
        harmed = int(row["harmed_correct"])
        if examples != baseline_correct + baseline_errors:
            raise ValueError(f"oracle counts do not close for {attribute}")
        expected_examples = int(formal.validation[f"q_{attribute}_whole"]["examples"])
        if examples != expected_examples:
            raise ValueError(f"oracle sample count differs from formal validation for {attribute}")
        formal_accuracy = float(
            formal.validation[f"q_{attribute}_whole"]["validation_accuracy"][0]
        )
        before = baseline_correct / examples
        after = (baseline_correct - harmed + recovered) / examples
        recovery = recovered / baseline_errors
        harm = harmed / baseline_correct
        _assert_close(
            before,
            formal_accuracy,
            f"{attribute} formal/oracle baseline",
            tolerance=1e-7,
        )
        for field, value in (
            ("accuracy_before", before),
            ("accuracy_after", after),
            ("accuracy_delta", after - before),
            ("recovery_rate", recovery),
            ("harm_rate", harm),
        ):
            _assert_close(float(row[field]), value, f"{attribute}.{field}")
            _assert_close(float(csv_row[field]), value, f"{attribute}.csv.{field}")
        clean = {
            "attribute": attribute,
            "examples": examples,
            "accuracy_before": before,
            "accuracy_after": after,
            "accuracy_delta": after - before,
            "baseline_errors": baseline_errors,
            "recovered_errors": recovered,
            "recovery_rate": recovery,
            "baseline_correct": baseline_correct,
            "harmed_correct": harmed,
            "harm_rate": harm,
        }
        rows.append(clean)
        for field in (
            "examples",
            "baseline_errors",
            "recovered_errors",
            "baseline_correct",
            "harmed_correct",
        ):
            totals[field] += int(clean[field])

    overall = summary.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("oracle summary has no overall metrics")
    before = totals["baseline_correct"] / totals["examples"]
    after = (
        totals["baseline_correct"] - totals["harmed_correct"] + totals["recovered_errors"]
    ) / totals["examples"]
    aggregate = {
        "attribute": "micro_overall",
        **totals,
        "accuracy_before": before,
        "accuracy_after": after,
        "accuracy_delta": after - before,
        "recovery_rate": totals["recovered_errors"] / totals["baseline_errors"],
        "harm_rate": totals["harmed_correct"] / totals["baseline_correct"],
    }
    for field in (
        "examples",
        "baseline_errors",
        "recovered_errors",
        "baseline_correct",
        "harmed_correct",
    ):
        if int(overall[field]) != int(aggregate[field]):
            raise ValueError(f"oracle overall count mismatch in {field}")
    for field in (
        "accuracy_before",
        "accuracy_after",
        "accuracy_delta",
        "recovery_rate",
        "harm_rate",
    ):
        _assert_close(float(overall[field]), float(aggregate[field]), f"oracle overall.{field}")
    return [*rows, aggregate]


def _route_rows(
    summary: dict[str, object],
    pair_rows: Sequence[dict[str, str]],
    nmi_rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if summary.get("case_definition") != (
        "Q-first correct, Q-whole wrong, whole value has >=2 tokens"
    ):
        raise ValueError("unexpected route bad-case definition")
    if (
        int(summary.get("layers", -1)) != 12
        or int(summary.get("experts", -1)) != 8
        or int(summary.get("top_k", -1)) != 2
    ):
        raise ValueError("unexpected MoE topology in route diagnostic")
    counts = summary.get("attributes")
    if not isinstance(counts, dict) or set(counts) != set(WHOLE_ATTRIBUTES):
        raise ValueError("route summary does not cover the exact five whole attributes")
    if sum(int(value) for value in counts.values()) != int(summary["examples"]):
        raise ValueError("route bad-case counts do not close")

    expected_pair_keys = {
        (attribute, layer, group)
        for attribute in WHOLE_ATTRIBUTES
        for layer in range(12)
        for group in ("same_t2", "different_t2")
    }
    pair_by_key = {
        (row["attribute"], int(row["layer"]), row["pair_group"]): row for row in pair_rows
    }
    if set(pair_by_key) != expected_pair_keys or len(pair_rows) != len(expected_pair_keys):
        raise ValueError("route pairwise table is not the complete 5x12x2 matrix")
    for row in pair_rows:
        if int(row["pair_count"]) <= 0:
            raise ValueError("route pairwise table contains an empty group")
        for field in ("t1_route_overlap", "t2_route_overlap"):
            if not 0.0 <= float(row[field]) <= 1.0:
                raise ValueError(f"route overlap outside [0,1]: {row}")
        _assert_close(
            float(row["branching_score"]),
            float(row["t1_route_overlap"]) - float(row["t2_route_overlap"]),
            "route branching score",
        )

    expected_nmi_keys = {
        (attribute, layer) for attribute in WHOLE_ATTRIBUTES for layer in range(12)
    }
    nmi_by_key = {(row["attribute"], int(row["layer"])): row for row in nmi_rows}
    if set(nmi_by_key) != expected_nmi_keys or len(nmi_rows) != len(expected_nmi_keys):
        raise ValueError("route NMI table is not the complete 5x12 matrix")
    for row in nmi_rows:
        for field in ("t1_top1_token_nmi", "t2_top1_token_nmi"):
            if not 0.0 <= float(row[field]) <= 1.0:
                raise ValueError(f"route NMI outside [0,1]: {row}")

    layer_rows: list[dict[str, object]] = []
    attribute_layer_rows: list[dict[str, object]] = []
    for attribute in WHOLE_ATTRIBUTES:
        for layer in range(12):
            same = pair_by_key[(attribute, layer, "same_t2")]
            different = pair_by_key[(attribute, layer, "different_t2")]
            attribute_layer_rows.append(
                {
                    "attribute": attribute,
                    "layer": layer,
                    "same_t2_branching_score": float(same["branching_score"]),
                    "different_t2_branching_score": float(different["branching_score"]),
                    "difference_in_differences": float(different["branching_score"])
                    - float(same["branching_score"]),
                    "same_t2_pair_count": int(same["pair_count"]),
                    "different_t2_pair_count": int(different["pair_count"]),
                    "t1_top1_token_nmi": float(
                        nmi_by_key[(attribute, layer)]["t1_top1_token_nmi"]
                    ),
                    "t2_top1_token_nmi": float(
                        nmi_by_key[(attribute, layer)]["t2_top1_token_nmi"]
                    ),
                }
            )
    for layer in range(12):
        selected = [row for row in pair_rows if int(row["layer"]) == layer]
        aggregate: dict[str, object] = {"layer": layer}
        for group in ("same_t2", "different_t2"):
            grouped = [row for row in selected if row["pair_group"] == group]
            total = sum(int(row["pair_count"]) for row in grouped)
            aggregate[f"{group}_pair_count"] = total
            for field in ("t1_route_overlap", "t2_route_overlap", "branching_score"):
                aggregate[f"{group}_{field}"] = (
                    sum(float(row[field]) * int(row["pair_count"]) for row in grouped) / total
                )
        aggregate["difference_in_differences"] = float(
            aggregate["different_t2_branching_score"]
        ) - float(aggregate["same_t2_branching_score"])
        layer_rows.append(aggregate)

    grouped_headline: dict[str, dict[str, float | int]] = {}
    for group in ("same_t2", "different_t2"):
        selected = [row for row in pair_rows if row["pair_group"] == group]
        total = sum(int(row["pair_count"]) for row in selected)
        grouped_headline[group] = {
            "pair_count": total,
            **{
                field: sum(float(row[field]) * int(row["pair_count"]) for row in selected)
                / total
                for field in ("t1_route_overlap", "t2_route_overlap", "branching_score")
            },
        }
    headline = {
        "bad_case_examples": int(summary["examples"]),
        "groups": grouped_headline,
        "difference_in_differences": (
            float(grouped_headline["different_t2"]["branching_score"])
            - float(grouped_headline["same_t2"]["branching_score"])
        ),
        "all_layers_positive": all(
            float(row["difference_in_differences"]) > 0 for row in layer_rows
        ),
        "max_t1_nmi_by_attribute": {
            attribute: max(
                float(row["t1_top1_token_nmi"])
                for row in nmi_rows
                if row["attribute"] == attribute
            )
            for attribute in WHOLE_ATTRIBUTES
        },
        "max_t2_nmi_by_attribute": {
            attribute: max(
                float(row["t2_top1_token_nmi"])
                for row in nmi_rows
                if row["attribute"] == attribute
            )
            for attribute in WHOLE_ATTRIBUTES
        },
    }
    return layer_rows, attribute_layer_rows, headline


def load_diagnostic_evidence(
    single_formal_root: str | Path,
    multi_formal_root: str | Path,
    diagnostics_root: str | Path,
) -> dict[str, object]:
    """Load and strictly cross-check the two completed diagnostic runs."""

    single = load_formal_run("single", single_formal_root)
    formal = load_formal_run("multi5_permute", multi_formal_root)
    validate_matched_runs(single, formal)
    diagnostics = Path(diagnostics_root).resolve()
    oracle_root = diagnostics / "oracle_first_token"
    route_root = diagnostics / "bad_case_routes"
    oracle = _read_json(oracle_root / "summary.json")
    routes = _read_json(route_root / "summary.json")
    _validate_common_identity(formal, oracle, routes)
    raw_manifest = _read_json(diagnostics / "raw_artifacts_manifest.json")
    _validate_raw_manifest(diagnostics, raw_manifest)
    oracle_rows = _oracle_rows(
        formal,
        oracle,
        _read_csv(oracle_root / "summary.csv"),
    )
    layer_rows, attribute_layer_rows, route_headline = _route_rows(
        routes,
        _read_csv(route_root / "pairwise_branching.csv"),
        _read_csv(route_root / "token_route_nmi.csv"),
    )
    return {
        "single_formal": single,
        "formal": formal,
        "diagnostics_root": diagnostics,
        "oracle": oracle,
        "routes": routes,
        "oracle_rows": oracle_rows,
        "route_layer_rows": layer_rows,
        "route_attribute_layer_rows": attribute_layer_rows,
        "route_headline": route_headline,
        "raw_manifest": raw_manifest,
    }


def _formal_whole_rows(single: FormalRun, multi: FormalRun) -> list[dict[str, object]]:
    return [
        row
        for row in tidy_rows((single, multi))
        if row["target"] == "whole"
    ]


def _save_figure(figure, destination: Path) -> None:
    figure.savefig(destination.with_suffix(".png"), dpi=200, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")


def _text_color(face_color: Sequence[float]) -> str:
    red, green, blue = face_color[:3]
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.52 else "#102a2e"


def _plot_oracle_table(rows: Sequence[dict[str, object]], destination: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors

    display = list(rows)
    labels = [
        ATTRIBUTE_LABELS.get(str(row["attribute"]), "MICRO OVERALL").upper() for row in display
    ]
    fields = (
        "accuracy_before",
        "accuracy_after",
        "accuracy_delta",
        "recovery_rate",
        "harm_rate",
    )
    headers = (
        "NAME ONLY\nACCURACY",
        "+ TRUE t1\nACCURACY",
        "CHANGE\n(pp)",
        "ERRORS\nRECOVERED",
        "CORRECT\nHARMED",
    )
    values = np.array([[100 * float(row[field]) for field in fields] for row in display])
    figure, axis = plt.subplots(figsize=(12.6, 5.9))
    axis.axis("off")
    table = axis.table(
        cellText=[
            [
                f"{value:.2f}" if field == "accuracy_delta" else f"{value:.2f}%"
                for value, field in zip(row, fields)
            ]
            for row in values
        ],
        rowLabels=labels,
        colLabels=headers,
        cellLoc="center",
        rowLoc="center",
        bbox=[0.12, 0.08, 0.85, 0.75],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    maps = (
        (plt.get_cmap("GnBu"), colors.Normalize(0, 60)),
        (plt.get_cmap("GnBu"), colors.Normalize(0, 60)),
        (
            plt.get_cmap("RdBu"),
            colors.TwoSlopeNorm(vmin=-11, vcenter=0, vmax=11),
        ),
        (plt.get_cmap("YlGn"), colors.Normalize(0, 35)),
        (plt.get_cmap("OrRd"), colors.Normalize(0, 35)),
    )
    for row_index, row in enumerate(values, start=1):
        for column_index, value in enumerate(row):
            cmap, normalization = maps[column_index]
            face = cmap(normalization(value))
            cell = table[(row_index, column_index)]
            cell.set_facecolor(face)
            cell.get_text().set_color(_text_color(face))
            cell.set_edgecolor("white")
            if row_index == len(values):
                cell.get_text().set_weight("bold")
        label_cell = table[(row_index, -1)]
        label_cell.set_facecolor("#e9f3f3" if row_index < len(values) else "#c9e2e3")
        label_cell.set_edgecolor("white")
        label_cell.get_text().set_weight("bold" if row_index == len(values) else "normal")
    for column_index in range(len(headers)):
        cell = table[(0, column_index)]
        cell.set_facecolor("#123b43")
        cell.set_edgecolor("white")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
    figure.text(
        0.5,
        0.94,
        "ORACLE TRUE-FIRST-TOKEN INTERVENTION",
        ha="center",
        fontsize=19,
        fontweight="bold",
        color="#123b43",
    )
    figure.text(
        0.5,
        0.885,
        "Same formal Q-whole head · person-held-out validation · no parameter updates",
        ha="center",
        fontsize=11,
        color="#4b6064",
    )
    figure.text(
        0.12,
        0.025,
        "Reading: positive change helps the unchanged head; harmed measures baseline-correct "
        "predictions that become wrong.",
        fontsize=9.5,
        color="#52666a",
    )
    _save_figure(figure, destination)
    plt.close(figure)


def _plot_route_table(headline: dict[str, object], destination: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors

    groups = headline["groups"]
    same = groups["same_t2"]
    different = groups["different_t2"]
    values = np.array(
        [
            [
                float(same["t1_route_overlap"]),
                float(same["t2_route_overlap"]),
                float(same["branching_score"]),
            ],
            [
                float(different["t1_route_overlap"]),
                float(different["t2_route_overlap"]),
                float(different["branching_score"]),
            ],
            [np.nan, np.nan, float(headline["difference_in_differences"])],
        ]
    )
    labels = ("SAME t2 CONTROL", "DIFFERENT t2 BRANCH", "DIFFERENCE-IN-DIFFERENCES")
    headers = ("t1 ROUTE\nOVERLAP", "t2 ROUTE\nOVERLAP", "BRANCHING\nSCORE")
    figure, axis = plt.subplots(figsize=(10.8, 4.7))
    axis.axis("off")
    texts = [
        ["" if np.isnan(value) else f"{value:+.3f}" if column == 2 else f"{value:.3f}"
         for column, value in enumerate(row)]
        for row in values
    ]
    table = axis.table(
        cellText=texts,
        rowLabels=labels,
        colLabels=headers,
        cellLoc="center",
        rowLoc="center",
        bbox=[0.22, 0.12, 0.72, 0.66],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    overlap_map = plt.get_cmap("GnBu")
    branch_map = plt.get_cmap("RdBu_r")
    overlap_norm = colors.Normalize(0, 1)
    branch_norm = colors.TwoSlopeNorm(vmin=-0.25, vcenter=0, vmax=0.25)
    for row_index, row in enumerate(values, start=1):
        for column_index, value in enumerate(row):
            cell = table[(row_index, column_index)]
            cell.set_edgecolor("white")
            if np.isnan(value):
                cell.set_facecolor("#f2f4f4")
                continue
            face = (
                overlap_map(overlap_norm(value))
                if column_index < 2
                else branch_map(branch_norm(value))
            )
            cell.set_facecolor(face)
            cell.get_text().set_color(_text_color(face))
            if row_index == len(values):
                cell.get_text().set_weight("bold")
        label_cell = table[(row_index, -1)]
        label_cell.set_facecolor("#e9f3f3" if row_index < len(values) else "#c9e2e3")
        label_cell.set_edgecolor("white")
        label_cell.get_text().set_weight("bold" if row_index == len(values) else "normal")
    for column_index in range(len(headers)):
        cell = table[(0, column_index)]
        cell.set_facecolor("#123b43")
        cell.set_edgecolor("white")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
    figure.text(
        0.5,
        0.93,
        "BAD-CASE MoE ROUTE BRANCHING",
        ha="center",
        fontsize=19,
        fontweight="bold",
        color="#123b43",
    )
    figure.text(
        0.5,
        0.87,
        "Q-first correct · Q-whole wrong · top-2 expert-set Jaccard · pair-count weighted",
        ha="center",
        fontsize=10.5,
        color="#4b6064",
    )
    figure.text(
        0.11,
        0.035,
        "Branching score = t1 overlap − t2 overlap. Positive DiD is descriptive evidence "
        "of route divergence after t2; sampled pairs are not independent trials.",
        fontsize=9.2,
        color="#52666a",
    )
    _save_figure(figure, destination)
    plt.close(figure)


def _plot_diagnostic_overview(
    oracle_rows: Sequence[dict[str, object]],
    layer_rows: Sequence[dict[str, object]],
    attribute_layer_rows: Sequence[dict[str, object]],
    route_headline: dict[str, object],
    destination: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors

    attribute_rows = [row for row in oracle_rows if row["attribute"] != "micro_overall"]
    labels = [ATTRIBUTE_LABELS[str(row["attribute"])] for row in attribute_rows]
    deltas = [100 * float(row["accuracy_delta"]) for row in attribute_rows]
    recovery = [100 * float(row["recovery_rate"]) for row in attribute_rows]
    harm = [100 * float(row["harm_rate"]) for row in attribute_rows]
    layers = [int(row["layer"]) for row in layer_rows]
    did = [float(row["difference_in_differences"]) for row in layer_rows]
    attributes = list(WHOLE_ATTRIBUTES)
    matrix = np.array(
        [
            [
                next(
                    float(row["difference_in_differences"])
                    for row in attribute_layer_rows
                    if row["attribute"] == attribute and int(row["layer"]) == layer
                )
                for layer in layers
            ]
            for attribute in attributes
        ]
    )

    figure = plt.figure(figsize=(15.5, 10.2), facecolor="#f7faf9")
    grid = figure.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.96,
        bottom=0.08,
        top=0.78,
        hspace=0.35,
        wspace=0.25,
    )
    axis_delta = figure.add_subplot(grid[0, 0])
    axis_rates = figure.add_subplot(grid[0, 1])
    axis_layer = figure.add_subplot(grid[1, 0])
    axis_heat = figure.add_subplot(grid[1, 1])
    for axis in (axis_delta, axis_rates, axis_layer, axis_heat):
        axis.set_facecolor("white")

    y = np.arange(len(labels))
    colors_delta = ["#bb3e4a" if value < 0 else "#168a8c" for value in deltas]
    axis_delta.barh(y, deltas, color=colors_delta, height=0.62)
    axis_delta.axvline(0, color="#42575a", linewidth=1)
    axis_delta.set_yticks(y, labels)
    axis_delta.invert_yaxis()
    axis_delta.set_xlabel("Accuracy change (percentage points)")
    axis_delta.set_title("A  Oracle intervention effect", loc="left", fontweight="bold")
    limit = max(abs(min(deltas)), abs(max(deltas))) + 2
    axis_delta.set_xlim(-limit, limit)
    for index, value in enumerate(deltas):
        axis_delta.text(
            value + (0.25 if value >= 0 else -0.25),
            index,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontweight="bold",
            color="#18383d",
        )
    axis_delta.grid(axis="x", alpha=0.15)

    width = 0.36
    axis_rates.bar(y - width / 2, recovery, width, label="errors recovered", color="#2a9d8f")
    axis_rates.bar(y + width / 2, harm, width, label="correct harmed", color="#e76f51")
    axis_rates.set_xticks(y, labels, rotation=24, ha="right")
    axis_rates.set_ylabel("Conditional rate (%)")
    axis_rates.set_title("B  Recovery–harm trade-off", loc="left", fontweight="bold")
    axis_rates.set_ylim(0, max([*recovery, *harm]) * 1.22)
    axis_rates.legend(frameon=False, ncols=2, loc="upper left")
    axis_rates.grid(axis="y", alpha=0.15)

    axis_layer.plot(layers, did, color="#0f7c80", marker="o", linewidth=2.4)
    axis_layer.fill_between(layers, 0, did, color="#5ab4ac", alpha=0.25)
    axis_layer.axhline(0, color="#42575a", linewidth=1)
    axis_layer.axvspan(-0.35, 3.35, color="#f4a261", alpha=0.12)
    axis_layer.text(1.5, max(did) * 0.91, "strongest early-layer signal", ha="center", fontsize=9)
    axis_layer.set_xticks(layers)
    axis_layer.set_xlabel("MoE layer")
    axis_layer.set_ylabel("Difference-in-differences")
    axis_layer.set_title("C  Route branching by layer", loc="left", fontweight="bold")
    axis_layer.grid(alpha=0.15)

    bound = max(abs(float(matrix.min())), abs(float(matrix.max())))
    image = axis_heat.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        norm=colors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
    )
    axis_heat.set_xticks(range(len(layers)), layers)
    axis_heat.set_yticks(
        range(len(attributes)),
        [ATTRIBUTE_LABELS[attribute] for attribute in attributes],
    )
    axis_heat.set_xlabel("MoE layer")
    axis_heat.set_title("D  Controlled branching contrast", loc="left", fontweight="bold")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            face = image.cmap(image.norm(value))
            axis_heat.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.3,
                color=_text_color(face),
            )

    overall = next(row for row in oracle_rows if row["attribute"] == "micro_overall")
    figure.text(
        0.07,
        0.95,
        "Q-WHOLE DIAGNOSTIC STUDY",
        fontsize=23,
        fontweight="bold",
        color="#123b43",
    )
    figure.text(
        0.07,
        0.91,
        "Two inference-only tests of token-conditioned readout and MoE route structure",
        fontsize=12,
        color="#4b6064",
    )
    figure.text(
        0.07,
        0.815,
        f"ORACLE NET  {100 * float(overall['accuracy_delta']):+.2f} pp",
        fontsize=12,
        fontweight="bold",
        color="#9b2f3b",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8e5e7", "edgecolor": "none"},
    )
    figure.text(
        0.38,
        0.815,
        f"BAD CASES  {int(route_headline['bad_case_examples']):,}",
        fontsize=12,
        fontweight="bold",
        color="#155d62",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#dff0ef", "edgecolor": "none"},
    )
    figure.text(
        0.69,
        0.815,
        f"ROUTE DiD  {float(route_headline['difference_in_differences']):+.3f}",
        fontsize=12,
        fontweight="bold",
        color="#155d62",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#dff0ef", "edgecolor": "none"},
    )
    figure.text(
        0.07,
        0.025,
        "Scope: multi5+permute formal · 50,118 held-out people per attribute · unchanged probes. "
        "Route contrasts are descriptive and conditioned on Q-first-correct/Q-whole-wrong cases.",
        fontsize=9.2,
        color="#52666a",
    )
    _save_figure(figure, destination)
    plt.close(figure)


def _plot_full_whole_comparison(
    whole_rows: Sequence[dict[str, object]],
    oracle_rows: Sequence[dict[str, object]],
    destination: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors

    attributes = list(WHOLE_ATTRIBUTES)
    conditions = ("single", "multi5_permute")
    p_matrices = {}
    for condition in conditions:
        p_matrices[condition] = np.array(
            [
                [
                    next(
                        float(row["accuracy"])
                        for row in whole_rows
                        if row["condition"] == condition
                        and row["split"] == "person_held_out_validation"
                        and row["kind"] == "p"
                        and row["attribute"] == attribute
                        and int(row["position"]) == position
                    )
                    for position in range(6)
                ]
                for attribute in attributes
            ]
        )
    q_single = [
        next(
            float(row["accuracy"])
            for row in whole_rows
            if row["condition"] == "single"
            and row["split"] == "person_held_out_validation"
            and row["kind"] == "q"
            and row["attribute"] == attribute
        )
        for attribute in attributes
    ]
    q_multi = [
        next(
            float(row["accuracy"])
            for row in whole_rows
            if row["condition"] == "multi5_permute"
            and row["split"] == "person_held_out_validation"
            and row["kind"] == "q"
            and row["attribute"] == attribute
        )
        for attribute in attributes
    ]
    oracle_by_attribute = {
        str(row["attribute"]): float(row["accuracy_after"])
        for row in oracle_rows
        if row["attribute"] != "micro_overall"
    }
    q_oracle = [oracle_by_attribute[attribute] for attribute in attributes]
    allen_multi = [value / 100 for value in ALLEN_ZHU_Q_REFERENCE["multi5_permute"]["whole"]]

    figure = plt.figure(figsize=(16, 10.8), facecolor="#f7faf9")
    grid = figure.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.94,
        bottom=0.08,
        top=0.86,
        hspace=0.32,
        wspace=0.28,
    )
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    heatmap = plt.get_cmap("GnBu")
    for axis, condition, title in zip(
        axes,
        conditions,
        ("A  Single formal · P-whole", "B  Multi5+permute formal · P-whole"),
    ):
        matrix = 100 * p_matrices[condition]
        image = axis.imshow(matrix, aspect="auto", cmap=heatmap, vmin=0, vmax=100)
        axis.set_xticks(range(6), [f"P{i}" for i in range(6)])
        axis.set_yticks(
            range(len(attributes)),
            [ATTRIBUTE_LABELS[attribute] for attribute in attributes],
        )
        axis.set_xlabel("Biography observation position")
        axis.set_title(title, loc="left", fontweight="bold")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                face = image.cmap(image.norm(value))
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=_text_color(face),
                    fontweight="bold" if value >= 95 else "normal",
                )
        figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03, label="Accuracy (%)")

    axis_q = figure.add_subplot(grid[1, :])
    axis_q.axis("off")
    q_values = np.array([q_single, q_multi, q_oracle, allen_multi]) * 100
    q_labels = (
        "THIS WORK · SINGLE · NAME ONLY",
        "THIS WORK · MULTI5+PERMUTE · NAME ONLY",
        "THIS WORK · MULTI5+PERMUTE · + TRUE t1",
        "ALLEN–ZHU · MULTI5+PERMUTE · NAME ONLY",
    )
    headers = [ATTRIBUTE_LABELS[attribute].upper() for attribute in attributes] + ["MACRO"]
    q_with_macro = np.column_stack((q_values, q_values.mean(axis=1)))
    table = axis_q.table(
        cellText=[[f"{value:.1f}" for value in row] for row in q_with_macro],
        rowLabels=q_labels,
        colLabels=headers,
        cellLoc="center",
        rowLoc="center",
        bbox=[0.18, 0.08, 0.78, 0.74],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    norm = colors.Normalize(0, 100)
    for row_index, row in enumerate(q_with_macro, start=1):
        for column_index, value in enumerate(row):
            face = heatmap(norm(value))
            cell = table[(row_index, column_index)]
            cell.set_facecolor(face)
            cell.set_edgecolor("white")
            cell.get_text().set_color(_text_color(face))
            if column_index == len(headers) - 1:
                cell.get_text().set_weight("bold")
        label_cell = table[(row_index, -1)]
        label_cell.set_facecolor("#e9f3f3" if row_index < 4 else "#ece8dc")
        label_cell.set_edgecolor("white")
        label_cell.get_text().set_weight("bold")
    for column_index in range(len(headers)):
        cell = table[(0, column_index)]
        cell.set_facecolor("#123b43")
        cell.set_edgecolor("white")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
    axis_q.set_title(
        "C  Q-whole: original formal readout, oracle intervention, and paper reference",
        loc="left",
        fontweight="bold",
        pad=8,
    )

    figure.text(
        0.08,
        0.95,
        "COMPLETE WHOLE-ATTRIBUTE READOUT COMPARISON",
        fontsize=22,
        fontweight="bold",
        color="#123b43",
    )
    figure.text(
        0.08,
        0.91,
        "All five whole-value tasks · all six P positions · Q name-only and true-t1 intervention",
        fontsize=11.5,
        color="#4b6064",
    )
    figure.text(
        0.08,
        0.025,
        "All project values are person-held-out probe validation. Allen–Zhu values are bioS "
        "Figure 7 context, not matched architecture/data/budget reproduction.",
        fontsize=9.2,
        color="#52666a",
    )
    _save_figure(figure, destination)
    plt.close(figure)


def build_diagnostic_report_artifacts(
    *,
    single_formal_root: str | Path,
    multi_formal_root: str | Path,
    diagnostics_root: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Audit both diagnostic runs and build their canonical report artifacts."""

    evidence = load_diagnostic_evidence(
        single_formal_root,
        multi_formal_root,
        diagnostics_root,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    oracle_rows = evidence["oracle_rows"]
    layer_rows = evidence["route_layer_rows"]
    attribute_layer_rows = evidence["route_attribute_layer_rows"]
    route_headline = evidence["route_headline"]
    single: FormalRun = evidence["single_formal"]
    formal: FormalRun = evidence["formal"]
    whole_rows = _formal_whole_rows(single, formal)

    _atomic_csv(output / "oracle_metrics.csv", oracle_rows)
    _atomic_csv(output / "route_layer_metrics.csv", layer_rows)
    _atomic_csv(output / "route_attribute_layer_metrics.csv", attribute_layer_rows)
    _atomic_csv(output / "formal_whole_metrics.csv", whole_rows)
    _plot_oracle_table(oracle_rows, figures / "oracle_intervention_table")
    _plot_route_table(route_headline, figures / "route_branching_table")
    _plot_diagnostic_overview(
        oracle_rows,
        layer_rows,
        attribute_layer_rows,
        route_headline,
        figures / "diagnostic_study_overview",
    )
    _plot_full_whole_comparison(
        whole_rows,
        oracle_rows,
        figures / "complete_whole_comparison",
    )

    overall = next(row for row in oracle_rows if row["attribute"] == "micro_overall")
    identity = formal.identity
    summary = {
        "status": "completed",
        "protocols": {
            "oracle": ORACLE_PROTOCOL,
            "routes": ROUTE_PROTOCOL,
        },
        "identity": {
            "condition": "multi5_permute",
            "single_formal_root": str(single.root),
            "formal_root": str(formal.root),
            "checkpoint": identity["checkpoint"],
            "checkpoint_model_sha256": identity["checkpoint_model_sha256"],
            "data": identity["data"],
            "data_manifest_sha256": identity["data_manifest_sha256"],
            "probe_cache": identity["probe_cache"],
            "probe_cache_manifest_sha256": identity["probe_cache_manifest_sha256"],
            "model_config_sha256": identity["model_config_sha256"],
            "seed": identity["seed"],
            "split": EXPECTED_SPLIT,
        },
        "oracle_headline": {
            key: overall[key]
            for key in (
                "examples",
                "accuracy_before",
                "accuracy_after",
                "accuracy_delta",
                "recovered_errors",
                "recovery_rate",
                "harmed_correct",
                "harm_rate",
            )
        },
        "route_headline": route_headline,
        "raw_artifacts": evidence["raw_manifest"],
        "artifacts": {
            "oracle_metrics": "oracle_metrics.csv",
            "route_layer_metrics": "route_layer_metrics.csv",
            "route_attribute_layer_metrics": "route_attribute_layer_metrics.csv",
            "formal_whole_metrics": "formal_whole_metrics.csv",
            "figures": sorted(
                path.relative_to(output).as_posix() for path in figures.glob("*.png")
            ),
        },
    }
    _atomic_json(output / "summary.json", summary)
    return summary
