"""Audited formal-study comparison and publication-quality SynBioS figures.

This module is deliberately read-only with respect to experiment runs. It
reconstructs metrics from the 22 independent validation JSON files rather than
trusting a potentially stale aggregate, and rejects mismatched run identities
before producing comparison artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


ATTRIBUTES = (
    "birth_date",
    "birth_city",
    "university",
    "major",
    "company",
    "company_city",
)
WHOLE_ATTRIBUTES = ATTRIBUTES[1:]
ATTRIBUTE_LABELS = {
    "birth_date": "Birth month",
    "birth_city": "Birth city",
    "university": "University",
    "major": "Major",
    "company": "Company",
    "company_city": "Company city",
}
EXPECTED_TASKS = tuple(
    f"{kind}_{attribute}_{target}"
    for attribute in ATTRIBUTES
    for target in ("first", "whole")
    if not (attribute == "birth_date" and target == "whole")
    for kind in ("p", "q")
)

# ArXiv v3, Figure 7, rows bioS single and bioS multi5+permute.
# These values are contextual references, not inputs to project metrics.
ALLEN_ZHU_Q_REFERENCE = {
    "source": "Physics of Language Models Part 3.1, arXiv:2309.14316v3, Figure 7",
    "single": {
        "first": [63.4, 1.9, 37.5, 3.1, 0.2, 13.1],
        "whole": [1.1, 0.3, 1.4, 0.1, 11.6],
    },
    "multi5_permute": {
        "first": [100.0, 100.0, 99.9, 100.0, 99.9, 99.8],
        "whole": [96.1, 72.6, 94.9, 99.6, 99.7],
    },
}


@dataclass(frozen=True)
class FormalRun:
    """Validated formal-run inputs and task-level records."""

    condition: str
    root: Path
    pipeline: dict[str, object]
    data_manifest: dict[str, object]
    cache_manifest: dict[str, object]
    validation: dict[str, dict[str, object]]
    training: dict[str, dict[str, object]]

    @property
    def identity(self) -> dict[str, object]:
        return dict(self.pipeline["identity"])

    @property
    def profiles_sha256(self) -> str:
        files = self.data_manifest["files"]
        return str(files["profiles.jsonl"]["sha256"])


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


def _task_files(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.glob("*.json")
        if path.stem in EXPECTED_TASKS
    }


def load_formal_run(condition: str, root: str | Path) -> FormalRun:
    """Load and strictly validate one completed 22-head formal run."""

    root = Path(root).resolve()
    pipeline = _read_json(root / "pipeline.json")
    if pipeline.get("status") != "completed" or pipeline.get("stage") != "formal":
        raise ValueError(f"{root} is not a completed formal pipeline")
    identity = pipeline.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{root}/pipeline.json has no identity")
    jobs = tuple(identity.get("jobs", ()))
    if set(jobs) != set(EXPECTED_TASKS) or len(jobs) != len(EXPECTED_TASKS):
        raise ValueError(f"{root} does not contain the exact 22-task probe matrix")
    if set(item.get("status") for item in pipeline.get("training", ())) != {"completed"}:
        raise ValueError(f"{root} has incomplete training tasks")
    if set(item.get("status") for item in pipeline.get("validation", ())) != {"completed"}:
        raise ValueError(f"{root} has incomplete validation tasks")

    data_root = Path(str(identity["data"])).resolve()
    cache_root = Path(str(identity["probe_cache"])).resolve()
    data_manifest_path = data_root / "manifest.json"
    cache_manifest_path = cache_root / "manifest.json"
    if _sha256(data_manifest_path) != identity["data_manifest_sha256"]:
        raise ValueError(f"{condition} data manifest SHA256 does not match pipeline identity")
    if _sha256(cache_manifest_path) != identity["probe_cache_manifest_sha256"]:
        raise ValueError(f"{condition} cache manifest SHA256 does not match pipeline identity")
    data_manifest = _read_json(data_manifest_path)
    cache_manifest = _read_json(cache_manifest_path)

    validation_paths = _task_files(root / "validation")
    training_paths = _task_files(root / "training")
    expected = set(EXPECTED_TASKS)
    if set(validation_paths) != expected or set(training_paths) != expected:
        raise ValueError(f"{condition} does not have exactly 22 training and validation JSON files")

    validation: dict[str, dict[str, object]] = {}
    training: dict[str, dict[str, object]] = {}
    checkpoint = Path(str(identity["checkpoint"])).resolve()
    for task in EXPECTED_TASKS:
        validation_record = _read_json(validation_paths[task])
        training_record = _read_json(training_paths[task])
        kind, remainder = task.split("_", 1)
        target = "whole" if remainder.endswith("_whole") else "first"
        attribute = remainder.removesuffix(f"_{target}")
        expected_metadata = (kind, attribute, target)
        for record, label in ((validation_record, "validation"), (training_record, "training")):
            actual = (record.get("kind"), record.get("attribute"), record.get("target"))
            if actual != expected_metadata:
                raise ValueError(f"{condition}/{task} {label} metadata mismatch: {actual}")
            if Path(str(record["checkpoint"])).resolve() != checkpoint:
                raise ValueError(f"{condition}/{task} uses the wrong backbone checkpoint")
            if record.get("probe_cache_manifest_sha256") not in (
                None,
                identity["probe_cache_manifest_sha256"],
            ):
                raise ValueError(f"{condition}/{task} uses the wrong probe cache")
        if validation_record.get("class_names") != training_record.get("class_names"):
            raise ValueError(f"{condition}/{task} class mapping differs between train and validation")
        expected_positions = 6 if kind == "p" else 1
        if len(validation_record.get("validation_accuracy", ())) != expected_positions:
            raise ValueError(f"{condition}/{task} has an invalid validation position count")
        if len(training_record.get("train_accuracy", ())) != expected_positions:
            raise ValueError(f"{condition}/{task} has an invalid train position count")
        validation[task] = validation_record
        training[task] = training_record

    return FormalRun(
        condition=condition,
        root=root,
        pipeline=pipeline,
        data_manifest=data_manifest,
        cache_manifest=cache_manifest,
        validation=validation,
        training=training,
    )


def validate_matched_runs(single: FormalRun, multi: FormalRun) -> None:
    """Require a matched scientific comparison, except for data augmentation."""

    if single.profiles_sha256 != multi.profiles_sha256:
        raise ValueError("single and multi runs do not use the same person/profile table")
    single_identity, multi_identity = single.identity, multi.identity
    for field in ("model_config_sha256", "seed", "steps", "runtime", "jobs"):
        if single_identity[field] != multi_identity[field]:
            raise ValueError(f"formal comparison mismatch in identity.{field}")
    for task in EXPECTED_TASKS:
        left, right = single.validation[task], multi.validation[task]
        for field in ("classes", "class_names"):
            if left[field] != right[field]:
                raise ValueError(f"formal comparison mismatch in {task}.{field}")
    if single.data_manifest.get("variant") != "single":
        raise ValueError("single run does not identify variant=single")
    if multi.data_manifest.get("variant") != "multi5+permute":
        raise ValueError("multi run does not identify variant=multi5+permute")


def _training_examples(record: dict[str, object]) -> int:
    monitoring = record.get("monitoring", {})
    train_evaluation = monitoring.get("train_evaluation", {}) if isinstance(monitoring, dict) else {}
    return int(train_evaluation.get("items_processed", 0))


def tidy_rows(runs: Sequence[FormalRun]) -> list[dict[str, object]]:
    """Return train-recall and held-out-validation metrics in tidy form."""

    rows: list[dict[str, object]] = []
    for run in runs:
        for task in EXPECTED_TASKS:
            validation = run.validation[task]
            training = run.training[task]
            for split, record, field, examples in (
                (
                    "probe_train_recall",
                    training,
                    "train_accuracy",
                    _training_examples(training),
                ),
                (
                    "person_held_out_validation",
                    validation,
                    "validation_accuracy",
                    int(validation["examples"]),
                ),
            ):
                for position, accuracy in enumerate(record[field]):
                    rows.append(
                        {
                            "condition": run.condition,
                            "split": split,
                            "task": task,
                            "kind": validation["kind"],
                            "target": validation["target"],
                            "attribute": validation["attribute"],
                            "position": position,
                            "accuracy": float(accuracy),
                            "examples": examples,
                            "classes": int(validation["classes"]),
                        }
                    )
    return rows


def _accuracy(
    rows: Sequence[dict[str, object]],
    condition: str,
    kind: str,
    target: str,
    attribute: str,
    position: int,
    split: str = "person_held_out_validation",
) -> float:
    matches = [
        float(row["accuracy"])
        for row in rows
        if row["condition"] == condition
        and row["kind"] == kind
        and row["target"] == target
        and row["attribute"] == attribute
        and row["position"] == position
        and row["split"] == split
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one metric for {condition}/{kind}/{target}/{attribute}/P{position}/{split}"
        )
    return matches[0]


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def headline_metrics(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Compute report endpoints without mixing task granularity or splits."""

    output: dict[str, object] = {}
    for condition in ("single", "multi5_permute"):
        p0_first_nondate = [
            _accuracy(rows, condition, "p", "first", attribute, 0)
            for attribute in ATTRIBUTES[1:]
        ]
        q_first = [
            _accuracy(rows, condition, "q", "first", attribute, 0)
            for attribute in ATTRIBUTES
        ]
        p0_whole = [
            _accuracy(rows, condition, "p", "whole", attribute, 0)
            for attribute in WHOLE_ATTRIBUTES
        ]
        p5_whole = [
            _accuracy(rows, condition, "p", "whole", attribute, 5)
            for attribute in WHOLE_ATTRIBUTES
        ]
        q_whole = [
            _accuracy(rows, condition, "q", "whole", attribute, 0)
            for attribute in WHOLE_ATTRIBUTES
        ]
        diagonal_first = [
            _accuracy(rows, condition, "p", "first", attribute, index)
            for index, attribute in enumerate(ATTRIBUTES)
        ]
        output[condition] = {
            "p_first_position0_non_birth_date_mean": _mean(p0_first_nondate),
            "p_first_fixed_order_diagonal_mean": _mean(diagonal_first),
            "q_first_six_attribute_macro_mean": _mean(q_first),
            "p_whole_position0_five_attribute_macro_mean": _mean(p0_whole),
            "p_whole_position5_five_attribute_macro_mean": _mean(p5_whole),
            "q_whole_five_attribute_macro_mean": _mean(q_whole),
        }
    output["delta_multi_minus_single"] = {
        key: output["multi5_permute"][key] - output["single"][key]
        for key in output["single"]
    }
    output["allen_zhu_q_reference_macro_means"] = {
        condition: {
            target: _mean([value / 100 for value in values])
            for target, values in targets.items()
        }
        for condition, targets in ALLEN_ZHU_Q_REFERENCE.items()
        if condition != "source"
    }
    return output


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _paper_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "paper_teal",
        ("#f5f1e8", "#d9eee8", "#85c9ba", "#24948a", "#075b62"),
    )


def _annotated_heatmap(
    axis,
    matrix: np.ndarray,
    *,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    title: str,
) -> object:
    image = axis.imshow(matrix, cmap=_paper_cmap(), vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(len(xlabels)), xlabels)
    axis.set_yticks(range(len(ylabels)), ylabels)
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    axis.tick_params(length=0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if value >= 68 else "#17343a"
            text = f"{value:.0f}" if value >= 99.95 else f"{value:.1f}"
            axis.text(column, row, text, ha="center", va="center", fontsize=8.5, color=color)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def _save_figure(figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def _matrix(
    rows: Sequence[dict[str, object]],
    condition: str,
    kind: str,
    target: str,
    attributes: Sequence[str],
) -> np.ndarray:
    width = 6 if kind == "p" else 1
    return np.asarray(
        [
            [
                100 * _accuracy(rows, condition, kind, target, attribute, position)
                for position in range(width)
            ]
            for attribute in attributes
        ]
    )


def plot_p_heatmaps(
    rows: Sequence[dict[str, object]],
    *,
    target: str,
    destination: Path,
) -> None:
    import matplotlib.pyplot as plt

    attributes = ATTRIBUTES if target == "first" else WHOLE_ATTRIBUTES
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True)
    for axis, condition, label in zip(
        axes,
        ("single", "multi5_permute"),
        ("Single biography", "5× paraphrase + permutation"),
    ):
        matrix = _matrix(rows, condition, "p", target, attributes)
        image = _annotated_heatmap(
            axis,
            matrix,
            xlabels=[f"P{index}" for index in range(6)],
            ylabels=[ATTRIBUTE_LABELS[value] for value in attributes],
            title=label,
        )
        axis.set_xlabel("Biography position (left → right)")
    label = "first-token" if target == "first" else "whole-attribute"
    figure.suptitle(
        f"P-probe · {label} held-out accuracy",
        fontsize=17,
        fontweight="bold",
        x=0.5,
        ha="center",
    )
    figure.colorbar(image, ax=axes, label="Accuracy (%)", fraction=0.025, pad=0.02)
    _save_figure(figure, destination)
    plt.close(figure)


def plot_q_table(rows: Sequence[dict[str, object]], destination: Path) -> None:
    import matplotlib.pyplot as plt

    values = []
    for condition in ("single", "multi5_permute"):
        values.append(
            [
                *(
                    100 * _accuracy(rows, condition, "q", "first", attribute, 0)
                    for attribute in ATTRIBUTES
                ),
                *(
                    100 * _accuracy(rows, condition, "q", "whole", attribute, 0)
                    for attribute in WHOLE_ATTRIBUTES
                ),
            ]
        )
    matrix = np.asarray(values)
    labels = [
        *(f"{ATTRIBUTE_LABELS[value]}\nfirst" for value in ATTRIBUTES),
        *(f"{ATTRIBUTE_LABELS[value]}\nwhole" for value in WHOLE_ATTRIBUTES),
    ]
    figure, axis = plt.subplots(figsize=(15.5, 3.6), constrained_layout=True)
    image = _annotated_heatmap(
        axis,
        matrix,
        xlabels=labels,
        ylabels=("Single biography", "5× paraphrase + permutation"),
        title="Name-only Q-probe · person-held-out validation",
    )
    axis.axvline(5.5, color="white", linewidth=5)
    axis.text(2.5, -0.88, "FIRST TOKEN", ha="center", fontsize=10, color="#46636a")
    axis.text(8.0, -0.88, "WHOLE ATTRIBUTE", ha="center", fontsize=10, color="#46636a")
    axis.tick_params(axis="x", labelrotation=35)
    figure.colorbar(image, ax=axis, label="Accuracy (%)", fraction=0.025, pad=0.02)
    _save_figure(figure, destination)
    plt.close(figure)


def plot_overview(
    rows: Sequence[dict[str, object]],
    *,
    cloze: dict[str, dict[str, object]],
    destination: Path,
) -> None:
    import matplotlib.pyplot as plt

    colors = ("#9b6a50", "#138a82")
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    conditions = ("single", "multi5_permute")
    condition_labels = ("Single", "5× + permute")
    cloze_values = [100 * float(cloze[name]["micro_field_accuracy"]) for name in conditions]
    axes[0].bar(condition_labels, cloze_values, color=colors, width=0.62)
    axes[0].set_ylim(99.85, 100.01)
    axes[0].set_ylabel("Exact field accuracy (%)")
    axes[0].set_title("A  Training-corpus source recall", loc="left", fontweight="bold")
    for index, value in enumerate(cloze_values):
        axes[0].text(index, value + 0.006, f"{value:.3f}%", ha="center", fontsize=10)

    attributes = ATTRIBUTES[1:]
    x = np.arange(len(attributes))
    width = 0.36
    for offset, condition, label, color in zip(
        (-width / 2, width / 2), conditions, condition_labels, colors
    ):
        values = [
            100 * _accuracy(rows, condition, "p", "first", attribute, 0)
            for attribute in attributes
        ]
        axes[1].bar(x + offset, values, width, label=label, color=color)
    axes[1].set_xticks(x, [ATTRIBUTE_LABELS[value] for value in attributes], rotation=30, ha="right")
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Held-out accuracy (%)")
    axes[1].set_title("B  Earliest P-position · first token", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="upper left")

    first_means = [
        100
        * _mean(
            [_accuracy(rows, condition, "q", "first", attribute, 0) for attribute in ATTRIBUTES]
        )
        for condition in conditions
    ]
    whole_means = [
        100
        * _mean(
            [
                _accuracy(rows, condition, "q", "whole", attribute, 0)
                for attribute in WHOLE_ATTRIBUTES
            ]
        )
        for condition in conditions
    ]
    x2 = np.arange(2)
    axes[2].bar(x2 - width / 2, first_means, width, label="First token", color="#297b86")
    axes[2].bar(x2 + width / 2, whole_means, width, label="Whole attribute", color="#d29c62")
    axes[2].set_xticks(x2, condition_labels)
    axes[2].set_ylim(0, 105)
    axes[2].set_ylabel("Macro held-out accuracy (%)")
    axes[2].set_title("C  Name-only Q-probe", loc="left", fontweight="bold")
    axes[2].legend(frameon=False, loc="upper left")
    for container in axes[2].containers:
        axes[2].bar_label(container, fmt="%.1f", padding=3, fontsize=9)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle(
        "Knowledge augmentation changes where facts become linearly readable",
        x=0.01,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    _save_figure(figure, destination)
    plt.close(figure)


def build_formal_report_artifacts(
    *,
    single_root: str | Path,
    multi_root: str | Path,
    single_cloze: str | Path,
    multi_cloze: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Validate inputs and write machine tables plus polished formal figures."""

    single = load_formal_run("single", single_root)
    multi = load_formal_run("multi5_permute", multi_root)
    validate_matched_runs(single, multi)
    rows = tidy_rows((single, multi))
    metrics = headline_metrics(rows)
    cloze = {
        "single": _read_json(Path(single_cloze).resolve()),
        "multi5_permute": _read_json(Path(multi_cloze).resolve()),
    }
    for condition, run in (("single", single), ("multi5_permute", multi)):
        if Path(str(cloze[condition]["checkpoint"])).resolve() != Path(
            str(run.identity["checkpoint"])
        ).resolve():
            raise ValueError(f"{condition} cloze result uses the wrong checkpoint")
        if cloze[condition].get("protocol") != "progressive_original_biography_cloze_greedy":
            raise ValueError(f"{condition} cloze result uses the wrong validation protocol")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output / "formal_probe_metrics.csv", rows)
    _atomic_json(output / "headline_metrics.json", metrics)
    _atomic_json(output / "allen_zhu_q_reference.json", ALLEN_ZHU_Q_REFERENCE)
    identity = {
        "protocol": "synbios_moe_formal_comparison_v1",
        "comparison_status": "matched",
        "single": {
            "formal_root": str(single.root),
            "pipeline_identity": single.identity,
            "profiles_sha256": single.profiles_sha256,
        },
        "multi5_permute": {
            "formal_root": str(multi.root),
            "pipeline_identity": multi.identity,
            "profiles_sha256": multi.profiles_sha256,
        },
        "cloze": {
            condition: {
                "path": str(Path(path).resolve()),
                "sha256": _sha256(Path(path).resolve()),
                "biographies": cloze[condition]["biographies"],
                "fields": cloze[condition]["fields"],
                "micro_field_accuracy": cloze[condition]["micro_field_accuracy"],
            }
            for condition, path in (
                ("single", single_cloze),
                ("multi5_permute", multi_cloze),
            )
        },
    }
    _atomic_json(output / "run_identity.json", identity)
    figures = output / "figures"
    plot_p_heatmaps(
        rows,
        target="first",
        destination=figures / "formal_p_first_heatmaps",
    )
    plot_p_heatmaps(
        rows,
        target="whole",
        destination=figures / "formal_p_whole_heatmaps",
    )
    plot_q_table(rows, figures / "formal_q_probe_table")
    plot_overview(rows, cloze=cloze, destination=figures / "formal_study_overview")
    summary = {
        "identity": identity,
        "headline_metrics": metrics,
        "allen_zhu_q_reference": ALLEN_ZHU_Q_REFERENCE,
        "artifacts": {
            "tidy_metrics": "formal_probe_metrics.csv",
            "figures": sorted(
                path.relative_to(output).as_posix() for path in figures.glob("*.png")
            ),
        },
    }
    _atomic_json(output / "summary.json", summary)
    return summary
