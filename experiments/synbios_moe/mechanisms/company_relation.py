"""Fit exposure-position curves for the company and company-city probes."""


import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ATTRIBUTES = ("birth_city", "university", "major", "company", "company_city")
DISPLAY_NAMES = {
    "birth_city": "Birth city",
    "university": "University",
    "major": "Major",
    "company": "Company",
    "company_city": "Company city",
}


def _load_curves(path: Path) -> dict[str, np.ndarray]:
    positions: dict[str, dict[int, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not (
                row["condition"] == "multi5_permute"
                and row["split"] == "person_held_out_validation"
                and row["kind"] == "p"
                and row["target"] == "whole"
            ):
                continue
            positions.setdefault(row["attribute"], {})[int(row["position"])] = float(
                row["accuracy"]
            )
    missing = [
        attribute
        for attribute in ATTRIBUTES
        if set(positions.get(attribute, {})) != set(range(6))
    ]
    if missing:
        raise ValueError(f"missing complete P-whole curves for: {missing}")
    return {
        attribute: np.asarray([positions[attribute][index] for index in range(6)])
        for attribute in ATTRIBUTES
    }


def _paired_exposure() -> np.ndarray:
    return np.asarray(
        [
            1.0 - math.comb(4, index) / math.comb(6, index) if index <= 4 else 1.0
            for index in range(6)
        ],
        dtype=np.float64,
    )


def _fit(y: np.ndarray, x: np.ndarray) -> dict[str, object]:
    design = np.column_stack((np.ones(len(x)), x))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ coefficients
    residual = y - prediction
    rss = float(np.sum(np.square(residual)))
    total = float(np.sum(np.square(y - y.mean())))
    leave_one_out_errors = []
    for held_out in range(len(y)):
        keep = np.arange(len(y)) != held_out
        fold_coefficients = np.linalg.lstsq(
            design[keep], y[keep], rcond=None
        )[0]
        leave_one_out_errors.append(
            float(y[held_out] - design[held_out] @ fold_coefficients)
        )
    return {
        "baseline": float(coefficients[0]),
        "gain": float(coefficients[1]),
        "saturation": float(coefficients.sum()),
        "prediction": prediction.tolist(),
        "rmse": float(math.sqrt(rss / len(y))),
        "r_squared": float(1.0 - rss / total) if total else None,
        "leave_one_out_rmse": float(
            math.sqrt(np.mean(np.square(leave_one_out_errors)))
        ),
    }


def _write_metrics(
    path: Path,
    curves: dict[str, np.ndarray],
    fits: dict[str, dict[str, object]],
) -> None:
    fields = (
        "attribute",
        "baseline",
        "gain",
        "saturation",
        "rmse",
        "r_squared",
        "leave_one_out_rmse",
        "observed_p0",
        "observed_p5",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for attribute in ATTRIBUTES:
            fit = fits[attribute]
            writer.writerow(
                {
                    "attribute": attribute,
                    **{key: fit[key] for key in fields[1:7]},
                    "observed_p0": float(curves[attribute][0]),
                    "observed_p5": float(curves[attribute][-1]),
                }
            )


def _plot(
    output: Path,
    curves: dict[str, np.ndarray],
    fits: dict[str, dict[str, object]],
) -> None:
    colors = {
        "observed": "#172b4d",
        "paired_fields": "#e45756",
    }
    positions = np.arange(6)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)
    for axis, attribute in zip(axes, ("company", "company_city")):
        observed = curves[attribute] * 100.0
        paired_fit = fits[attribute]
        axis.plot(
            positions,
            np.asarray(paired_fit["prediction"]) * 100.0,
            "-",
            color=colors["paired_fields"],
            linewidth=2.8,
            label="Company OR company-city",
        )
        axis.scatter(
            positions,
            observed,
            s=68,
            color=colors["observed"],
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
            label="Observed held-out accuracy",
        )
        for index, value in enumerate(observed):
            axis.annotate(
                f"{value:.1f}",
                (index, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=colors["observed"],
            )
        axis.set_title(DISPLAY_NAMES[attribute], fontsize=15, weight="bold")
        axis.set_xlabel("P-probe position", fontsize=11)
        axis.set_xticks(positions)
        axis.set_xticklabels([f"P{index}" for index in positions])
        axis.grid(axis="y", color="#d9e2ec", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.03,
            0.97,
            (
                "Paired-field fit\n"
                f"$R^2$ = {paired_fit['r_squared']:.5f}\n"
                f"RMSE = {100 * paired_fit['rmse']:.3f} pp\n"
                f"LOO RMSE = {100 * paired_fit['leave_one_out_rmse']:.3f} pp"
            ),
            transform=axis.transAxes,
            va="top",
            fontsize=9.5,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#fff5f4",
                "edgecolor": "#e45756",
                "alpha": 0.95,
            },
        )
    axes[0].set_ylabel("P-whole held-out accuracy (%)", fontsize=11)
    axes[0].set_ylim(38, 104)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "Position growth follows exposure to either linked work attribute",
        fontsize=17,
        weight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.085,
        (
            "Paired exposure probability: "
            r"$1-\binom{4}{j}/\binom{6}{j}$ = "
            "0%, 33.3%, 60.0%, 80.0%, 93.3%, 100%"
        ),
        ha="center",
        fontsize=9.5,
        color="#52606d",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.98))
    fig.savefig(
        output.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "mini-train-sys"},
    )
    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Creator": "mini-train-sys",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def fit_company_pair(input_csv: Path, output_dir: Path) -> dict[str, object]:
    curves = _load_curves(input_csv)
    paired_exposure = _paired_exposure()
    fits = {
        attribute: _fit(curve, paired_exposure)
        for attribute, curve in curves.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(output_dir / "fit_metrics.csv", curves, fits)
    _plot(output_dir / "company_pair_position_fit", curves, fits)
    summary = {
        "protocol": "synbios_company_pair_position_fit_v1",
        "input": input_csv.as_posix(),
        "condition": "multi5_permute",
        "split": "person_held_out_validation",
        "endpoint": "p_whole",
        "positions": list(range(6)),
        "paired_exposure_probability": paired_exposure.tolist(),
        "observed": {key: values.tolist() for key, values in curves.items()},
        "fits": fits,
        "headline": {
            "company_paired_r_squared": fits["company"]["r_squared"],
            "company_paired_rmse": fits["company"]["rmse"],
            "company_city_paired_r_squared": fits["company_city"]["r_squared"],
            "company_city_paired_rmse": fits["company_city"]["rmse"],
            "interpretation": (
                "The position curves follow the probability that either company or "
                "company_city has appeared earlier in the randomly permuted biography."
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = fit_company_pair(args.input, args.output)
    print(json.dumps(summary["headline"], indent=2))


if __name__ == "__main__":
    main()
