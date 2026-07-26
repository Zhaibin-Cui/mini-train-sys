"""Generate compact portfolio figures from canonical machine-readable results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_route_layer_did(input_csv: Path, output_stem: Path) -> None:
    with input_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    layers = [int(row["layer"]) for row in rows]
    did = [float(row["difference_in_differences"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    colors = ["#f97316" if layer <= 3 else "#2563eb" for layer in layers]
    ax.bar(layers, did, color=colors, width=0.72, edgecolor="white", linewidth=0.8)
    ax.axhline(0.0, color="#334155", linewidth=1.0)
    ax.set_xticks(layers)
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("Route branching difference-in-differences")
    ax.set_title("Token-conditioned MoE route branching by layer")
    ax.grid(axis="y", alpha=0.22)
    ax.text(
        0.99,
        0.96,
        "orange: layers 0–3",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#9a3412",
    )
    for layer, value in zip(layers, did, strict=True):
        ax.text(layer, value + 0.014, f"{value:.3f}", ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-layer-csv", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_route_layer_did(args.route_layer_csv, args.output_stem)


if __name__ == "__main__":
    main()
