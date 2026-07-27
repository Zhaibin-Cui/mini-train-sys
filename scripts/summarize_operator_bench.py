"""Merge isolated dense/MoE scans into the canonical kernel benchmark report."""


import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORMAL_TOKENS = 112 * 512
HEADLINE_KERNELS = (
    "rmsnorm",
    "rope",
    "swiglu",
    "cross_entropy",
    "fused_linear_cross_entropy",
    "attention",
    "router_postprocess",
    "fused_moe",
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: _safe(value) for key, value in row.items()} for row in rows])
    temporary.replace(path)


def _accepted(row: dict[str, object]) -> bool:
    return (
        row.get("status") == "ok"
        and row.get("fwd_correct") is True
        and row.get("bwd_correct") is True
    )


def _providers(kernel: str) -> tuple[str, ...]:
    return ("torch", "triton", "cuda") if kernel == "attention" else ("torch", "triton")


def _profile(kernel: str) -> str:
    return "default" if kernel == "router_postprocess" else "project_formal"


def _load_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    source_summaries = []
    for path in (Path(args.dense), Path(args.moe)):
        payload = _read(path)
        source_summaries.append(
            {
                "path": str(path),
                "quality_gate_passed": payload["quality_gate_passed"],
                "case_failures": len(payload["case_failures"]),
                "correctness_failures": len(payload["correctness_failures"]),
            }
        )
        for row in payload["rows"]:
            copied = dict(row)
            copied["source_artifact"] = str(path)
            rows.append(copied)

    for path in sorted(Path(args.extensions).glob("*.json")):
        payload = _read(path)
        profile = payload.get("profile")
        profile_name = profile.get("name") if isinstance(profile, dict) else "default"
        for row in payload["rows"]:
            copied = dict(row)
            copied.setdefault("profile", profile_name)
            copied["source_artifact"] = str(path)
            rows.append(copied)
    return rows, source_summaries


def _select(rows: list[dict[str, object]], kernel: str) -> tuple[int, str]:
    providers = _providers(kernel)
    profile = _profile(kernel)
    candidates: dict[int, dict[str, dict[str, object]]] = {}
    for row in rows:
        if row.get("kernel") != kernel or row.get("profile") != profile:
            continue
        candidates.setdefault(int(row["size"]), {})[str(row["provider"])] = row
    common = {
        size: by_provider
        for size, by_provider in candidates.items()
        if all(provider in by_provider and _accepted(by_provider[provider]) for provider in providers)
    }
    if FORMAL_TOKENS in common:
        return FORMAL_TOKENS, "exact_project_formal_shape"
    if not common:
        raise ValueError(f"no correct common shape for {kernel}: {sorted(candidates)}")
    # Predeclared fallback: maximize optimized full-step logical throughput
    # among shapes completed by every comparison backend.
    selected = max(
        common,
        key=lambda size: size / float(common[size]["triton"]["full_p50_ms"]),
    )
    return selected, "common_backend_full_step_throughput_optimum"


def _headline_rows(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output = []
    selections = []
    for kernel in HEADLINE_KERNELS:
        selected_size, rule = _select(rows, kernel)
        profile = _profile(kernel)
        selected = [
            row
            for row in rows
            if row.get("kernel") == kernel
            and row.get("profile") == profile
            and int(row["size"]) == selected_size
            and row.get("provider") in _providers(kernel)
        ]
        by_provider = {str(row["provider"]): row for row in selected}
        torch_row = by_provider["torch"]
        selections.append(
            {
                "kernel": kernel,
                "profile": profile,
                "selected_size": selected_size,
                "selection_rule": rule,
                "formal_size": FORMAL_TOKENS,
            }
        )
        for provider in _providers(kernel):
            row = by_provider[provider]
            flattened = {
                "kernel": kernel,
                "profile": profile,
                "provider": provider,
                "selected_size": selected_size,
                "selection_rule": rule,
                "dtype": row.get("dtype", "torch.bfloat16"),
                "fwd_correct": row["fwd_correct"],
                "bwd_correct": row["bwd_correct"],
                "fwd_p50_ms": row["fwd_p50_ms"],
                "fwd_p95_ms": row["fwd_p95_ms"],
                "bwd_p50_ms": row["bwd_p50_ms"],
                "bwd_p95_ms": row["bwd_p95_ms"],
                "full_p50_ms": row["full_p50_ms"],
                "full_p95_ms": row["full_p95_ms"],
                "fwd_peak_allocated_mb": row["fwd_peak_mem_mb"],
                "bwd_peak_allocated_mb": row["bwd_peak_mem_mb"],
                "full_peak_allocated_mb": row["full_peak_mem_mb"],
                "full_speedup_vs_torch": (
                    1.0
                    if provider == "torch"
                    else float(torch_row["full_p50_ms"]) / float(row["full_p50_ms"])
                ),
                "full_latency_reduction_percent": (
                    0.0
                    if provider == "torch"
                    else 100
                    * (
                        1
                        - float(row["full_p50_ms"])
                        / float(torch_row["full_p50_ms"])
                    )
                ),
                "full_peak_allocated_reduction_percent": (
                    0.0
                    if provider == "torch"
                    else 100
                    * (
                        1
                        - float(row["full_peak_mem_mb"])
                        / max(float(torch_row["full_peak_mem_mb"]), 1e-12)
                    )
                ),
                "native_cuda": kernel == "attention" and provider == "cuda",
                "source_artifact": row["source_artifact"],
            }
            output.append(flattened)
    return output, selections


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator

    optimized = [row for row in rows if row["provider"] != "torch"]
    display_names = {
        "rmsnorm": "RMSNorm",
        "rope": "RoPE",
        "swiglu": "SwiGLU",
        "cross_entropy": "Cross Entropy†",
        "fused_linear_cross_entropy": "Fused Linear CE†",
        "attention": "FlashAttention",
        "router_postprocess": "Router postprocess",
        "fused_moe": "Fused MoE",
    }
    labels = [
        f"{display_names[str(row['kernel'])]}  ·  {str(row['provider']).title()}"
        for row in optimized
    ]
    speedups = [float(row["full_speedup_vs_torch"]) for row in optimized]
    memory = [float(row["full_peak_allocated_reduction_percent"]) for row in optimized]
    colors = ["#15808d" if row["provider"] == "triton" else "#7357cf" for row in optimized]
    y = list(range(len(labels)))
    figure, axes = plt.subplots(1, 2, figsize=(18, 10), sharey=True)
    figure.patch.set_facecolor("#f7f9fc")

    for axis in axes:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, labelsize=13, colors="#1e293b")
        axis.tick_params(axis="x", labelsize=11, colors="#64748b")
        axis.grid(axis="x", color="#e8edf4", linewidth=1, zorder=0)
        axis.set_axisbelow(True)
        for index in range(len(labels)):
            if index % 2 == 0:
                axis.axhspan(index - 0.48, index + 0.48, color="#f8fafc", zorder=-1)

    axes[0].barh(y, speedups, height=0.62, color=colors, zorder=3)
    axes[0].axvline(1.0, color="#94a3b8", linewidth=1.5, linestyle="--", zorder=2)
    axes[0].set_xlim(0, 7.25)
    axes[0].xaxis.set_major_locator(MultipleLocator(1))
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_title("Full-step speedup", loc="left", fontsize=20, fontweight="bold", pad=26, color="#0f172a")
    axes[0].text(
        0,
        1.02,
        "Higher is better  ·  dashed line = PyTorch reference",
        transform=axes[0].transAxes,
        fontsize=12.5,
        color="#64748b",
    )
    axes[0].set_xlabel("Relative speed (×)", fontsize=12, color="#475569", labelpad=12)
    for index, value in enumerate(speedups):
        axes[0].text(
            value + 0.10,
            index,
            f"{value:.2f}×",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="#0f172a",
        )

    axes[1].barh(y, memory, height=0.62, color=colors, zorder=3)
    axes[1].axvline(0.0, color="#94a3b8", linewidth=1.5, zorder=2)
    axes[1].set_xlim(0, 108)
    axes[1].xaxis.set_major_locator(MultipleLocator(20))
    # The two panels share the same rows. Repeating long kernel names on the
    # right crowds the speed labels in the center gutter, so label rows once.
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].set_title("Peak allocated memory", loc="left", fontsize=20, fontweight="bold", pad=26, color="#0f172a")
    axes[1].text(
        0,
        1.02,
        "Reduction relative to the matched PyTorch path",
        transform=axes[1].transAxes,
        fontsize=12.5,
        color="#64748b",
    )
    axes[1].set_xlabel("Lower peak allocation (%)", fontsize=12, color="#475569", labelpad=12)
    for index, value in enumerate(memory):
        axes[1].text(
            min(value + 2.0, 101),
            index,
            f"{value:.1f}%",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="#0f172a",
        )

    figure.suptitle(
        "GPU kernel benchmark  ·  MiniTrainSys",
        x=0.07,
        y=0.99,
        ha="left",
        fontsize=27,
        fontweight="bold",
        color="#0f172a",
    )
    figure.text(
        0.07,
        0.91,
        "RTX 4090 24 GB  ·  BF16  ·  293M MoE workload  ·  forward + backward",
        ha="left",
        fontsize=13,
        color="#475569",
    )
    figure.legend(
        handles=[Patch(facecolor="#15808d", label="Triton"), Patch(facecolor="#7357cf", label="Native CUDA")],
        loc="upper right",
        bbox_to_anchor=(0.945, 0.982),
        frameon=False,
        ncol=2,
        fontsize=11,
    )
    figure.text(
        0.07,
        0.025,
        "† Cross Entropy and Fused Linear CE use the largest common runnable shape: 8,192 tokens. "
        "All other rows use 57,344 tokens. Values are P50 measurements after correctness gates.",
        ha="left",
        fontsize=10.5,
        color="#64748b",
    )
    figure.subplots_adjust(left=0.23, right=0.97, top=0.79, bottom=0.10, wspace=0.16)
    figure.savefig(output, dpi=220, facecolor=figure.get_facecolor())
    plt.close(figure)


def _markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Kernel | Backend | Shape tokens | Full P50 ms | Speedup | Peak alloc MB | Memory Δ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["provider"] == "torch":
            continue
        lines.append(
            f"| {row['kernel']} | {row['provider']} | {row['selected_size']:,} | "
            f"{float(row['full_p50_ms']):.3f} | "
            f"{float(row['full_speedup_vs_torch']):.2f}× | "
            f"{float(row['full_peak_allocated_mb']):.1f} | "
            f"{float(row['full_peak_allocated_reduction_percent']):+.1f}% |"
        )
    return "\n".join(lines)


def summarize(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, source_summaries = _load_rows(args)
    headline, selections = _headline_rows(rows)
    capacity = _read(Path(args.capacity))["rows"][0]
    capacity["source_artifact"] = str(Path(args.capacity))
    quality = all(item["quality_gate_passed"] for item in source_summaries)
    result = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "quality_gate_passed": quality,
        "gpu": "NVIDIA GeForce RTX 4090 24 GB",
        "dtype": "BF16",
        "project_formal_profile": {
            "tokens_per_rank_layer": FORMAL_TOKENS,
            "local_batch": 112,
            "sequence_length": 512,
            "hidden": 768,
            "intermediate": 1024,
            "heads": 12,
            "head_dim": 64,
            "vocab": 50257,
            "experts": 8,
            "top_k": 2,
        },
        "source_quality": source_summaries,
        "selections": selections,
        "headline_rows": headline,
        "fused_loss_formal_capacity": capacity,
        "native_cuda_kernels": ["attention"],
        "cuda_facade_fallback_note": (
            "CUDA has a native implementation only for attention; other CUDA facade "
            "operators fall back and are not reported as native CUDA kernels."
        ),
    }
    _write_csv(output / "kernel_benchmark_summary.csv", headline)
    _atomic_json(output / "kernel_benchmark_summary.json", result)
    _plot(headline, output / "kernel_benchmark_overview.png")

    capacity_row = capacity
    readme = f"""# Industrial kernel benchmark summary

## Question or hypothesis

Do MiniTrain's Triton kernels and native CUDA FlashAttention improve full-step latency and peak
allocated memory over shape-matched Torch implementations on one RTX 4090?

## Exact compared conditions

All paired rows use the same shape, BF16 dtype, GPU, 50 ms warmup, 200 ms measurement window, and
fresh CUDA process per shape. The primary profile is the formal 293.49M SynBioS MoE per-rank
workload: local batch 112, sequence 512, 57,344 tokens, H=768, I=1,024, D=64, vocab=50,257,
E=8, K=2. Mixtral-class scans are retained in raw artifacts as an appendix.

## Run and workload identity

The lifecycle and commands are recorded in the retained run records entries “RTX 4090 industrial per-kernel
benchmark”, “MoE relative-gradient gate and fused-loss capacity follow-up”, and “Industrial
kernel scan boundary extensions”. This is an operator benchmark and has no train/validation
dataset split.

## Primary metrics

{_markdown_table(headline)}

Explicit CrossEntropy and the Torch fused-loss reference cannot complete the full 57,344-token
vocab=50,257 comparison within 24 GB, so their paired headline uses the predeclared largest
common-backend throughput-optimal shape. Separately, Triton fused loss completed the formal
57,344-token capacity case after correctness passed at {int(capacity_row['validation_tokens']):,}
tokens: full P50 **{float(capacity_row['full_p50_ms']):.2f} ms**, peak allocated delta
**{float(capacity_row['full_peak_mem_mb']):.1f} MiB**.

## Supporting artifacts

- Machine-readable summary: `kernel_benchmark_summary.json`
- Flat table: `kernel_benchmark_summary.csv`
- Overview: `kernel_benchmark_overview.png`
- Dense raw summary: `{args.dense}`
- MoE quality-gated summary: `{args.moe}`
- Boundary extensions: `{args.extensions}`
- Fused-loss capacity evidence: `{args.capacity}`

## Interpretation

The table reports every optimized result, including slowdowns. Native CUDA is claimed only for
FlashAttention; RMSNorm, RoPE, SwiGLU, both losses, Router, and fused MoE are Triton. A speed result
is not a correctness proof; all selected paired rows passed forward and backward checks.

## Limitations or threats to validity

Peak memory is allocator **allocated delta**, not whole-process reserved VRAM. Fused-loss formal
capacity is not a same-shape speedup because the Torch reference OOMed. BF16 fused-MoE gradients
use both retained elementwise statistics and a scale-aware gate (relative L2 <=1%, cosine
>=0.9999). End-to-end FSDP results are a separate stage.

## Next decision/action

Use these operator results as the kernel-level résumé table. Do not infer whole-training speedup
from them; run the formal four-GPU fixed-workload and fixed-92%-reserved-VRAM comparisons.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--dense",
        default="artifacts/operator_benchmark/rtx4090_24gb/dense/summary.json",
    )
    result.add_argument(
        "--moe",
        default=(
            "artifacts/operator_benchmark/rtx4090_24gb/"
            "moe_r2_relative_gate/summary.json"
        ),
    )
    result.add_argument(
        "--extensions",
        default="artifacts/operator_benchmark/rtx4090_24gb/extensions/raw",
    )
    result.add_argument(
        "--capacity",
        default=(
            "artifacts/operator_benchmark/rtx4090_24gb/dense_capacity/"
            "fused_linear_cross_entropy_project_formal_57344_triton.json"
        ),
    )
    result.add_argument(
        "--output",
        default="artifacts/operator_benchmark/resume_summary",
    )
    return result


if __name__ == "__main__":
    summarize(parser().parse_args())
