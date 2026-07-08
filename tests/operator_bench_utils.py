import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable

import torch


TensorMap = dict[str, torch.Tensor]
ForwardFn = Callable[[str, TensorMap], Any]
MakeCaseFn = Callable[[int], "BenchCase"]
METRIC_NAMES = (
    "fwd_p50_ms",
    "fwd_p95_ms",
    "fwd_peak_mem_mb",
    "fwd_speedup",
    "bwd_p50_ms",
    "bwd_p95_ms",
    "bwd_peak_mem_mb",
    "bwd_speedup",
)


@dataclass
class BenchCase:
    tensors: TensorMap
    grad_names: tuple[str, ...] = ()


def release_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def free_case(case: BenchCase) -> None:
    case.tensors.clear()
    release_cache()


def flatten_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from flatten_tensors(item)
        return
    raise TypeError(f"Unsupported output type: {type(value)!r}")


def clone_case(case: BenchCase, *, requires_grad: bool) -> BenchCase:
    tensors = {}
    for name, tensor in case.tensors.items():
        clone = tensor.detach().clone()
        if requires_grad and name in case.grad_names:
            clone.requires_grad_(True)
        tensors[name] = clone
    return BenchCase(tensors=tensors, grad_names=case.grad_names)


def set_requires_grad(case: BenchCase) -> None:
    for name in case.grad_names:
        case.tensors[name].requires_grad_(True)


def zero_grads(case: BenchCase) -> None:
    for name in case.grad_names:
        tensor = case.tensors[name]
        tensor.grad = None


def close_stats(actual: Any, expected: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    actual_tensors = tuple(flatten_tensors(actual))
    expected_tensors = tuple(flatten_tensors(expected))
    if len(actual_tensors) != len(expected_tensors):
        return {"correct": False, "max_abs": math.inf, "max_rel": math.inf}

    correct = True
    max_abs = 0.0
    max_rel = 0.0
    for actual_tensor, expected_tensor in zip(actual_tensors, expected_tensors):
        actual_f = actual_tensor.detach().float()
        expected_f = expected_tensor.detach().float()
        abs_err = (actual_f - expected_f).abs()
        rel_err = abs_err / expected_f.abs().clamp_min(1e-8)
        max_abs = max(max_abs, float(abs_err.max().item()))
        max_rel = max(max_rel, float(rel_err.max().item()))
        try:
            torch.testing.assert_close(actual_tensor, expected_tensor, atol=atol, rtol=rtol)
        except AssertionError:
            correct = False

    return {"correct": correct, "max_abs": max_abs, "max_rel": max_rel}


def output_loss(output: Any) -> torch.Tensor:
    losses = [tensor.float().sum() for tensor in flatten_tensors(output)]
    return sum(losses)


def gradient_output(case: BenchCase, provider: str, forward: ForwardFn) -> Any:
    zero_grads(case)
    output = forward(provider, case.tensors)
    loss = output_loss(output)
    loss.backward()
    del loss
    return output


def make_backward_graph(
    case: BenchCase,
    provider: str,
    forward: ForwardFn,
) -> tuple[Any, torch.Tensor]:
    zero_grads(case)
    output = forward(provider, case.tensors)
    loss = output_loss(output)
    return output, loss


def gradient_stats(
    actual_case: BenchCase,
    expected_case: BenchCase,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    actual_grads = []
    expected_grads = []
    for name in actual_case.grad_names:
        actual_grads.append(actual_case.tensors[name].grad)
        expected_grads.append(expected_case.tensors[name].grad)
    return close_stats(actual_grads, expected_grads, atol=atol, rtol=rtol)


def correctness_stats(
    make_case: MakeCaseFn,
    size: int,
    provider: str,
    forward: ForwardFn,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    actual_case = make_case(size)
    expected_case = clone_case(actual_case, requires_grad=True)
    set_requires_grad(actual_case)

    try:
        actual_output = gradient_output(actual_case, provider, forward)
        expected_output = gradient_output(expected_case, "torch", forward)
        fwd = close_stats(actual_output, expected_output, atol=atol, rtol=rtol)
        bwd = gradient_stats(actual_case, expected_case, atol=atol, rtol=rtol)
        del actual_output, expected_output
        return {
            "fwd_correct": fwd["correct"],
            "fwd_max_abs": fwd["max_abs"],
            "fwd_max_rel": fwd["max_rel"],
            "bwd_correct": bwd["correct"],
            "bwd_max_abs": bwd["max_abs"],
            "bwd_max_rel": bwd["max_rel"],
        }
    finally:
        free_case(actual_case)
        free_case(expected_case)


def latency_ms(
    fn: Callable[[], None],
    *,
    warmup_ms: int,
    rep_ms: int,
    fallback_warmup_iters: int = 5,
    fallback_iters: int = 20,
) -> tuple[float, float, str]:
    try:
        import triton

        p50, p95 = triton.testing.do_bench(
            fn,
            warmup=warmup_ms,
            rep=rep_ms,
            quantiles=[0.5, 0.95],
        )
        return float(p50), float(p95), "triton.do_bench"
    except ImportError:
        pass

    for _ in range(fallback_warmup_iters):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(fallback_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    values = torch.tensor(samples, dtype=torch.float64)
    p50, p95 = torch.quantile(values, torch.tensor([0.5, 0.95], dtype=torch.float64)).tolist()
    return float(p50), float(p95), "cuda_events"


def peak_memory_mb(fn: Callable[[], None]) -> float:
    release_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    release_cache()
    return max(0.0, (peak - baseline) / 2**20)


def backward_latency_ms(
    case: BenchCase,
    provider: str,
    forward: ForwardFn,
    *,
    warmup_iters: int = 5,
    iters: int = 20,
) -> tuple[float, float, str]:
    for _ in range(warmup_iters):
        output, loss = make_backward_graph(case, provider, forward)
        torch.cuda.synchronize()
        loss.backward()
        torch.cuda.synchronize()
        del loss, output

    samples = []
    for _ in range(iters):
        output, loss = make_backward_graph(case, provider, forward)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
        del loss, output

    zero_grads(case)
    release_cache()
    values = torch.tensor(samples, dtype=torch.float64)
    p50, p95 = torch.quantile(values, torch.tensor([0.5, 0.95], dtype=torch.float64)).tolist()
    return float(p50), float(p95), "cuda_events_backward_only"


def backward_peak_memory_mb(case: BenchCase, provider: str, forward: ForwardFn) -> float:
    release_cache()
    output, loss = make_backward_graph(case, provider, forward)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del loss, output
    zero_grads(case)
    release_cache()
    return max(0.0, (peak - baseline) / 2**20)


def benchmark_step(
    make_case: MakeCaseFn,
    size: int,
    provider: str,
    forward: ForwardFn,
    *,
    mode: str,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, Any]:
    case = make_case(size)
    if mode == "bwd":
        seed_case = case
        case = clone_case(seed_case, requires_grad=True)
        free_case(seed_case)
    try:
        if mode == "fwd":
            def step() -> None:
                with torch.no_grad():
                    forward(provider, case.tensors)
            p50, p95, timer = latency_ms(step, warmup_ms=warmup_ms, rep_ms=rep_ms)
            zero_grads(case)
            release_cache()
            memory = peak_memory_mb(step)
        elif mode == "bwd":
            p50, p95, timer = backward_latency_ms(case, provider, forward)
            memory = backward_peak_memory_mb(case, provider, forward)
        else:
            raise ValueError(f"Unknown benchmark mode: {mode}")

        zero_grads(case)
        return {
            f"{mode}_p50_ms": p50,
            f"{mode}_p95_ms": p95,
            f"{mode}_peak_mem_mb": memory,
            "timer": timer,
        }
    finally:
        free_case(case)


def bench_provider(
    *,
    kernel: str,
    provider: str,
    size: int,
    size_label: str,
    make_case: MakeCaseFn,
    forward: ForwardFn,
    torch_fwd_p50_ms: float | None,
    torch_bwd_p50_ms: float | None,
    atol: float,
    rtol: float,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, Any]:
    row = {
        "kernel": kernel,
        "provider": provider,
        "size": size,
        "size_label": size_label,
        "status": "ok",
    }
    try:
        row.update(correctness_stats(make_case, size, provider, forward, atol=atol, rtol=rtol))
        row.update(
            benchmark_step(
                make_case,
                size,
                provider,
                forward,
                mode="fwd",
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
            )
        )
        row.update(
            benchmark_step(
                make_case,
                size,
                provider,
                forward,
                mode="bwd",
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
            )
        )
        row["fwd_speedup"] = (
            1.0 if torch_fwd_p50_ms in (None, 0.0) else torch_fwd_p50_ms / row["fwd_p50_ms"]
        )
        row["bwd_speedup"] = (
            1.0 if torch_bwd_p50_ms in (None, 0.0) else torch_bwd_p50_ms / row["bwd_p50_ms"]
        )
    except Exception as exc:
        row.update(
            {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "fwd_correct": False,
                "bwd_correct": False,
            }
        )
        row.update({name: math.nan for name in METRIC_NAMES})
    finally:
        release_cache()
    return row


def bench_sweep(
    *,
    kernel: str,
    providers: Iterable[str],
    sizes: Iterable[int],
    size_label: str,
    make_case: MakeCaseFn,
    forward: ForwardFn,
    warmup_ms: int = 25,
    rep_ms: int = 100,
    atol: float = 2e-2,
    rtol: float = 2e-2,
) -> list[dict[str, Any]]:
    if isinstance(providers, str):
        providers = (providers,)

    rows = []
    for size in sizes:
        torch_fwd_p50 = None
        torch_bwd_p50 = None
        for provider in providers:
            row = bench_provider(
                kernel=kernel,
                provider=provider,
                size=size,
                size_label=size_label,
                make_case=make_case,
                forward=forward,
                torch_fwd_p50_ms=torch_fwd_p50,
                torch_bwd_p50_ms=torch_bwd_p50,
                atol=atol,
                rtol=rtol,
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
            )
            if provider == "torch" and row["status"] == "ok":
                torch_fwd_p50 = row["fwd_p50_ms"]
                torch_bwd_p50 = row["bwd_p50_ms"]
                row["fwd_speedup"] = 1.0
                row["bwd_speedup"] = 1.0
            rows.append(row)
        release_cache()
    return rows


def to_dataframe(rows: list[dict[str, Any]]) -> Any:
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:
        return rows


def to_summary_dataframe(rows: list[dict[str, Any]]) -> Any:
    columns = [
        "kernel",
        "provider",
        "size",
        "status",
        "fwd_correct",
        "bwd_correct",
        "fwd_max_abs",
        "bwd_max_abs",
        "fwd_p50_ms",
        "fwd_p95_ms",
        "fwd_peak_mem_mb",
        "fwd_speedup",
        "bwd_p50_ms",
        "bwd_p95_ms",
        "bwd_peak_mem_mb",
        "bwd_speedup",
        "error",
    ]
    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        present = [name for name in columns if name in frame.columns]
        return frame[present]
    except ImportError:
        return [{name: row.get(name) for name in columns if name in row} for row in rows]


YLABELS = {
    "fwd_p50_ms": "forward p50 latency (ms)",
    "fwd_p95_ms": "forward p95 latency (ms)",
    "fwd_peak_mem_mb": "forward peak memory delta (MB)",
    "fwd_speedup": "forward speedup vs torch",
    "bwd_p50_ms": "backward-only p50 latency (ms)",
    "bwd_p95_ms": "backward-only p95 latency (ms)",
    "bwd_peak_mem_mb": "backward-only peak memory delta (MB)",
    "bwd_speedup": "backward-only speedup vs torch",
}


def _plot_metric(ax: Any, rows: list[dict[str, Any]], metric: str) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok" and row.get(metric) is not None]
    if not ok_rows:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    providers = list(dict.fromkeys(row["provider"] for row in ok_rows))
    sizes = sorted(set(int(row["size"]) for row in ok_rows))
    colors = {"torch": "#3B6EA8", "triton": "#E87722", "cuda": "#2E8B57"}

    for provider in providers:
        ys = []
        for size in sizes:
            match = next(
                (
                    row
                    for row in ok_rows
                    if row["provider"] == provider and int(row["size"]) == size
                ),
                None,
            )
            ys.append(float("nan") if match is None else match[metric])
        ax.plot(
            sizes,
            ys,
            marker="o",
            linewidth=1.8,
            label=provider,
            color=colors.get(provider),
        )

    ax.set_title(YLABELS[metric], fontsize=10, weight="bold")
    ax.set_xscale("log", base=2)
    if metric.endswith(("_ms", "_mb")):
        ax.set_yscale("log")
    if metric.endswith("_speedup"):
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.grid(axis="both", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_kernel_grid(
    rows: list[dict[str, Any]],
    *,
    metrics: Iterable[str] = METRIC_NAMES,
    save_path: Path | None = None,
) -> Any:
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    metrics = tuple(metrics)
    kernel = next((row["kernel"] for row in rows if "kernel" in row), "kernel")
    size_label = next((row["size_label"] for row in rows if "size_label" in row), "size")
    cols = 4
    rows_count = math.ceil(len(metrics) / cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(4.2 * cols, 3.4 * rows_count), dpi=140)
    fig.patch.set_facecolor("#FBFBFD")
    flat_axes = list(axes.ravel() if hasattr(axes, "ravel") else [axes])

    for ax, metric in zip(flat_axes, metrics):
        ax.set_facecolor("#FBFBFD")
        _plot_metric(ax, rows, metric)
        ax.set_xlabel(size_label, fontsize=9)

    for ax in flat_axes[len(metrics) :]:
        ax.set_axis_off()

    handles, labels = [], []
    for ax in flat_axes[: len(metrics)]:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    fig.suptitle(f"{kernel} benchmark summary", fontsize=14, weight="bold", y=0.99)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=max(1, len(labels)),
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_kernel(rows: list[dict[str, Any]], *, metric: str, save_path: Path | None = None) -> Any:
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    ok_rows = [row for row in rows if row["status"] == "ok" and row.get(metric) is not None]
    if not ok_rows:
        print(f"skip plot: no successful benchmark rows for {metric}")
        return None

    kernel = ok_rows[0]["kernel"]
    size_label = ok_rows[0]["size_label"]
    providers = list(dict.fromkeys(row["provider"] for row in ok_rows))
    sizes = sorted(set(int(row["size"]) for row in ok_rows))
    colors = {"torch": "#3B6EA8", "triton": "#E87722", "cuda": "#2E8B57"}

    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    fig.patch.set_facecolor("#FBFBFD")
    ax.set_facecolor("#FBFBFD")
    for provider in providers:
        ys = []
        for size in sizes:
            match = next(
                (
                    row
                    for row in ok_rows
                    if row["provider"] == provider and int(row["size"]) == size
                ),
                None,
            )
            ys.append(float("nan") if match is None else match[metric])
        ax.plot(
            sizes,
            ys,
            marker="o",
            linewidth=2.0,
            label=provider,
            color=colors.get(provider),
        )

    ax.set_title(f"{kernel}: {YLABELS[metric]}", fontsize=14, weight="bold")
    ax.set_xlabel(size_label)
    ax.set_ylabel(YLABELS[metric])
    ax.set_xscale("log", base=2)
    if metric.endswith(("_ms", "_mb")):
        ax.set_yscale("log")
    if metric.endswith("_speedup"):
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.grid(axis="both", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
