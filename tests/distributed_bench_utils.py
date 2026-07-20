"""High-level orchestration and presentation for the distributed benchmark notebook.

The notebook deliberately calls only :class:`DistributedBenchmark`.  Command
construction, environment validation, subprocess execution, report loading,
repeat aggregation, quality gates, and display live here so the same workflow
can also be driven from a plain Python session on the target server.
"""

from __future__ import annotations

import csv
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SUPPORTED_STRATEGIES = ("single", "ddp", "fsdp")
SUPPORTED_WORLD_SIZES = (1, 4, 8)
SUPPORTED_TOPOLOGIES = {
    "single": (1,),
    "ddp": SUPPORTED_WORLD_SIZES,
    "fsdp": SUPPORTED_WORLD_SIZES,
}
DEFAULT_COLUMNS = (
    "strategy",
    "world_size",
    "local_batch_size",
    "global_batch_size",
    "repeats_successful",
    "step_time_ms_p50",
    "step_time_ms_p95",
    "throughput_tokens_per_sec",
    "weak_scaling_efficiency_percent",
    "peak_memory_allocated_mb",
    "system_peak_memory_allocated_mb",
    "memory_utilization_percent",
    "data_stall_percent",
    "workers_per_node",
)


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root from either the root or a notebook directory."""

    path = Path(start or Path.cwd()).resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if (candidate / "scripts" / "run_dist_bench.py").is_file() and (
            candidate / "pyproject.toml"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        f"could not find mini-train-sys above {path}; start Jupyter inside the repository"
    )


def load_report(path: str | Path) -> dict:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report.get("results"), list) or not isinstance(
        report.get("failures"), list
    ):
        raise ValueError(f"invalid distributed benchmark report: {path}")
    return report


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)


def aggregate_rows(report_or_path: dict | str | Path) -> list[dict]:
    """Average successful repeats into one row per strategy/world-size/batch case."""

    report = (
        load_report(report_or_path)
        if isinstance(report_or_path, (str, Path))
        else report_or_path
    )
    groups: dict[tuple[str, int, int], list[dict]] = {}
    for row in report["results"]:
        key = (row["strategy"], row["world_size"], row["local_batch_size"])
        groups.setdefault(key, []).append(row)

    aggregated = []
    for (strategy, world_size, local_batch), rows in sorted(groups.items()):
        item = dict(rows[0])
        numeric_keys = {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        for key in numeric_keys:
            item[key] = _mean(float(row[key]) for row in rows if key in row)
        item.update(
            {
                "strategy": strategy,
                "world_size": world_size,
                "local_batch_size": local_batch,
                "global_batch_size": local_batch * world_size,
                "repeats_successful": len(rows),
            }
        )
        aggregated.append(item)
    return aggregated


def compact_rows(
    report_or_path: dict | str | Path,
    columns: Sequence[str] = DEFAULT_COLUMNS,
) -> list[dict]:
    """Return display-focused, repeat-aggregated rows without raw provenance fields."""

    return [
        {
            key: round(value, 3) if isinstance(value := row.get(key), float) else value
            for key in columns
            if key in row
        }
        for row in aggregate_rows(report_or_path)
    ]


def quality_gates(report_or_path: dict | str | Path) -> dict[str, object]:
    """Apply weak-scaling, data-stall, memory-headroom, and repeat-completion gates."""

    report = (
        load_report(report_or_path)
        if isinstance(report_or_path, (str, Path))
        else report_or_path
    )
    rows = aggregate_rows(report)
    efficiencies = [
        row["weak_scaling_efficiency_percent"]
        for row in rows
        if row["world_size"] > 1 and "weak_scaling_efficiency_percent" in row
    ]
    expected_repeats = int(report.get("settings", {}).get("repeats", 1))
    settings = report.get("settings", {})
    requested_cases = settings.get("requested_cases")
    if requested_cases is None:
        requested_case_count = len(settings.get("strategies", [])) * len(
            settings.get("world_sizes", [])
        )
    else:
        requested_case_count = len(requested_cases)
    expected_cases = requested_case_count * expected_repeats
    return {
        "runs_succeeded": len(report["results"]),
        "runs_failed": len(report["failures"]),
        "all_requested_repeats_completed": (
            (not expected_cases or len(report["results"]) == expected_cases)
            and not report["failures"]
            and all(row["repeats_successful"] == expected_repeats for row in rows)
        ),
        "weak_scaling_efficiency_ge_80pct": not efficiencies or min(efficiencies) >= 80,
        "data_stall_le_5pct": all(row["data_stall_percent"] <= 5 for row in rows),
        "memory_headroom_ge_10pct": all(
            row["memory_utilization_percent"] <= 90 for row in rows
        ),
    }


def capacity_frontier(report_or_path: dict | str | Path) -> list[dict]:
    report = (
        load_report(report_or_path)
        if isinstance(report_or_path, (str, Path))
        else report_or_path
    )
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in aggregate_rows(report):
        groups.setdefault((row["strategy"], row["world_size"]), []).append(row)
    return [
        {
            "strategy": strategy,
            "world_size": world_size,
            "largest_successful_local_batch": max(
                row["local_batch_size"] for row in rows
            ),
            "largest_successful_global_batch": max(
                row["global_batch_size"] for row in rows
            ),
        }
        for (strategy, world_size), rows in sorted(groups.items())
    ]


def _table(rows: list[dict]):
    try:
        import pandas as pd
    except ImportError:
        return rows
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class BenchmarkReport:
    """A validated summary report with notebook-friendly presentation helpers."""

    path: Path

    @property
    def data(self) -> dict:
        return load_report(self.path)

    @property
    def suite(self) -> str:
        return str(self.data["suite"])

    def table(self):
        return _table(compact_rows(self.data))

    def failures(self):
        return _table(self.data["failures"])

    def gates(self) -> dict[str, object]:
        return quality_gates(self.data)

    def frontier(self):
        return _table(capacity_frontier(self.data))

    def figure(self):
        """Build the standard weak-scaling or capacity visualization."""

        import matplotlib.pyplot as plt

        rows = aggregate_rows(self.data)
        if self.suite == "weak":
            figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
            for strategy in sorted({row["strategy"] for row in rows}):
                selected = sorted(
                    (row for row in rows if row["strategy"] == strategy),
                    key=lambda row: row["world_size"],
                )
                worlds = [row["world_size"] for row in selected]
                axes[0, 0].plot(
                    worlds,
                    [row["throughput_tokens_per_sec"] for row in selected],
                    marker="o",
                    label=strategy.upper(),
                )
                efficiencies = [
                    row.get("weak_scaling_efficiency_percent", 100.0) for row in selected
                ]
                axes[0, 1].plot(
                    worlds, efficiencies, marker="o", label=strategy.upper()
                )
                axes[1, 0].plot(
                    worlds,
                    [row["step_time_ms_p50"] for row in selected],
                    marker="o",
                    label=strategy.upper(),
                )
                axes[1, 1].plot(
                    worlds,
                    [row["memory_utilization_percent"] for row in selected],
                    marker="o",
                    label=f"{strategy.upper()} memory",
                )
                axes[1, 1].plot(
                    worlds,
                    [row["data_stall_percent"] for row in selected],
                    marker="x",
                    linestyle="--",
                    label=f"{strategy.upper()} data stall",
                )
            axes[0, 0].set_title("Global throughput")
            axes[0, 0].set_ylabel("tokens/s")
            axes[0, 1].set_title("Weak-scaling efficiency")
            axes[0, 1].set_ylabel("percent")
            axes[0, 1].axhline(80, color="tab:red", linestyle="--", label="80% gate")
            axes[1, 0].set_title("Step latency P50")
            axes[1, 0].set_ylabel("ms")
            axes[1, 1].set_title("Memory utilization and data stall")
            axes[1, 1].set_ylabel("percent")
            axes[1, 1].axhline(90, color="tab:red", linestyle=":", label="90% memory")
            axes[1, 1].axhline(5, color="tab:orange", linestyle=":", label="5% stall")
            for axis in axes.flat:
                axis.set_xlabel("world size")
                axis.set_xticks(sorted({row["world_size"] for row in rows}))
                axis.grid(alpha=0.25)
                axis.legend()
            figure.suptitle("DDP/FSDP weak scaling", fontsize=15)
            return figure

        figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        frontier = capacity_frontier(self.data)
        labels = [f"{row['strategy'].upper()}\n{row['world_size']} GPU" for row in frontier]
        axes[0].bar(
            labels,
            [row["largest_successful_local_batch"] for row in frontier],
            color="tab:blue",
        )
        axes[0].set_title("Largest successful local batch")
        axes[0].set_ylabel("samples / GPU")
        for strategy, world_size in sorted(
            {(row["strategy"], row["world_size"]) for row in rows}
        ):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["strategy"] == strategy and row["world_size"] == world_size
                ),
                key=lambda row: row["local_batch_size"],
            )
            axes[1].plot(
                [row["local_batch_size"] for row in selected],
                [row["memory_utilization_percent"] for row in selected],
                marker="o",
                label=f"{strategy.upper()}-{world_size}GPU",
            )
        axes[1].axhline(90, color="tab:red", linestyle="--", label="90% headroom gate")
        axes[1].set_title("Successful cases: memory vs local batch")
        axes[1].set_xlabel("local batch")
        axes[1].set_ylabel("peak allocated memory (%)")
        axes[1].grid(alpha=0.25)
        handles, legend_labels = axes[1].get_legend_handles_labels()
        unique = dict(zip(legend_labels, handles))
        axes[1].legend(unique.values(), unique.keys(), fontsize=8)
        figure.suptitle("DDP/FSDP capacity frontier", fontsize=15)
        return figure

    def save_presentation(self, figure=None) -> dict[str, Path]:
        """Persist aggregate tables, gates/frontier, failures, and the standard figure."""

        target = self.path.parent / "presentation"
        target.mkdir(parents=True, exist_ok=True)
        rows = aggregate_rows(self.data)

        def write_csv(path: Path, records: list[dict]) -> None:
            if not records:
                path.write_text("", encoding="utf-8")
                return
            fields = sorted({key for record in records for key in record})
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)

        results_path = target / "results_aggregated.csv"
        failures_path = target / "failures.csv"
        metrics_path = target / (
            "quality_gates.json" if self.suite == "weak" else "capacity_frontier.json"
        )
        figure_path = target / f"{self.suite}_overview.png"
        readme_path = target / "README.md"
        write_csv(results_path, rows)
        write_csv(failures_path, self.data["failures"])
        metrics = self.gates() if self.suite == "weak" else capacity_frontier(self.data)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        chart = figure or self.figure()
        chart.savefig(figure_path, dpi=160, bbox_inches="tight")
        readme_path.write_text(
            "# Distributed benchmark presentation artifacts\n\n"
            f"Source summary: `{self.path}`\n\n"
            "- `results_aggregated.csv`: one row per strategy/world-size/local-batch; "
            "numeric repeat metrics are averaged.\n"
            "- `failures.csv`: OOM, timeout, exit code, log path, and error tail.\n"
            f"- `{metrics_path.name}`: "
            + (
                "weak-scaling efficiency, data-stall, memory, and completion gates.\n"
                if self.suite == "weak"
                else "largest successful local/global batch for each topology.\n"
            )
            + f"- `{figure_path.name}`: standard notebook visualization.\n",
            encoding="utf-8",
        )
        return {
            "results": results_path,
            "failures": failures_path,
            "metrics": metrics_path,
            "figure": figure_path,
            "readme": readme_path,
        }

    def show(self, *, save: bool = True) -> dict[str, Path] | None:
        """Display report tables/plots and persist presentation artifacts by default."""

        sections: list[tuple[str, object]] = [("Results", self.table())]
        if self.suite == "weak":
            sections.append(("Quality gates", self.gates()))
        else:
            sections.append(("Capacity frontier", self.frontier()))
        if self.data["failures"]:
            sections.append(("Failures / OOM boundary", self.failures()))
        figure = self.figure()
        saved = self.save_presentation(figure) if save else None
        try:
            from IPython.display import display
        except ImportError:
            for title, value in sections:
                print(f"\n{title}\n{value}")
            if saved:
                print("\nSaved artifacts")
                for name, path in saved.items():
                    print(f"{name}: {path}")
            return saved
        for title, value in sections:
            print(title)
            display(value)
        display(figure)
        if saved:
            print("Saved artifacts")
            display({name: str(path) for name, path in saved.items()})
        return saved


@dataclass(frozen=True)
class BenchmarkSettings:
    world_sizes: tuple[int, ...] = (1, 4, 8)
    strategies: tuple[str, ...] = SUPPORTED_STRATEGIES
    model_config: str = "configs/model_125m_moe.yaml"
    local_batch: int = 4
    capacity_batches: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    warmup_steps: int = 10
    measure_steps: int = 30
    repeats: int = 3
    capacity_warmup_steps: int = 3
    capacity_measure_steps: int = 5
    case_timeout_seconds: int = 1800

    def __post_init__(self) -> None:
        invalid_worlds = set(self.world_sizes) - set(SUPPORTED_WORLD_SIZES)
        invalid_strategies = set(self.strategies) - set(SUPPORTED_STRATEGIES)
        if invalid_worlds:
            raise ValueError(f"unsupported world sizes: {sorted(invalid_worlds)}")
        if invalid_strategies:
            raise ValueError(f"unsupported strategies: {sorted(invalid_strategies)}")
        positive = (
            self.local_batch,
            *self.capacity_batches,
            self.warmup_steps,
            self.measure_steps,
            self.repeats,
            self.capacity_warmup_steps,
            self.capacity_measure_steps,
            self.case_timeout_seconds,
        )
        if not self.world_sizes or not self.strategies or any(value <= 0 for value in positive):
            raise ValueError("benchmark dimensions and step counts must be positive and non-empty")


class DistributedBenchmark:
    """Server-ready facade used by the distributed benchmark notebook."""

    def __init__(
        self,
        *,
        output: str | Path = "artifacts/distributed_benchmark",
        root: str | Path | None = None,
        settings: BenchmarkSettings | None = None,
        python: str | Path = sys.executable,
    ) -> None:
        self.root = find_project_root(root)
        output_path = Path(output)
        self.output = (
            output_path.resolve()
            if output_path.is_absolute()
            else (self.root / output_path).resolve()
        )
        self.settings = settings or BenchmarkSettings()
        # Keep a virtualenv interpreter path intact. ``Path.resolve()`` follows the
        # venv's ``bin/python`` symlink to the system interpreter, which drops the
        # virtualenv context when benchmark subprocesses are launched.
        self.python = str(Path(python).absolute())
        self.cli = self.root / "scripts" / "run_dist_bench.py"

    def preflight(self, *, require_cuda: bool = True) -> dict[str, object]:
        """Validate files, Python environment, CUDA visibility, and requested topology."""

        import torch

        errors: list[str] = []
        warnings: list[str] = []
        required_files = [self.cli, self.root / self.settings.model_config]
        required_files.extend(
            self.root
            / f"configs/server/rtx4090_24gb/runs/{strategy}_{world_size}gpu.yaml"
            for strategy in self.settings.strategies
            for world_size in self.settings.world_sizes
            if world_size in SUPPORTED_TOPOLOGIES[strategy]
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            errors.append("missing files: " + ", ".join(missing))
        if not Path(self.python).is_file():
            errors.append(f"Python executable does not exist: {self.python}")
        if not torch.distributed.is_available():
            errors.append("torch.distributed is unavailable")
        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        requested = max(self.settings.world_sizes)
        if require_cuda and not torch.cuda.is_available():
            errors.append("CUDA is unavailable in the notebook Python environment")
        elif require_cuda and cuda_count < requested:
            errors.append(f"requested {requested} GPUs but only {cuda_count} are visible")
        if shutil.which("nvidia-smi") is None:
            (errors if require_cuda else warnings).append("nvidia-smi is not on PATH")
        report = {
            "ok": not errors,
            "root": str(self.root),
            "python": self.python,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "visible_gpus": cuda_count,
            "requested_world_sizes": self.settings.world_sizes,
            "strategies": self.settings.strategies,
            "output": str(self.output),
            "errors": errors,
            "warnings": warnings,
        }
        if errors:
            raise RuntimeError("benchmark preflight failed:\n- " + "\n- ".join(errors))
        return report

    def inventory(self) -> dict:
        inventory_dir = self.output / "inventory"
        self._execute([self.python, str(self.cli), "inventory", "--output", str(inventory_dir)])
        return json.loads((inventory_dir / "inventory.json").read_text(encoding="utf-8"))

    def command(
        self,
        suite: str,
        *,
        strategies: Sequence[str] | None = None,
        rerun_existing: bool = False,
    ) -> list[str]:
        if suite not in {"weak", "capacity"}:
            raise ValueError("suite must be 'weak' or 'capacity'")
        chosen = tuple(strategies or self.settings.strategies)
        invalid = set(chosen) - set(self.settings.strategies)
        if invalid:
            raise ValueError(f"strategies are outside this session: {sorted(invalid)}")
        output_dir = self.output / suite
        args = [
            self.python,
            str(self.cli),
            "run",
            "--suite",
            suite,
            "--strategies",
            *chosen,
            "--world-sizes",
            *(str(value) for value in self.settings.world_sizes),
            "--model-config",
            self.settings.model_config,
            "--output",
            str(output_dir),
            "--case-timeout-seconds",
            str(self.settings.case_timeout_seconds),
        ]
        if suite == "weak":
            args.extend(
                [
                    "--local-batch",
                    str(self.settings.local_batch),
                    "--warmup-steps",
                    str(self.settings.warmup_steps),
                    "--measure-steps",
                    str(self.settings.measure_steps),
                    "--repeats",
                    str(self.settings.repeats),
                ]
            )
        else:
            args.extend(
                [
                    "--batch-sizes",
                    *(str(value) for value in self.settings.capacity_batches),
                    "--warmup-steps",
                    str(self.settings.capacity_warmup_steps),
                    "--measure-steps",
                    str(self.settings.capacity_measure_steps),
                    "--repeats",
                    "1",
                ]
            )
        if rerun_existing:
            args.append("--rerun-existing")
        return args

    def run_weak(
        self,
        *,
        strategies: Sequence[str] | None = None,
        rerun_existing: bool = False,
    ) -> BenchmarkReport:
        if 1 not in self.settings.world_sizes:
            raise ValueError("weak scaling requires world_size=1 as its efficiency baseline")
        return self._run("weak", strategies=strategies, rerun_existing=rerun_existing)

    def run_capacity(
        self,
        *,
        strategies: Sequence[str] | None = None,
        rerun_existing: bool = False,
    ) -> BenchmarkReport:
        return self._run("capacity", strategies=strategies, rerun_existing=rerun_existing)

    def load(self, suite: str) -> BenchmarkReport:
        path = self.output / suite / f"{suite}_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"benchmark summary does not exist: {path}")
        load_report(path)
        return BenchmarkReport(path)

    def _run(
        self,
        suite: str,
        *,
        strategies: Sequence[str] | None,
        rerun_existing: bool,
    ) -> BenchmarkReport:
        command = self.command(
            suite, strategies=strategies, rerun_existing=rerun_existing
        )
        self._execute(command)
        return self.load(suite)

    def _execute(self, command: Sequence[str]) -> None:
        print("RUN", shlex.join(str(part) for part in command), flush=True)
        completed = subprocess.run([str(part) for part in command], cwd=self.root, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"benchmark command failed with exit code {completed.returncode}: "
                + shlex.join(str(part) for part in command)
            )
