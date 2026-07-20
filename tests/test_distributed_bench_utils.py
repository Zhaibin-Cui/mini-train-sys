from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import distributed_bench_utils as bench_utils


def weak_report() -> dict:
    rows = []
    for strategy in ("ddp", "fsdp"):
        for world_size, throughput in ((1, 1000.0), (4, 3600.0)):
            for repeat in range(2):
                rows.append(
                    {
                        "strategy": strategy,
                        "world_size": world_size,
                        "local_batch_size": 4,
                        "global_batch_size": 4 * world_size,
                        "repeat": repeat,
                        "step_time_ms_p50": 100.0 + repeat,
                        "step_time_ms_p95": 110.0 + repeat,
                        "throughput_tokens_per_sec": throughput,
                        "weak_scaling_efficiency_percent": (
                            100.0 if world_size == 1 else 90.0
                        ),
                        "peak_memory_allocated_mb": 1000.0,
                        "system_peak_memory_allocated_mb": 1000.0 * world_size,
                        "memory_utilization_percent": 50.0,
                        "data_stall_percent": 2.0,
                        "workers_per_node": 4 * world_size,
                    }
                )
    return {
        "schema_version": 2,
        "suite": "weak",
        "settings": {"repeats": 2},
        "results": rows,
        "failures": [],
    }


def capacity_report() -> dict:
    rows = []
    for strategy in ("ddp", "fsdp"):
        for world_size in (1, 4):
            for batch_size in (1, 2, 4):
                rows.append(
                    {
                        "strategy": strategy,
                        "world_size": world_size,
                        "local_batch_size": batch_size,
                        "global_batch_size": batch_size * world_size,
                        "repeat": 0,
                        "step_time_ms_p50": 100.0,
                        "step_time_ms_p95": 110.0,
                        "throughput_tokens_per_sec": 1000.0,
                        "peak_memory_allocated_mb": 1000.0 * batch_size,
                        "system_peak_memory_allocated_mb": 1000.0 * batch_size * world_size,
                        "memory_utilization_percent": 15.0 * batch_size,
                        "data_stall_percent": 2.0,
                        "workers_per_node": 4 * world_size,
                    }
                )
    return {
        "schema_version": 2,
        "suite": "capacity",
        "settings": {"repeats": 1},
        "results": rows,
        "failures": [
            {
                "strategy": "ddp",
                "world_size": 1,
                "batch_size": 8,
                "repeat": 0,
                "oom": True,
                "timed_out": False,
                "returncode": 1,
            }
        ],
    }


def test_repeat_aggregation_and_quality_gates():
    report = weak_report()
    rows = bench_utils.aggregate_rows(report)

    assert len(rows) == 4
    assert all(row["repeats_successful"] == 2 for row in rows)
    assert rows[0]["step_time_ms_p50"] == pytest.approx(100.5)
    assert bench_utils.quality_gates(report) == {
        "runs_succeeded": 8,
        "runs_failed": 0,
        "all_requested_repeats_completed": True,
        "weak_scaling_efficiency_ge_80pct": True,
        "data_stall_le_5pct": True,
        "memory_headroom_ge_10pct": True,
    }


def test_session_builds_same_environment_resumable_commands(tmp_path):
    settings = bench_utils.BenchmarkSettings(
        world_sizes=(1, 4),
        strategies=("ddp",),
        repeats=2,
        capacity_batches=(1, 2),
    )
    session = bench_utils.DistributedBenchmark(
        root=Path.cwd(),
        output=tmp_path,
        settings=settings,
        python=sys.executable,
    )

    weak = session.command("weak")
    capacity = session.command("capacity", rerun_existing=True)
    assert weak[0] == str(Path(sys.executable).absolute())
    assert "--world-sizes" in weak and "4" in weak
    assert "--repeats" in weak and weak[weak.index("--repeats") + 1] == "2"
    assert "--batch-sizes" in capacity
    assert "--rerun-existing" in capacity


def test_session_preserves_virtualenv_python_symlink(tmp_path):
    real_python = tmp_path / "system-python"
    real_python.write_text("", encoding="utf-8")
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(real_python)

    session = bench_utils.DistributedBenchmark(
        root=Path.cwd(),
        output=tmp_path / "output",
        python=venv_python,
    )

    assert session.python == str(venv_python.absolute())


def test_single_strategy_is_limited_to_one_gpu(tmp_path):
    settings = bench_utils.BenchmarkSettings(
        world_sizes=(1, 4),
        strategies=("single", "fsdp"),
        capacity_batches=(1,),
    )
    session = bench_utils.DistributedBenchmark(
        root=Path.cwd(), output=tmp_path, settings=settings
    )

    command = session.command("capacity")
    assert command[command.index("--strategies") + 1 : command.index("--world-sizes")] == [
        "single",
        "fsdp",
    ]
    assert bench_utils.SUPPORTED_TOPOLOGIES["single"] == (1,)


def test_report_visualization_and_presentation_are_persisted(tmp_path):
    pytest.importorskip("matplotlib")
    path = tmp_path / "weak_summary.json"
    path.write_text(json.dumps(weak_report()), encoding="utf-8")

    outputs = bench_utils.BenchmarkReport(path).save_presentation()

    assert outputs is not None
    assert all(output.is_file() for output in outputs.values())
    assert outputs["figure"].stat().st_size > 0
    assert "repeats_successful" in outputs["results"].read_text(encoding="utf-8")

    capacity_path = tmp_path / "capacity_summary.json"
    capacity_path.write_text(json.dumps(capacity_report()), encoding="utf-8")
    capacity_outputs = bench_utils.BenchmarkReport(capacity_path).save_presentation()
    assert capacity_outputs["figure"].stat().st_size > 0
    assert json.loads(capacity_outputs["metrics"].read_text(encoding="utf-8"))


def test_notebook_keeps_only_high_level_benchmark_calls():
    notebook = json.loads(
        Path("tests/distributed_server_benchmark.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "DistributedBenchmark(" in source
    assert "bench.preflight()" in source
    assert "bench.run_weak()" in source
    assert "bench.run_capacity()" in source
    assert "!python" not in source


def test_distributed_worker_does_not_read_config_from_wrapped_model():
    source = Path("scripts/run_dist_bench.py").read_text(encoding="utf-8")

    wrapped_section = source.split("model = strategy.wrap_model(model)", 1)[1]
    assert "model.cfg" not in wrapped_section
    assert "model_seq_len" in wrapped_section
    assert wrapped_section.count("if dist.is_initialized():") >= 2
