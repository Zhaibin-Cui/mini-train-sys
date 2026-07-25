import argparse
import importlib.util
import json
from pathlib import Path

from tests.dense_operator_bench_runner import PROFILES as DENSE_PROFILES
from tests.moe_operator_bench_runner import PROFILES, ROUTER_SIZES


def _script(name):
    path = Path("scripts") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _backend_row(backend, repeat, throughput, step_ms, allocated, reserved):
    return {
        "strategy": "fsdp",
        "world_size": 4,
        "local_batch_size": 112,
        "seq_len": 512,
        "precision": "bf16",
        "parameter_count": 293_494_272,
        "ops_backend": backend,
        "repeat": repeat,
        "throughput_tokens_per_sec": throughput,
        "step_time_ms_mean": step_ms,
        "peak_memory_allocated_mb": allocated,
        "peak_memory_reserved_mb": reserved,
        "gpu_memory_total_mb": 24_000,
        "backend_dispatch": (
            {
                "attention": {"native_cuda": 480, "fallback": 0},
                "native_cuda_operators": ["attention"],
            }
            if backend == "cuda"
            else None
        ),
    }


def test_backend_presentation_requires_native_dispatch_and_writes_artifacts(tmp_path):
    source = tmp_path / "backend_summary.json"
    source.write_text(
        json.dumps(
            {
                "suite": "backend",
                "failures": [],
                "results": [
                    _backend_row("torch", 0, 300_000, 760, 20_000, 22_000),
                    _backend_row("torch", 1, 302_000, 755, 20_100, 22_000),
                    _backend_row("triton", 0, 350_000, 650, 18_500, 20_500),
                    _backend_row("triton", 1, 352_000, 647, 18_600, 20_500),
                    _backend_row("cuda", 0, 360_000, 635, 18_000, 20_000),
                    _backend_row("cuda", 1, 362_000, 632, 18_100, 20_000),
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "presentation"
    validation = tmp_path / "backend_validation.json"
    validation.write_text(
        json.dumps({"quality_gate_passed": True}),
        encoding="utf-8",
    )

    _script("run_dist_bench").present_backend(
        argparse.Namespace(
            input=str(source),
            validation=str(validation),
            output=str(output),
        )
    )

    result = json.loads((output / "backend_comparison.json").read_text(encoding="utf-8"))
    assert result["quality_gate_passed"]
    assert result["comparison"]["throughput_speedup_cuda_vs_torch"] > 1
    assert result["comparison"]["throughput_speedup_cuda_vs_triton"] > 1
    assert (output / "backend_aggregates.csv").is_file()
    assert (output / "backend_comparison.png").is_file()


def test_capacity_presentation_uses_reserved_vram_and_backend_specific_best_batch(
    tmp_path,
):
    source = tmp_path / "capacity_summary.json"
    rows = []
    for backend, cases in {
        "torch": ((64, 210_000, 18_000), (80, 225_000, 22_500)),
        "triton": (
            (64, 235_000, 17_000),
            (80, 260_000, 18_500),
            (96, 285_000, 21_500),
        ),
        "cuda": (
            (64, 245_000, 16_500),
            (96, 295_000, 20_000),
            (112, 310_000, 22_600),
        ),
    }.items():
        for batch, throughput, reserved in cases:
            for repeat in range(2):
                row = _backend_row(
                    backend,
                    repeat,
                    throughput + repeat * 1_000,
                    700,
                    reserved - 1_000,
                    reserved,
                )
                row["local_batch_size"] = batch
                row["global_batch_size"] = batch * 4
                rows.append(row)
    source.write_text(
        json.dumps(
            {
                "suite": "capacity",
                "settings": {
                    "repeats": 2,
                    "batch_sizes": [64, 80, 96, 112, 128],
                },
                "results": rows,
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "presentation"
    _script("run_dist_bench").present_backend_capacity(
        argparse.Namespace(
            input=str(source),
            output=str(output),
            memory_limit_percent=92.0,
            min_repeats=2,
        )
    )

    result = json.loads(
        (output / "capacity_backend_comparison.json").read_text(encoding="utf-8")
    )
    selected = {row["ops_backend"]: row for row in result["selected"]}
    assert selected["torch"]["local_batch_size"] == 64
    assert selected["triton"]["local_batch_size"] == 96
    assert selected["cuda"]["local_batch_size"] == 96
    assert selected["cuda"]["throughput_speedup_vs_torch"] > 1
    assert result["common_eligible_batches"] == [64]
    assert result["common_fixed_batch"] == 64
    assert result["quality_gate_passed"]
    assert (output / "capacity_selected.csv").is_file()
    assert (output / "capacity_backend_comparison.png").is_file()


def test_4090_scan_covers_formal_token_load_and_industrial_reference():
    formal = PROFILES["project_formal"]
    industrial = PROFILES["mixtral_7b"]
    assert 112 * 512 in formal.token_sizes
    assert (industrial.hidden, industrial.intermediate) == (4096, 14336)
    assert max(ROUTER_SIZES) >= 262_144

    dense_formal = DENSE_PROFILES["project_formal"]
    dense_industrial = DENSE_PROFILES["mixtral_dense"]
    assert dense_formal.token_sizes[-1] == 112 * 512
    assert (dense_formal.hidden, dense_formal.head_dim, dense_formal.vocab) == (
        768,
        64,
        50257,
    )
    assert (dense_industrial.hidden, dense_industrial.intermediate) == (4096, 14336)


def test_server_notebook_uses_project_kernel_and_isolated_runner():
    for path, runner in (
        ("tests/operator_bench_linux_server.ipynb", "DenseOperatorBenchmark"),
        ("tests/moe_operator_bench_linux_server.ipynb", "MoeOperatorBenchmark"),
    ):
        notebook = json.loads(Path(path).read_text(encoding="utf-8"))
        assert notebook["metadata"]["kernelspec"]["name"] == "mini-train-sys"
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert runner in source
        assert "bench_sweep" not in source


def test_notebook_error_scanner_reports_cell_identity(tmp_path):
    path = tmp_path / "failed.ipynb"
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "outputs": [
                            {
                                "output_type": "error",
                                "ename": "RuntimeError",
                                "evalue": "boom",
                                "traceback": ["line"],
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert _script("run_server_notebook")._cell_errors(path)[0]["cell"] == 0


def test_ground_truth_first_probe_helper_uses_indexed_cuda_device():
    source = Path("scripts/bash/synbios_ground_truth_first_whole.sh").read_text(
        encoding="utf-8"
    )
    assert "CUDA_VISIBLE_DEVICES=" in source
    assert "--device cuda:0" in source
    assert "--device cuda " not in source


def test_ground_truth_first_probe_uses_selected_pilot_budget_and_dynamic_scheduler():
    source = Path("scripts/bash/synbios_ground_truth_first_whole.sh").read_text(
        encoding="utf-8"
    )
    assert 'STEPS="${STEPS:-4000}"' in source
    assert 'P_BATCH_SIZE="${P_BATCH_SIZE:-128}"' in source
    assert 'Q_BATCH_SIZE="${Q_BATCH_SIZE:-768}"' in source
    assert "train-ground-truth-first-whole" in source
    assert "--max-validation-examples" not in source
    assert "prediction-batch-size" not in source
    assert "wait -n -p finished_pid" in source
    assert "skip completed task:" in source
    diagnostics = Path("experiments/synbios_moe/probe_diagnostics.py").read_text(
        encoding="utf-8"
    )
    assert "evaluate_validation=False" in diagnostics
    assert diagnostics.count("evaluate_ground_truth_first_whole_by_source_position(") == 2


def test_synbios_backend_benchmark_keeps_two_comparison_views():
    source = Path("scripts/bash/synbios_backend_benchmark.sh").read_text(encoding="utf-8")
    assert "validate-backend" in source
    assert "--suite backend" in source
    assert '--local-batch "$COMMON_BATCH"' in source
    assert '"common_fixed_batch"' in source
    assert "--warmup-steps 10 --measure-steps 30 --repeats 3" in source
    assert "--suite capacity" in source
    assert "--batch-sizes 1 2 4 8 16 24 32 48 64 80 96 112 120 128" in source
    assert "--warmup-steps 5 --measure-steps 20 --repeats 2" in source
    assert "--ops-backends torch triton cuda" in source
    assert "--memory-limit-percent 92 --min-repeats 2" in source
    assert "export_test_results.sh" in source
