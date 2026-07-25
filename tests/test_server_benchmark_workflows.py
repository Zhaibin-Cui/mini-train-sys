import argparse
import importlib.util
import json
from pathlib import Path

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


def test_4090_scan_covers_formal_token_load_and_industrial_reference():
    formal = PROFILES["project_formal"]
    industrial = PROFILES["mixtral_7b"]
    assert 112 * 512 in formal.token_sizes
    assert (industrial.hidden, industrial.intermediate) == (4096, 14336)
    assert max(ROUTER_SIZES) >= 262_144


def test_server_notebook_uses_project_kernel_and_isolated_runner():
    notebook = json.loads(
        Path("tests/moe_operator_bench_linux_server.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["metadata"]["kernelspec"]["name"] == "mini-train-sys"
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "MoeOperatorBenchmark" in source
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
