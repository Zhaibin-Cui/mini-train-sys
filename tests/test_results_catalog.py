from pathlib import Path

from scripts.build_results_catalog import build_catalog
from scripts.build_results_catalog import classify_log
from scripts.audit_results_export import artifact_mapping


def test_log_categories_are_stable():
    assert classify_log("kernel_industrial.log") == "benchmarks"
    assert classify_log("probe_formal_single.log") == "experiments"
    assert classify_log("final_regression.log") == "validation"
    assert classify_log("export_results.log") == "maintenance"


def test_catalog_indexes_tensorboard_and_retention(tmp_path: Path):
    results = tmp_path / "results"
    event = (
        results
        / "formal_runs/synbios_moe/runs/synbios_moe_single_fsdp_4gpu/run"
        / "events.out.tfevents.test"
    )
    event.parent.mkdir(parents=True)
    event.write_bytes(b"event")
    (results / "benchmarks/operator_benchmark").mkdir(parents=True)
    (results / "benchmarks/operator_benchmark/summary.json").write_text("{}")

    artifacts = tmp_path / "artifacts"
    dataset = artifacts / "synbios_moe/single"
    dataset.mkdir(parents=True)
    (dataset / "manifest.json").write_text('{"variant": "single"}')
    (dataset / "payload.jsonl").write_text("large payload")

    payload = build_catalog(results, artifacts)

    assert payload["tensorboard"]["events"] == 1
    assert {group["scope"] for group in payload["groups"]} >= {
        "kernels",
        "synbios_moe",
    }
    assert (results / "catalog/artifacts.json").is_file()
    assert (results / "catalog/retention.json").is_file()
    assert (results / "catalog/summary.md").is_file()
    assert (results / "tensorboard/index.csv").is_file()


def test_export_mapping_covers_pushable_and_large_files(tmp_path: Path):
    results = tmp_path / "results"
    target, reason = artifact_mapping(
        Path("logs/probe_formal.log"), results
    )
    assert target == results / "logs/experiments/probe_formal.log"
    assert reason is None

    target, reason = artifact_mapping(
        Path("synbios_moe/checkpoints/run/final/model.pt"), results
    )
    assert target is None
    assert reason == "model_export"

    target, reason = artifact_mapping(
        Path("synbios_moe/results/run/probe_pipeline/formal/summary/summary.json"),
        results,
    )
    assert target == (
        results
        / "formal_runs/synbios_moe/results/run/probe_pipeline/formal/summary/summary.json"
    )
    assert reason is None
