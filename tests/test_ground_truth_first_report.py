import json

from experiments.synbios_moe.ground_truth_first_report import (
    ATTRIBUTES,
    KINDS,
    PROTOCOL,
    _condition,
    summarize_ground_truth_first_whole,
)


def test_ground_truth_first_report_requires_and_renders_complete_matrix(tmp_path):
    for kind in KINDS:
        positions = 6
        for attribute in ATTRIBUTES:
            payload = {
                "protocol": PROTOCOL,
                "kind": kind,
                "attribute": attribute,
                "data": "/data/multi5_permute",
                "probe_cache": "/data/multi5_permute/probe_cache",
                "probe_dir": "/data/formal/training",
                "checkpoint": "/data/checkpoint",
                "probe_cache_manifest_sha256": "abc",
                "ground_truth_first_token": True,
                "rank": 2,
                "architecture_match": {
                    "rank": 2,
                    "reference_rank": 2,
                    "low_rank_embedding_delta": True,
                },
                "alignment_checks": {"complete": True},
                "ground_truth_first_accuracy_by_position": [1.0] * positions,
                "whole_accuracy_validation_by_source_position": [0.6] * positions,
                "classes": 100,
                "steps": 3000,
                "batch_size": 128,
            }
            (tmp_path / f"{kind}_{attribute}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

    summary = summarize_ground_truth_first_whole(tmp_path)

    assert summary["tasks"] == 5
    assert len(summary["rows"]) == 30
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "summary.json").is_file()
    report = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Question or hypothesis" in report
    assert "Exact compared conditions" in report
    assert "Primary metrics" in report
    assert "Limitations and threats to validity" in report
    assert all((tmp_path / path).is_file() for path in summary["figures"])


def test_ground_truth_first_report_identifies_single_condition_from_data_path():
    assert _condition({"data": "/data/synbios_moe/single"}) == "single"
