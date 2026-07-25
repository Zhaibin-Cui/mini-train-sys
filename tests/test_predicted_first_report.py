import json

from experiments.synbios_moe.predicted_first_report import (
    ATTRIBUTES,
    KINDS,
    PROTOCOL,
    summarize_predicted_first_whole,
)


def test_predicted_first_report_requires_and_renders_complete_matrix(tmp_path):
    for kind in KINDS:
        positions = 6 if kind == "p" else 1
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
                "first_accuracy_validation_by_position": [0.8] * positions,
                "whole_accuracy_validation_by_source_position": [0.6] * positions,
                "classes": 100,
                "steps": 3000,
                "batch_size": 128 if kind == "p" else 768,
            }
            (tmp_path / f"{kind}_{attribute}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

    summary = summarize_predicted_first_whole(tmp_path)

    assert summary["tasks"] == 10
    assert len(summary["rows"]) == 35
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert all((tmp_path / path).is_file() for path in summary["figures"])
