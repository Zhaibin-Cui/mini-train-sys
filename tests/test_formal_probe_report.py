import json

import pytest

from experiments.synbios_moe.formal_report import (
    ATTRIBUTES,
    EXPECTED_TASKS,
    WHOLE_ATTRIBUTES,
    FormalRun,
    headline_metrics,
    load_formal_run,
    validate_matched_runs,
)


def _validation_rows(single_value: float, multi_value: float):
    rows = []
    for condition, value in (("single", single_value), ("multi5_permute", multi_value)):
        for attribute in ATTRIBUTES:
            for target in ("first", "whole"):
                if attribute == "birth_date" and target == "whole":
                    continue
                for kind, positions in (("p", 6), ("q", 1)):
                    for position in range(positions):
                        rows.append(
                            {
                                "condition": condition,
                                "split": "person_held_out_validation",
                                "kind": kind,
                                "target": target,
                                "attribute": attribute,
                                "position": position,
                                "accuracy": value,
                            }
                        )
    return rows


def _formal_run(condition: str, profiles_sha256: str) -> FormalRun:
    identity = {
        "model_config_sha256": "model",
        "seed": 1337,
        "steps": {"first": 4000, "whole": 12000},
        "runtime": {"batch": 128},
        "jobs": list(EXPECTED_TASKS),
    }
    records = {
        task: {"classes": 3, "class_names": ["a", "b", "c"]} for task in EXPECTED_TASKS
    }
    return FormalRun(
        condition=condition,
        root=None,
        pipeline={"identity": identity},
        data_manifest={
            "variant": "single" if condition == "single" else "multi5+permute",
            "files": {"profiles.jsonl": {"sha256": profiles_sha256}},
        },
        cache_manifest={},
        validation=records,
        training=records,
    )


def test_headline_metrics_keep_conditions_and_endpoints_separate():
    metrics = headline_metrics(_validation_rows(0.1, 0.9))
    assert metrics["single"]["q_first_six_attribute_macro_mean"] == pytest.approx(0.1)
    assert metrics["multi5_permute"]["q_whole_five_attribute_macro_mean"] == pytest.approx(0.9)
    assert metrics["delta_multi_minus_single"][
        "p_first_position0_non_birth_date_mean"
    ] == pytest.approx(0.8)


def test_matched_run_validation_rejects_different_people():
    with pytest.raises(ValueError, match="same person/profile"):
        validate_matched_runs(
            _formal_run("single", "profile-a"),
            _formal_run("multi5_permute", "profile-b"),
        )


def test_load_formal_run_rejects_incomplete_pipeline(tmp_path):
    (tmp_path / "pipeline.json").write_text(
        json.dumps({"status": "running", "stage": "formal"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a completed formal pipeline"):
        load_formal_run("single", tmp_path)


def test_expected_matrix_has_22_tasks_and_five_whole_attributes():
    assert len(EXPECTED_TASKS) == 22
    assert len(WHOLE_ATTRIBUTES) == 5
