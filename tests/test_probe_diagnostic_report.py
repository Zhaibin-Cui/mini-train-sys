import pytest

from experiments.synbios_moe.diagnostic_report import _oracle_rows, _route_rows
from experiments.synbios_moe.formal_report import WHOLE_ATTRIBUTES, FormalRun


def _formal(validation_accuracy: float = 0.4) -> FormalRun:
    validation = {
        f"q_{attribute}_whole": {
            "examples": 10,
            "validation_accuracy": [validation_accuracy],
        }
        for attribute in WHOLE_ATTRIBUTES
    }
    return FormalRun(
        condition="multi5_permute",
        root=None,
        pipeline={"identity": {}},
        data_manifest={},
        cache_manifest={},
        validation=validation,
        training={},
    )


def _oracle_attribute_row(attribute: str) -> dict[str, object]:
    return {
        "attribute": attribute,
        "examples": 10,
        "accuracy_before": 0.4,
        "accuracy_after": 0.5,
        "accuracy_delta": 0.1,
        "baseline_errors": 6,
        "recovered_errors": 2,
        "recovery_rate": 2 / 6,
        "baseline_correct": 4,
        "harmed_correct": 1,
        "harm_rate": 0.25,
    }


def test_oracle_report_reconstructs_counts_and_matches_formal_baseline():
    attributes = [_oracle_attribute_row(attribute) for attribute in WHOLE_ATTRIBUTES]
    overall = {
        "examples": 50,
        "accuracy_before": 0.4,
        "accuracy_after": 0.5,
        "accuracy_delta": 0.1,
        "baseline_errors": 30,
        "recovered_errors": 10,
        "recovery_rate": 1 / 3,
        "baseline_correct": 20,
        "harmed_correct": 5,
        "harm_rate": 0.25,
    }
    rows = _oracle_rows(
        _formal(),
        {"attributes": attributes, "overall": overall},
        [{key: str(value) for key, value in row.items()} for row in attributes],
    )
    assert len(rows) == 6
    assert rows[-1]["attribute"] == "micro_overall"
    assert rows[-1]["accuracy_delta"] == pytest.approx(0.1)


def test_oracle_report_rejects_baseline_that_differs_from_formal():
    attributes = [_oracle_attribute_row(attribute) for attribute in WHOLE_ATTRIBUTES]
    with pytest.raises(ValueError, match="formal/oracle baseline"):
        _oracle_rows(
            _formal(validation_accuracy=0.3),
            {"attributes": attributes, "overall": {}},
            [{key: str(value) for key, value in row.items()} for row in attributes],
        )


def _route_inputs():
    pairs = []
    nmi = []
    for attribute in WHOLE_ATTRIBUTES:
        for layer in range(12):
            pairs.extend(
                [
                    {
                        "attribute": attribute,
                        "layer": str(layer),
                        "pair_group": "same_t2",
                        "pair_count": "100",
                        "t1_route_overlap": "0.6",
                        "t2_route_overlap": "0.7",
                        "branching_score": "-0.1",
                    },
                    {
                        "attribute": attribute,
                        "layer": str(layer),
                        "pair_group": "different_t2",
                        "pair_count": "100",
                        "t1_route_overlap": "0.6",
                        "t2_route_overlap": "0.3",
                        "branching_score": "0.3",
                    },
                ]
            )
            nmi.append(
                {
                    "attribute": attribute,
                    "layer": str(layer),
                    "t1_top1_token_nmi": "0.2",
                    "t2_top1_token_nmi": "0.4",
                    "examples": "10",
                }
            )
    summary = {
        "case_definition": "Q-first correct, Q-whole wrong, whole value has >=2 tokens",
        "layers": 12,
        "experts": 8,
        "top_k": 2,
        "examples": 50,
        "attributes": {attribute: 10 for attribute in WHOLE_ATTRIBUTES},
    }
    return summary, pairs, nmi


def test_route_report_builds_complete_controlled_layer_contrast():
    layer_rows, attribute_rows, headline = _route_rows(*_route_inputs())
    assert len(layer_rows) == 12
    assert len(attribute_rows) == 60
    assert headline["difference_in_differences"] == pytest.approx(0.4)
    assert headline["all_layers_positive"] is True


def test_route_report_rejects_incomplete_attribute_layer_matrix():
    summary, pairs, nmi = _route_inputs()
    with pytest.raises(ValueError, match="complete 5x12x2 matrix"):
        _route_rows(summary, pairs[:-1], nmi)
