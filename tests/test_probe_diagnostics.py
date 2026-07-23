from experiments.synbios_moe.probe_diagnostics import (
    insert_oracle_first_token,
    pairwise_route_summary,
    summarize_oracle_rows,
)


def test_oracle_insertion_preserves_final_eos_readout():
    assert insert_oracle_first_token([99, 4, 5, 99], 17, 99) == [99, 4, 5, 17, 99]


def test_oracle_summary_separates_recovery_and_harm():
    rows = [
        {"whole_before_correct": False, "whole_after_correct": True},
        {"whole_before_correct": False, "whole_after_correct": False},
        {"whole_before_correct": True, "whole_after_correct": False},
        {"whole_before_correct": True, "whole_after_correct": True},
    ]
    summary = summarize_oracle_rows(rows)
    assert summary["accuracy_before"] == 0.5
    assert summary["accuracy_after"] == 0.5
    assert summary["recovery_rate"] == 0.5
    assert summary["harm_rate"] == 0.5


def test_pairwise_route_summary_detects_second_token_branching():
    cases = []
    for case_id, t2, t2_route in (
        (0, 10, [[2, 3], [2, 3]]),
        (1, 10, [[2, 3], [2, 3]]),
        (2, 11, [[4, 5], [4, 5]]),
        (3, 11, [[4, 5], [4, 5]]),
    ):
        cases.append(
            {
                "case_id": case_id,
                "attribute": "university",
                "t1_id": 7,
                "t2_id": t2,
                "routes": [
                    [[0, 1], [0, 1]],
                    t2_route,
                ],
            }
        )
    rows = pairwise_route_summary(cases, layers=2, pair_limit=20, seed=3)
    same = [row for row in rows if row["pair_group"] == "same_t2"]
    different = [row for row in rows if row["pair_group"] == "different_t2"]
    assert all(row["t1_route_overlap"] == 1.0 for row in same + different)
    assert all(row["t2_route_overlap"] == 1.0 for row in same)
    assert all(row["t2_route_overlap"] == 0.0 for row in different)
    assert all(row["branching_score"] == 1.0 for row in different)


def test_pairwise_route_summary_allows_no_eligible_control_pairs():
    cases = [
        {
            "attribute": "company",
            "t1_id": 7,
            "t2_id": 10,
            "routes": [[[0, 1]], [[2, 3]]],
        }
    ]
    assert pairwise_route_summary(cases, layers=1, pair_limit=10) == []
