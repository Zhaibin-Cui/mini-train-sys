import unittest

from experiments.synbios_moe.mechanisms.routing import normalized_mutual_information
from experiments.synbios_moe.mechanisms.token_routes import (
    pairwise_route_summary,
    route_jaccard,
)


class RoutingMetricTest(unittest.TestCase):
    def test_perfect_assignment_has_unit_score(self):
        score = normalized_mutual_information(
            assignments=[0, 0, 1, 1],
            labels=[0, 0, 1, 1],
        )

        self.assertAlmostEqual(score, 1.0)

    def test_balanced_independent_assignment_has_zero_score(self):
        score = normalized_mutual_information(
            assignments=[0, 0, 1, 1],
            labels=[0, 1, 0, 1],
        )

        self.assertAlmostEqual(score, 0.0)

    def test_empty_or_mismatched_inputs_return_zero(self):
        self.assertEqual(normalized_mutual_information([], []), 0.0)
        self.assertEqual(normalized_mutual_information([0], [0, 1]), 0.0)

    def test_route_jaccard_uses_the_routed_expert_sets(self):
        self.assertEqual(route_jaccard([1, 2], [1, 2]), 1.0)
        self.assertEqual(route_jaccard([1, 2], [3, 4]), 0.0)
        self.assertAlmostEqual(route_jaccard([1, 2], [2, 3]), 1 / 3)

    def test_pairwise_branching_separates_shared_t1_from_different_t2(self):
        cases = [
            {
                "attribute": "major",
                "t1_id": 10,
                "t2_id": t2,
                "routes": [[[0, 1]], [[2, 3] if t2 == 20 else [4, 5]]],
            }
            for t2 in (20, 20, 21, 21)
        ]

        rows = pairwise_route_summary(cases, layers=1, pair_limit=8, seed=7)
        by_group = {row["pair_group"]: row for row in rows}

        self.assertEqual(by_group["same_t2"]["branching_score"], 0.0)
        self.assertEqual(by_group["different_t2"]["branching_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
