import unittest

from experiments.synbios_moe.router_analysis import normalized_mutual_information


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


if __name__ == "__main__":
    unittest.main()
