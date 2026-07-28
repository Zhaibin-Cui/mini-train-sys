import unittest

from experiments.synbios_moe.mechanisms.first_token_intervention import (
    GroundTruthFirstWholeDataset,
)
from experiments.synbios_moe.mechanisms.oracle_first_token import (
    insert_oracle_first_token,
    summarize_oracle_rows,
)
from experiments.synbios_moe.probes.model import ProbeBatchItem


class _ProbeItems:
    def __init__(self, item: ProbeBatchItem, class_names: list[str]) -> None:
        self.item = item
        self.class_names = class_names

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> ProbeBatchItem:
        if index != 0:
            raise IndexError(index)
        return self.item


class FirstTokenInterventionTest(unittest.TestCase):
    def test_whole_probe_input_uses_the_aligned_first_token_label(self):
        first = _ProbeItems(
            ProbeBatchItem(
                input_ids=[10, 11, 12, 13, 14, 15],
                positions=[0, 1, 2, 3, 4, 5],
                label=1,
            ),
            class_names=["101", "202"],
        )
        whole = _ProbeItems(
            ProbeBatchItem(
                input_ids=[10, 11, 12, 13, 14, 15],
                positions=[0, 1, 2, 3, 4, 5],
                label=7,
            ),
            class_names=["value"],
        )
        dataset = GroundTruthFirstWholeDataset(
            first_data=first,
            whole_data=whole,
            token_ids_by_class=[101, 202],
        )

        rebuilt = dataset[3]

        self.assertEqual(rebuilt.input_ids, [10, 11, 12, 13, 202])
        self.assertEqual(rebuilt.positions, [4])
        self.assertEqual(rebuilt.label, 7)

    def test_oracle_input_keeps_the_final_q_readout_token(self):
        rebuilt = insert_oracle_first_token([50256, 11, 50256], 42, 50256)

        self.assertEqual(rebuilt, [50256, 11, 42, 50256])

    def test_oracle_summary_counts_recovered_and_harmed_examples(self):
        summary = summarize_oracle_rows(
            [
                {"whole_before_correct": False, "whole_after_correct": True},
                {"whole_before_correct": False, "whole_after_correct": False},
                {"whole_before_correct": True, "whole_after_correct": False},
            ]
        )

        self.assertAlmostEqual(summary["accuracy_before"], 1 / 3)
        self.assertAlmostEqual(summary["accuracy_after"], 1 / 3)
        self.assertAlmostEqual(summary["recovery_rate"], 1 / 2)
        self.assertEqual(summary["harmed_correct"], 1)


if __name__ == "__main__":
    unittest.main()
