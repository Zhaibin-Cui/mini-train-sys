import unittest

from experiments.synbios_moe.mechanisms.token_conditioning import (
    GroundTruthFirstWholeDataset,
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


if __name__ == "__main__":
    unittest.main()
