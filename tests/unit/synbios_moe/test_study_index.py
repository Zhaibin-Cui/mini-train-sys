import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
INDEX_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "formal_runs"
    / "synbios_moe"
    / "study_index.json"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _last_event(path: Path, event_name: str) -> dict:
    last_match = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") == event_name:
                last_match = event
    if last_match is None:
        raise AssertionError(f"{path} has no {event_name!r} event")
    return last_match


class FormalStudyIndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = _load_json(INDEX_PATH)

    def test_all_repository_paths_exist(self) -> None:
        shared = self.index["shared"]
        shared_paths = (
            shared["model_config"],
            shared["probe_config"],
            shared["entrypoint"],
            shared["training_entrypoint"],
        )
        indexed_paths = list(shared_paths)
        for condition in self.index["conditions"].values():
            indexed_paths.extend(condition["paths"].values())
        indexed_paths.extend(self.index["comparison"].values())

        missing = [
            path
            for path in indexed_paths
            if not (REPOSITORY_ROOT / path).exists()
        ]
        self.assertEqual(missing, [])

    def test_conditions_share_the_profile_table(self) -> None:
        expected_hash = self.index["shared"]["profiles_sha256"]
        for condition in self.index["conditions"].values():
            lineage = _load_json(
                REPOSITORY_ROOT / condition["paths"]["dataset_lineage"]
            )
            self.assertEqual(
                lineage["source_dataset"]["profiles_sha256"],
                expected_hash,
            )
            self.assertEqual(
                lineage["source_dataset"]["people"],
                condition["expected"]["people"],
            )
            self.assertEqual(
                lineage["source_dataset"]["biographies"],
                condition["expected"]["biographies"],
            )

    def test_retained_training_runs_reach_the_indexed_endpoint(self) -> None:
        for condition in self.index["conditions"].values():
            events_path = REPOSITORY_ROOT / condition["paths"]["training_events"]
            final_train = _last_event(events_path, "train")
            final_checkpoint = _last_event(events_path, "checkpoint")
            expected = condition["expected"]

            self.assertEqual(final_train["step"], expected["training_steps"])
            self.assertEqual(final_train["epoch"], expected["training_epochs"])
            self.assertTrue(
                final_checkpoint["path"].endswith(
                    Path(condition["paths"]["final_checkpoint"]).name
                )
            )

    def test_cloze_shards_are_contiguous_and_match_the_totals(self) -> None:
        for condition in self.index["conditions"].values():
            summary = _load_json(
                REPOSITORY_ROOT / condition["paths"]["cloze_summary"]
            )
            expected = condition["expected"]
            next_start = 0
            for shard_range in summary["ranges"]:
                self.assertEqual(shard_range["start_index"], next_start)
                next_start += shard_range["biographies"]

            self.assertEqual(next_start, expected["biographies"])
            self.assertEqual(summary["biographies"], expected["biographies"])
            self.assertEqual(summary["fields"], expected["cloze_fields"])
            self.assertAlmostEqual(
                summary["micro_field_accuracy"],
                expected["cloze_micro_accuracy"],
            )

    def test_formal_comparison_matches_both_conditions(self) -> None:
        identity = _load_json(
            REPOSITORY_ROOT / self.index["comparison"]["run_identity"]
        )
        self.assertEqual(identity["comparison_status"], "matched")
        expected_hash = self.index["shared"]["profiles_sha256"]
        for name in self.index["conditions"]:
            self.assertEqual(identity[name]["profiles_sha256"], expected_hash)


if __name__ == "__main__":
    unittest.main()
