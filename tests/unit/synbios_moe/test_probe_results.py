import json
import tempfile
import unittest
from pathlib import Path

from experiments.synbios_moe.probes.commands import probe_train_command_builder
from experiments.synbios_moe.probes.results import summarize_probe_results
from experiments.synbios_moe.probes.spec import ProbeJob


def _validation_payload(*, accuracy: float, profiles_sha256: str) -> dict:
    return {
        "kind": "q",
        "attribute": "major",
        "target": "first",
        "validation_accuracy": [accuracy],
        "classes": 20,
        "examples": 50,
        "dataset_manifest": {
            "files": {"profiles.jsonl": {"sha256": profiles_sha256}}
        },
    }


class ProbeResultsTest(unittest.TestCase):
    def test_summary_checks_identity_and_writes_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            single = root / "single"
            augmented = root / "augmented"
            single.mkdir()
            augmented.mkdir()
            (single / "q_major_first.json").write_text(
                json.dumps(
                    _validation_payload(
                        accuracy=0.1,
                        profiles_sha256="shared-profiles",
                    )
                ),
                encoding="utf-8",
            )
            (augmented / "q_major_first.json").write_text(
                json.dumps(
                    _validation_payload(
                        accuracy=0.9,
                        profiles_sha256="shared-profiles",
                    )
                ),
                encoding="utf-8",
            )

            summary = summarize_probe_results(
                {"single": single, "multi5_permute": augmented},
                root / "summary",
                expected_jobs=(ProbeJob("q", "major", "first"),),
            )

            self.assertEqual(len(summary["rows"]), 2)
            self.assertAlmostEqual(summary["comparisons"][0]["delta"], 0.8)
            self.assertTrue((root / "summary" / "summary.json").is_file())
            self.assertTrue((root / "summary" / "comparison.csv").is_file())

    def test_training_command_uses_target_specific_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            builder = probe_train_command_builder(
                script=root / "synbios_moe.py",
                data=root / "data",
                cache=root / "cache",
                model_config=root / "model.yaml",
                checkpoint=root / "checkpoint",
                output_dir=root / "output",
                steps={
                    "p_first": 4,
                    "p_whole": 12,
                    "q_first": 8,
                    "q_whole": 24,
                },
                seed=1337,
                quiet=True,
                log_interval=10,
                tensorboard=False,
                batch_sizes={"p": 2, "q": 4},
                validation_batch_sizes={"p": 8, "q": 16},
                checkpoint_interval_steps=20,
                evaluate_train=False,
            )

            job_command = builder(ProbeJob("p", "major", "whole"), "cpu")
            steps_index = job_command.command.index("--steps") + 1

            self.assertEqual(job_command.command[steps_index], "12")
            self.assertEqual(job_command.output.name, "p_major_whole.pt")
            self.assertIn("--no-tensorboard", job_command.command)


if __name__ == "__main__":
    unittest.main()
