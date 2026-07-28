import tempfile
import unittest
from pathlib import Path

from experiments.synbios_moe.mechanisms.comparison_report import (
    _same_path,
    _validate_raw_manifest,
)


class ReportLineageTest(unittest.TestCase):
    def test_server_and_exported_formal_paths_share_a_logical_identity(self):
        server = (
            "/data/mini-train-sys/artifacts/synbios_moe/results/"
            "multi5_permute_fsdp_4gpu/probe_pipeline/formal/training"
        )
        exported = (
            "C:/repo/results/formal_runs/synbios_moe/results/"
            "multi5_permute_fsdp_4gpu/probe_pipeline/formal/training"
        )

        self.assertTrue(_same_path(server, exported))

    def test_unrelated_run_paths_do_not_match(self):
        self.assertFalse(
            _same_path(
                "/data/mini-train-sys/artifacts/synbios_moe/results/single/formal",
                "C:/repo/results/formal_runs/synbios_moe/results/multi/formal",
            )
        )

    def test_externally_retained_raw_evidence_may_be_absent(self):
        manifest = {
            "format_version": 1,
            "retention": "mounted artifact; excluded from Git result export",
            "artifacts": [
                {
                    "logical_path": "oracle_first_token/records.csv",
                    "bytes": 1,
                    "sha256": "0" * 64,
                },
                {
                    "logical_path": "bad_case_routes/bad_cases.csv",
                    "bytes": 1,
                    "sha256": "0" * 64,
                },
                {
                    "logical_path": "bad_case_routes/route_records.csv",
                    "bytes": 1,
                    "sha256": "0" * 64,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            _validate_raw_manifest(Path(directory), manifest)


if __name__ == "__main__":
    unittest.main()
