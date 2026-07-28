import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_results_catalog import build_catalog
from scripts.build_results_manifest import build_manifest, sha256_file, verify_manifest


class ResultsCatalogTest(unittest.TestCase):
    def test_local_catalog_refresh_preserves_server_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            catalog = results / "catalog"
            catalog.mkdir(parents=True)
            published = {
                "schema_version": 1,
                "artifact_root": "/data/mini-train-sys/artifacts",
                "groups": [
                    {
                        "name": "formal_checkpoints",
                        "logical_path": "/data/checkpoints",
                        "files": 4,
                        "size_bytes": 1024,
                        "retention": "server_only",
                    }
                ],
            }
            (catalog / "retention.json").write_text(
                json.dumps(published),
                encoding="utf-8",
            )
            sample = results / "benchmarks" / "sample.json"
            sample.parent.mkdir(parents=True)
            sample.write_text("{}\n", encoding="utf-8")

            build_catalog(results, root / "missing-artifacts")

            retained = json.loads((catalog / "retention.json").read_text(encoding="utf-8"))
            self.assertEqual(retained, published)
            artifacts = json.loads((catalog / "artifacts.json").read_text(encoding="utf-8"))
            self.assertEqual(artifacts["files"][0]["path"], "benchmarks/sample.json")
            for generated in (
                catalog / "artifacts.json",
                catalog / "retention.json",
                catalog / "summary.md",
                results / "tensorboard" / "index.csv",
            ):
                self.assertNotIn(b"\r\n", generated.read_bytes())

    def test_manifest_uses_repository_relative_forward_slash_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            sample = results / "nested" / "sample.txt"
            sample.parent.mkdir(parents=True)
            sample.write_text("sample\n", encoding="utf-8")

            manifest = build_manifest(results)
            line = manifest.read_text(encoding="utf-8").strip()

            self.assertEqual(line, f"{sha256_file(sample)}  results/nested/sample.txt")
            self.assertNotIn(b"\r\n", manifest.read_bytes())
            verify_manifest(results)

    def test_manifest_check_finds_unlisted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            first = results / "first.txt"
            first.parent.mkdir(parents=True)
            first.write_text("first\n", encoding="utf-8")
            build_manifest(results)
            (results / "second.txt").write_text("second\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unlisted"):
                verify_manifest(results)


if __name__ == "__main__":
    unittest.main()
