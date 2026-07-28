import json
import tempfile
import unittest
from pathlib import Path

from experiments.synbios_moe.artifact_io import (
    read_csv_rows,
    read_json_object,
    resolve_hashed_file,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)


class ArtifactIoTest(unittest.TestCase):
    def test_json_round_trip_is_sorted_and_newline_terminated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "summary.json"

            write_json_atomic(path, {"z": 1, "a": 2})

            self.assertEqual(read_json_object(path), {"a": 2, "z": 1})
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            self.assertLess(path.read_text(encoding="utf-8").index('"a"'), 20)

    def test_csv_round_trip_uses_stable_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            rows = [{"name": "first", "value": 1}, {"name": "second", "value": 2}]

            write_csv_atomic(path, rows)

            self.assertEqual(
                read_csv_rows(path),
                [{"name": "first", "value": "1"}, {"name": "second", "value": "2"}],
            )
            self.assertNotIn(b"\r\n", path.read_bytes())

    def test_empty_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "empty CSV"):
                write_csv_atomic(Path(directory) / "empty.csv", [])

    def test_text_and_hash_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "note.txt"

            write_text_atomic(path, "line\n\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "line\n")
            self.assertEqual(
                sha256_file(path),
                "893e89e669b5a4f9e5136d565f51e341a0c5e5531816c9c1a806d90df66a45f4",
            )

    def test_json_reader_rejects_non_object_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "list.json"
            path.write_text(json.dumps([1, 2]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "JSON object"):
                read_json_object(path)

    def test_hashed_file_uses_verified_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fallback = root / "exported" / "manifest.json"
            write_json_atomic(fallback, {"variant": "single"})
            expected = sha256_file(fallback)

            resolved = resolve_hashed_file(
                root / "server" / "manifest.json",
                expected,
                fallbacks=(fallback,),
                label="dataset manifest",
            )

            self.assertEqual(resolved, fallback)

    def test_hashed_file_rejects_wrong_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            fallback = Path(directory) / "manifest.json"
            write_json_atomic(fallback, {"variant": "single"})

            with self.assertRaisesRegex(ValueError, "fallback SHA256 mismatch"):
                resolve_hashed_file(
                    Path(directory) / "missing.json",
                    "0" * 64,
                    fallbacks=(fallback,),
                    label="dataset manifest",
                )


if __name__ == "__main__":
    unittest.main()
