#!/usr/bin/env python3
"""Build the SHA256 manifest for every retained result file."""

import argparse
import hashlib
from pathlib import Path


MANIFEST_NAME = "MANIFEST.sha256"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(results_root: Path) -> Path:
    root = results_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"results directory does not exist: {root}")
    destination = root / MANIFEST_NAME
    files = result_files(root)
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root.parent).as_posix()}"
        for path in files
    ]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(destination)
    return destination


def result_files(results_root: Path) -> list[Path]:
    return sorted(
        path
        for path in results_root.resolve().rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    )


def verify_manifest(results_root: Path) -> None:
    root = results_root.resolve()
    manifest = root / MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError(f"missing result manifest: {manifest}")
    expected: dict[Path, str] = {}
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid manifest line {number}: {line}") from exc
        path = (root.parent / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"manifest path escapes results directory: {relative}")
        if path in expected:
            raise ValueError(f"duplicate manifest path: {relative}")
        expected[path] = digest
    actual = set(result_files(root))
    listed = set(expected)
    missing = sorted(actual - listed)
    stale = sorted(listed - actual)
    if missing or stale:
        raise ValueError(
            "manifest file set mismatch: "
            f"unlisted={[path.relative_to(root).as_posix() for path in missing]}, "
            f"missing={[path.relative_to(root).as_posix() for path in stale]}"
        )
    mismatched = [
        path.relative_to(root).as_posix()
        for path, digest in expected.items()
        if sha256_file(path) != digest
    ]
    if mismatched:
        raise ValueError("result SHA256 mismatch: " + ", ".join(mismatched))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the existing manifest instead of rebuilding it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        verify_manifest(args.results)
        print(f"verified {args.results / MANIFEST_NAME}")
    else:
        print(build_manifest(args.results))


if __name__ == "__main__":
    main()
