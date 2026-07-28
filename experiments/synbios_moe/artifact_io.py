"""Read, fingerprint, and atomically write SynBioS artifacts."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence


def read_json_object(path: str | Path) -> dict[str, object]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {source}")
    return payload


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_hashed_file(
    recorded_path: str | Path,
    expected_sha256: str,
    *,
    fallbacks: Sequence[str | Path] = (),
    label: str,
) -> Path:
    recorded = Path(recorded_path)
    if recorded.is_file():
        actual = sha256_file(recorded)
        if actual != expected_sha256:
            raise ValueError(
                f"{label} SHA256 mismatch: expected {expected_sha256}, found {actual} at {recorded}"
            )
        return recorded

    checked = [recorded]
    for candidate_value in fallbacks:
        candidate = Path(candidate_value)
        checked.append(candidate)
        if not candidate.is_file():
            continue
        actual = sha256_file(candidate)
        if actual != expected_sha256:
            raise ValueError(
                f"{label} fallback SHA256 mismatch: expected {expected_sha256}, "
                f"found {actual} at {candidate}"
            )
        return candidate
    paths = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(f"missing {label}; checked: {paths}")


def write_json_atomic(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def write_csv_atomic(
    path: str | Path,
    rows: Sequence[dict[str, object]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    destination = Path(path)
    columns = list(fieldnames or (list(rows[0]) if rows else ()))
    if not columns:
        raise ValueError(f"cannot write an empty CSV: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def write_text_atomic(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(destination)


def save_figure_pair(
    figure,
    destination: str | Path,
    *,
    dpi: int,
    facecolor: str | None = None,
) -> None:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight"}
    if facecolor is not None:
        options["facecolor"] = facecolor
    figure.savefig(output.with_suffix(".png"), dpi=dpi, **options)
    figure.savefig(output.with_suffix(".pdf"), **options)
