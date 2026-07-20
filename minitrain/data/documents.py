"""Streaming document readers, deterministic cleaning, and text chunking.

The module deliberately keeps the raw-text stages separate::

    files -> Document -> normalized/filtered Document -> non-overlapping chunks

Tokenization happens later.  Keeping this boundary explicit makes cleaning and
split decisions auditable without involving tokenizer-dependent behavior.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str
    metadata: dict[str, object] = field(default_factory=dict)


def iter_documents(paths: Iterable[str | Path], *, text_field: str = "text") -> Iterator[Document]:
    """Yield documents from UTF-8 txt, JSONL, or optional Parquet inputs."""

    # Every input format is normalized to the same small record.  Stable IDs
    # matter because the downstream train/validation split hashes Document.id.
    for raw_path in paths:
        path = Path(raw_path)
        suffix = path.suffix.lower()
        source = str(path.resolve())
        if suffix in {".txt", ".text", ".md"}:
            yield Document(id=f"{source}:0", text=path.read_text("utf-8"), source=source)
        elif suffix in {".jsonl", ".jsonlines"}:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = row.get(text_field)
                    if not isinstance(text, str):
                        raise ValueError(
                            f"{path}:{line_number} field {text_field!r} must be a string"
                        )
                    doc_id = str(row.get("id", f"{source}:{line_number}"))
                    metadata = {key: value for key, value in row.items() if key != text_field}
                    yield Document(id=doc_id, text=text, source=source, metadata=metadata)
        elif suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as error:
                raise RuntimeError("Parquet input requires `pip install -e .[data]`") from error
            parquet = pq.ParquetFile(path)
            row_offset = 0
            for row_group in range(parquet.num_row_groups):
                table = parquet.read_row_group(row_group, columns=[text_field])
                for index, text in enumerate(table.column(text_field).to_pylist()):
                    if not isinstance(text, str):
                        raise ValueError(f"{path} row {row_offset + index} is not text")
                    yield Document(
                        id=f"{source}:{row_offset + index}",
                        text=text,
                        source=source,
                        metadata={"row_group": row_group},
                    )
                row_offset += table.num_rows
        else:
            raise ValueError(f"unsupported document input format: {path}")


@dataclass(frozen=True)
class CleaningConfig:
    enabled: bool = True
    min_chars: int = 32
    min_alpha_ratio: float = 0.05
    max_control_ratio: float = 0.01
    max_repeated_line_fraction: float = 0.5
    exact_dedup: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class CleaningStats:
    seen: int = 0
    accepted: int = 0
    rejected_empty: int = 0
    rejected_short: int = 0
    rejected_alpha_ratio: int = 0
    rejected_control_ratio: int = 0
    rejected_repetition: int = 0
    rejected_duplicate: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class DocumentCleaner:
    """Small, auditable baseline cleaner; not a substitute for full FineWeb."""

    _horizontal_space = re.compile(r"[^\S\r\n]+")
    _blank_lines = re.compile(r"\n{3,}")

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self.config = config or CleaningConfig()
        self.stats = CleaningStats()
        self._seen_hashes: set[str] = set()

    def clean(self, document: Document) -> Document | None:
        """Normalize and filter one document, returning ``None`` when rejected."""

        self.stats.seen += 1
        if not self.config.enabled:
            self.stats.accepted += 1
            return document

        # Stage 1: canonicalize platform/Unicode representation.  The control
        # ratio is measured before removal so a control-heavy document cannot
        # become deceptively clean merely because those bytes were discarded.
        original = document.text.replace("\r\n", "\n").replace("\r", "\n")
        control_count = sum(
            1 for char in original if unicodedata.category(char) == "Cc" and char not in "\n\t"
        )
        control_ratio = control_count / max(1, len(original))
        if control_ratio > self.config.max_control_ratio:
            self.stats.rejected_control_ratio += 1
            return None

        text = "".join(
            char
            for char in unicodedata.normalize("NFC", original)
            if unicodedata.category(char) != "Cc" or char in "\n\t"
        )
        text = self._horizontal_space.sub(" ", text)
        text = "\n".join(line.strip() for line in text.splitlines())
        text = self._blank_lines.sub("\n\n", text).strip()

        # Stage 2: cheap, deterministic quality gates.  Each rejection reason
        # owns a counter so a generated manifest explains what was removed.
        if not text:
            self.stats.rejected_empty += 1
            return None
        if len(text) < self.config.min_chars:
            self.stats.rejected_short += 1
            return None

        visible = [char for char in text if not char.isspace()]
        alpha_ratio = sum(char.isalpha() for char in visible) / max(1, len(visible))
        if alpha_ratio < self.config.min_alpha_ratio:
            self.stats.rejected_alpha_ratio += 1
            return None

        lines = [line.casefold() for line in text.splitlines() if line.strip()]
        if len(lines) >= 4:
            counts = Counter(lines)
            repeated = sum(count - 1 for count in counts.values() if count > 1)
            if repeated / len(lines) > self.config.max_repeated_line_fraction:
                self.stats.rejected_repetition += 1
                return None

        # Stage 3: exact dedup is intentionally after normalization.  CRLF/LF
        # and canonically equivalent Unicode therefore hash to the same bytes;
        # near-duplicate or semantic dedup is outside this baseline cleaner.
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if self.config.exact_dedup and digest in self._seen_hashes:
            self.stats.rejected_duplicate += 1
            return None
        self._seen_hashes.add(digest)
        self.stats.accepted += 1
        return Document(
            id=document.id,
            text=text,
            source=document.source,
            metadata={**document.metadata, "clean_sha256": digest},
        )


def chunk_document(document: Document, *, max_chars: int) -> Iterator[Document]:
    """Split a long document without overlap, preferring natural boundaries."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    text = document.text
    if len(text) <= max_chars:
        yield document
        return

    start = 0
    chunk_index = 0
    while start < len(text):
        # Search only inside the current maximum-size window.  A boundary in
        # the latter half is preferred; otherwise use the hard character cap.
        # Advancing start=stop means chunks never overlap training targets.
        hard_stop = min(start + max_chars, len(text))
        stop = hard_stop
        if hard_stop < len(text):
            window = text[start:hard_stop]
            candidates = (window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
            natural = max(candidates)
            if natural >= max_chars // 2:
                stop = start + natural + 1
        chunk = text[start:stop].strip()
        if chunk:
            yield Document(
                id=f"{document.id}#chunk-{chunk_index:05d}",
                text=chunk,
                source=document.source,
                metadata={
                    **document.metadata,
                    "parent_id": document.id,
                    "chunk_index": chunk_index,
                    "char_start": start,
                    "char_stop": stop,
                },
            )
            chunk_index += 1
        start = stop
