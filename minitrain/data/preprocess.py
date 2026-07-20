"""End-to-end raw-document to memory-mapped token-shard preprocessing.

The persisted pipeline is::

    read -> clean/dedup -> deterministic split -> chunk -> tokenize
         -> boundary-prefix -> atomic token shards + document index -> manifest

Shards are storage boundaries, while ``documents.idx`` preserves logical
document spans for epoch-level document repacking in the DataLoader.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from minitrain.data.documents import (
    CleaningConfig,
    DocumentCleaner,
    chunk_document,
    iter_documents,
)
from minitrain.data.tokenizer import Tokenizer


TOKEN_MANIFEST = "manifest.json"
TOKEN_MANIFEST_VERSION = 1


def token_storage_dtype(vocab_size: int) -> np.dtype:
    """Choose the smallest explicit little-endian integer covering all IDs."""

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if vocab_size <= 2**16:
        return np.dtype("<u2")
    if vocab_size <= 2**32:
        return np.dtype("<u4")
    raise ValueError("vocab_size exceeds uint32 token storage")


def deterministic_split(document_id: str, *, validation_fraction: float, seed: int) -> str:
    """Map a stable document ID to a split independently of input order."""

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    digest = hashlib.sha256(f"{seed}:{document_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if bucket < validation_fraction else "train"


@dataclass
class ShardRecord:
    path: str
    num_tokens: int
    num_bytes: int
    sha256: str


@dataclass
class DocumentIndexRecord:
    path: str
    entries: int
    dtype: str
    columns: list[str]
    num_bytes: int
    sha256: str


class TokenShardWriter:
    """Incrementally write bounded raw token shards using atomic finalization.

    Normal documents remain intact at physical shard boundaries.  Only a
    single document larger than ``max_tokens`` is split, and every continuation
    receives a fresh boundary token.  ``document_spans`` always addresses the
    logical split-wide token stream rather than offsets local to one file.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        split: str,
        dtype: np.dtype,
        max_tokens: int,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.output_dir = output_dir
        self.split = split
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.split_dir = output_dir / split
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[ShardRecord] = []
        self._handle: BinaryIO | None = None
        self._temp_path: Path | None = None
        self._final_path: Path | None = None
        self._tokens_in_shard = 0
        self.total_tokens = 0
        self.document_spans: list[tuple[int, int]] = []

    def _open(self) -> None:
        # A final .bin name is never visible while bytes are still being
        # produced; _finalize publishes it in one same-directory rename.
        shard_index = len(self.records)
        self._final_path = self.split_dir / f"shard_{shard_index:05d}.bin"
        self._temp_path = self._final_path.with_suffix(".bin.tmp")
        self._handle = self._temp_path.open("wb")
        self._tokens_in_shard = 0

    def _write_contiguous(self, token_ids: list[int]) -> None:
        """Append one already-sized span using the manifest storage dtype."""

        if not token_ids:
            return
        if len(token_ids) > self.max_tokens - self._tokens_in_shard:
            raise ValueError("contiguous token sequence does not fit current shard")
        if max(token_ids) >= 2 ** (8 * self.dtype.itemsize):
            raise ValueError(f"token id does not fit {self.dtype.name}")
        if min(token_ids) < 0:
            raise ValueError("token ids must be non-negative")
        if self._handle is None:
            self._open()
        values = np.asarray(token_ids, dtype=self.dtype)
        assert self._handle is not None
        self._handle.write(values.tobytes(order="C"))
        self._tokens_in_shard += len(token_ids)
        self.total_tokens += len(token_ids)
        if self._tokens_in_shard == self.max_tokens:
            self._finalize()

    def write_document(self, token_ids: list[int], *, boundary_token_id: int) -> int:
        """Write one boundary-prefixed chunk without ordinary mid-document splits."""

        if not token_ids or token_ids[0] != boundary_token_id:
            raise ValueError("document token sequence must start with boundary_token_id")
        # Close early rather than split a normal document between two files.
        # Unused capacity is not lost data; the next document starts a new file.
        remaining = self.max_tokens - self._tokens_in_shard
        if len(token_ids) <= self.max_tokens:
            if self._tokens_in_shard and len(token_ids) > remaining:
                self._finalize()
            start = self.total_tokens
            self._write_contiguous(token_ids)
            self.document_spans.append((start, len(token_ids)))
            return len(token_ids)

        # A single chunk can still exceed the shard budget. Each continuation
        # begins with BOS so every physical shard starts from explicit context.
        if self._tokens_in_shard:
            self._finalize()
        payload = token_ids[1:]
        written = 0
        while payload:
            count = self.max_tokens - 1
            segment = [boundary_token_id, *payload[:count]]
            start = self.total_tokens
            self._write_contiguous(segment)
            self.document_spans.append((start, len(segment)))
            written += len(segment)
            payload = payload[count:]
            if payload and self._tokens_in_shard:
                self._finalize()
        return written

    def _finalize(self) -> None:
        if self._handle is None or self._tokens_in_shard == 0:
            return
        # Commit protocol: Python buffer -> OS -> durable storage -> atomic
        # publication.  The checksum is streamed afterwards, so shard size does
        # not become peak RAM usage.
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        assert self._temp_path is not None and self._final_path is not None
        self._temp_path.replace(self._final_path)
        hasher = hashlib.sha256()
        with self._final_path.open("rb") as shard:
            for block in iter(lambda: shard.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
        self.records.append(
            ShardRecord(
                path=self._final_path.relative_to(self.output_dir).as_posix(),
                num_tokens=self._tokens_in_shard,
                num_bytes=self._final_path.stat().st_size,
                sha256=digest,
            )
        )
        self._handle = None
        self._temp_path = None
        self._final_path = None
        self._tokens_in_shard = 0

    def close(self) -> None:
        self._finalize()

    def write_document_index(self) -> DocumentIndexRecord:
        """Persist global ``(token_offset, token_length)`` pairs atomically."""

        # Fixed-width little-endian u64 pairs make the index portable and allow
        # randomized packing to locate documents without parsing raw JSONL.
        final_path = self.split_dir / "documents.idx"
        temporary = final_path.with_suffix(".idx.tmp")
        values = np.asarray(self.document_spans, dtype="<u8").reshape(-1, 2)
        with temporary.open("wb") as handle:
            handle.write(values.tobytes(order="C"))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(final_path)
        hasher = hashlib.sha256()
        with final_path.open("rb") as index_file:
            for block in iter(lambda: index_file.read(1024 * 1024), b""):
                hasher.update(block)
        return DocumentIndexRecord(
            path=final_path.relative_to(self.output_dir).as_posix(),
            entries=len(self.document_spans),
            dtype="<u8",
            columns=["token_offset", "token_length"],
            num_bytes=final_path.stat().st_size,
            sha256=hasher.hexdigest(),
        )


def tokenizer_training_texts(
    paths: list[str | Path],
    *,
    text_field: str,
    cleaning: CleaningConfig,
    max_document_chars: int,
    max_training_chars: int,
):
    """Yield cleaned chunks until a deterministic character budget is reached."""

    # This path feeds BPE training only.  It intentionally applies the same
    # cleaner/chunker as corpus tokenization, then enforces a character budget
    # without first materializing the full training corpus.
    cleaner = DocumentCleaner(cleaning)
    characters = 0
    for document in iter_documents(paths, text_field=text_field):
        cleaned = cleaner.clean(document)
        if cleaned is None:
            continue
        for chunk in chunk_document(cleaned, max_chars=max_document_chars):
            remaining = max_training_chars - characters
            if remaining <= 0:
                return
            text = chunk.text[:remaining]
            if text:
                yield text
                characters += len(text)


def prepare_token_shards(
    paths: list[str | Path],
    *,
    output_dir: str | Path,
    tokenizer: Tokenizer,
    text_field: str = "text",
    cleaning: CleaningConfig | None = None,
    max_document_chars: int = 100_000,
    max_shard_tokens: int = 10_000_000,
    validation_fraction: float = 0.01,
    split_seed: int = 42,
) -> Path:
    """Clean, chunk, tokenize, split, and persist a reproducible corpus."""

    # Stage 1: construct split writers with one dtype derived from the tokenizer
    # vocabulary.  Every shard in this corpus therefore shares one manifest
    # interpretation (<u2 for <=65,536 tokens, otherwise <u4).
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaning = cleaning or CleaningConfig()
    cleaner = DocumentCleaner(cleaning)
    dtype = token_storage_dtype(tokenizer.vocab_size)
    writers = {
        split: TokenShardWriter(
            output_dir,
            split=split,
            dtype=dtype,
            max_tokens=max_shard_tokens,
        )
        for split in ("train", "validation")
    }
    split_stats = {split: {"documents": 0, "chunks": 0, "tokens": 0} for split in writers}

    # Stage 2: stream documents through normalization, stable-ID splitting,
    # non-overlapping chunking, tokenization, and boundary insertion.
    for document in iter_documents(paths, text_field=text_field):
        cleaned = cleaner.clean(document)
        if cleaned is None:
            continue
        split = deterministic_split(
            cleaned.id,
            validation_fraction=validation_fraction,
            seed=split_seed,
        )
        split_stats[split]["documents"] += 1
        for chunk in chunk_document(cleaned, max_chars=max_document_chars):
            tokens = [tokenizer.boundary_token_id, *tokenizer.encode(chunk.text)]
            if max(tokens, default=0) >= tokenizer.vocab_size:
                raise ValueError("tokenizer emitted an id outside its declared vocabulary")
            written = writers[split].write_document(
                tokens,
                boundary_token_id=tokenizer.boundary_token_id,
            )
            split_stats[split]["chunks"] += 1
            split_stats[split]["tokens"] += written

    # Stage 3: publish any open shards, then write a split-wide document index.
    # Index entries cover every stored token exactly once and remain useful even
    # when an entry crosses the conceptual packing grid used during training.
    for writer in writers.values():
        writer.close()
    document_indices = {split: writers[split].write_document_index() for split in writers}

    # Stage 4: the manifest is the commit record tying tokenizer identity,
    # cleaning/split policy, binary dtype, shard checksums, and indices together.
    manifest = {
        "format_version": TOKEN_MANIFEST_VERSION,
        "dtype": dtype.str,
        "bytes_per_token": dtype.itemsize,
        "vocab_size": tokenizer.vocab_size,
        "boundary_token_id": tokenizer.boundary_token_id,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "document_boundary": "prepend boundary_token_id to every document chunk",
        "chunking": {"max_document_chars": max_document_chars, "overlap_chars": 0},
        "cleaning": cleaning.to_dict(),
        "cleaning_stats": cleaner.stats.to_dict(),
        "splitting": {
            "algorithm": "sha256(seed:document_id)",
            "seed": split_seed,
            "validation_fraction": validation_fraction,
        },
        "splits": {
            split: {
                **split_stats[split],
                "shards": [asdict(record) for record in writers[split].records],
                "document_index": asdict(document_indices[split]),
            }
            for split in writers
        },
    }
    manifest_path = output_dir / TOKEN_MANIFEST
    # Publish the manifest last.  Consumers therefore see either the previous
    # complete dataset contract or this complete one, never a half-written JSON.
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    temporary.replace(manifest_path)
    return manifest_path


def load_manifest(path: str | Path) -> tuple[Path, dict[str, object]]:
    path = Path(path)
    manifest_path = path if path.is_file() else path / TOKEN_MANIFEST
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("format_version") != TOKEN_MANIFEST_VERSION:
        raise ValueError(f"unsupported token manifest version: {manifest.get('format_version')}")
    return manifest_path, manifest
