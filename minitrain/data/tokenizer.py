"""Tokenizer adapters with reproducible artifacts and stable token identities.

Both supported backends implement one contract::

    cleaned text <-> token ids + boundary id + reproducible fingerprint

The fingerprint is part of the data contract: token shards record it and the
training loader can reject a vocabulary/runtime mismatch before optimization.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any


TOKENIZER_METADATA_FILE = "tokenizer_metadata.json"
CUSTOM_TOKENIZER_FILE = "tokenizer.json"
TOKENIZER_FORMAT_VERSION = 1


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


class Tokenizer(ABC):
    """Stable tokenizer contract used by preprocessing and training tools."""

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def boundary_token_id(self) -> int:
        """Token prepended to every independent document or document chunk."""

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """Identity of the vocabulary, merge rules, specials, and runtime spec."""

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, tokens: list[int]) -> str: ...

    @abstractmethod
    def save(self, output_dir: str | Path) -> Path: ...


class ByteLevelBPETokenizer(Tokenizer):
    """A trained byte-level BPE backed by Hugging Face `tokenizers`."""

    boundary_token = "<|bos|>"

    def __init__(self, backend: Any, *, fingerprint: str | None = None) -> None:
        self._backend = backend
        boundary_id = backend.token_to_id(self.boundary_token)
        if boundary_id is None:
            raise ValueError(f"custom tokenizer is missing {self.boundary_token!r}")
        self._boundary_token_id = int(boundary_id)
        self._fingerprint = fingerprint or self._compute_fingerprint()

    @classmethod
    def train_from_iterator(
        cls,
        text_iterator: Iterable[str],
        *,
        vocab_size: int,
        min_frequency: int = 2,
    ) -> "ByteLevelBPETokenizer":
        if vocab_size < 257:
            raise ValueError("byte-level BPE vocab_size must be at least 257")
        try:
            from tokenizers import Tokenizer as BackendTokenizer
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.trainers import BpeTrainer
        except ImportError as error:
            raise RuntimeError("custom BPE training requires `pip install -e .[data]`") from error

        # Byte-level pre-tokenization starts from all 256 byte values.  Legal
        # UTF-8 text can therefore always fall back to bytes instead of UNK;
        # BPE training only learns frequent adjacent-byte merges on top.
        backend = BackendTokenizer(BPE())
        backend.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
        backend.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[cls.boundary_token],
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=False,
        )
        backend.train_from_iterator(text_iterator, trainer=trainer)
        return cls(backend)

    @property
    def vocab_size(self) -> int:
        return int(self._backend.get_vocab_size(with_added_tokens=True))

    @property
    def boundary_token_id(self) -> int:
        return self._boundary_token_id

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def encode(self, text: str) -> list[int]:
        return list(self._backend.encode(text, add_special_tokens=False).ids)

    def decode(self, tokens: list[int]) -> str:
        return self._backend.decode(tokens, skip_special_tokens=False)

    def _compute_fingerprint(self) -> str:
        # Hash the canonical backend JSON, special-token identity, and format
        # version together.  Vocab size alone cannot detect changed merge IDs.
        payload = {
            "format_version": TOKENIZER_FORMAT_VERSION,
            "kind": "byte_bpe",
            "boundary_token": self.boundary_token,
            "backend_json": json.loads(self._backend.to_str()),
        }
        return hashlib.sha256(_canonical_json(payload)).hexdigest()

    def save(self, output_dir: str | Path) -> Path:
        """Save executable tokenizer state plus small human-readable metadata."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._backend.save(str(output_dir / CUSTOM_TOKENIZER_FILE))
        metadata = {
            "format_version": TOKENIZER_FORMAT_VERSION,
            "kind": "byte_bpe",
            "vocab_size": self.vocab_size,
            "boundary_token": self.boundary_token,
            "boundary_token_id": self.boundary_token_id,
            "fingerprint": self.fingerprint,
        }
        path = output_dir / TOKENIZER_METADATA_FILE
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, tokenizer_dir: str | Path) -> "ByteLevelBPETokenizer":
        """Load an artifact and verify its backend still matches the metadata."""

        tokenizer_dir = Path(tokenizer_dir)
        try:
            from tokenizers import Tokenizer as BackendTokenizer
        except ImportError as error:
            raise RuntimeError("loading custom BPE requires `pip install -e .[data]`") from error
        metadata = json.loads((tokenizer_dir / TOKENIZER_METADATA_FILE).read_text("utf-8"))
        backend = BackendTokenizer.from_file(str(tokenizer_dir / CUSTOM_TOKENIZER_FILE))
        tokenizer = cls(backend, fingerprint=metadata["fingerprint"])
        if tokenizer._compute_fingerprint() != tokenizer.fingerprint:
            raise ValueError("custom tokenizer artifact fingerprint does not match metadata")
        return tokenizer


class TiktokenTokenizer(Tokenizer):
    """Adapter for an existing named tiktoken vocabulary."""

    def __init__(self, encoding_name: str) -> None:
        try:
            import tiktoken
        except ImportError as error:
            raise RuntimeError("tiktoken backend requires `pip install -e .[data]`") from error
        self.encoding_name = encoding_name
        self._backend = tiktoken.get_encoding(encoding_name)
        self._boundary_token_id = int(self._backend.eot_token)
        version = importlib.metadata.version("tiktoken")
        # Named tiktoken encodings are runtime-provided rather than copied into
        # this repository.  Include the runtime version in their identity so a
        # silently changed external encoding cannot reuse old token shards.
        spec = {
            "format_version": TOKENIZER_FORMAT_VERSION,
            "kind": "tiktoken",
            "encoding_name": encoding_name,
            "tiktoken_version": version,
            "vocab_size": self.vocab_size,
            "boundary_token_id": self.boundary_token_id,
        }
        self._runtime_version = version
        self._fingerprint = hashlib.sha256(_canonical_json(spec)).hexdigest()

    @property
    def vocab_size(self) -> int:
        return int(self._backend.n_vocab)

    @property
    def boundary_token_id(self) -> int:
        return self._boundary_token_id

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def encode(self, text: str) -> list[int]:
        return list(self._backend.encode_ordinary(text))

    def decode(self, tokens: list[int]) -> str:
        return self._backend.decode(tokens)

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": TOKENIZER_FORMAT_VERSION,
            "kind": "tiktoken",
            "encoding_name": self.encoding_name,
            "runtime_version": self._runtime_version,
            "vocab_size": self.vocab_size,
            "boundary_token_id": self.boundary_token_id,
            "fingerprint": self.fingerprint,
        }
        path = output_dir / TOKENIZER_METADATA_FILE
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def load_tokenizer(tokenizer_dir: str | Path) -> Tokenizer:
    """Dispatch from metadata and enforce the recorded tokenizer identity."""

    tokenizer_dir = Path(tokenizer_dir)
    metadata = json.loads((tokenizer_dir / TOKENIZER_METADATA_FILE).read_text("utf-8"))
    kind = metadata.get("kind")
    if kind == "byte_bpe":
        return ByteLevelBPETokenizer.load(tokenizer_dir)
    if kind == "tiktoken":
        tokenizer = TiktokenTokenizer(metadata["encoding_name"])
        if tokenizer.fingerprint != metadata["fingerprint"]:
            raise ValueError(
                "tiktoken runtime fingerprint changed; use the recorded runtime version "
                f"{metadata.get('runtime_version')} or rebuild the token shards"
            )
        return tokenizer
    raise ValueError(f"unsupported tokenizer kind: {kind!r}")
