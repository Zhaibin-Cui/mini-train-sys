"""CLI for reproducible raw-text tokenizer and token-shard preparation."""

# ruff: noqa: E402 -- the repository root must be importable for direct CLI use.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minitrain.data.documents import CleaningConfig  # noqa: E402
from minitrain.data.preprocess import (
    load_manifest,
    prepare_token_shards,
    tokenizer_training_texts,
)  # noqa: E402
from minitrain.data.tokenizer import (
    ByteLevelBPETokenizer,
    TiktokenTokenizer,
    load_tokenizer,
)  # noqa: E402


SUPPORTED_INPUT_SUFFIXES = {".txt", ".text", ".md", ".jsonl", ".jsonlines", ".parquet"}


def resolve_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(
                child
                for child in sorted(path.rglob("*"))
                if child.is_file() and child.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
            )
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(path)
    if not paths:
        raise ValueError("no supported input documents found")
    return paths


def cleaning_from_args(args: argparse.Namespace) -> CleaningConfig:
    return CleaningConfig(
        enabled=not args.no_clean,
        min_chars=args.min_chars,
        min_alpha_ratio=args.min_alpha_ratio,
        max_control_ratio=args.max_control_ratio,
        max_repeated_line_fraction=args.max_repeated_line_fraction,
        exact_dedup=not args.no_exact_dedup,
    )


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="+", help="Files or directories (txt/jsonl/parquet)")
    parser.add_argument("--text-field", default="text", help="JSONL/Parquet text column")
    parser.add_argument("--max-document-chars", type=int, default=100_000)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--min-chars", type=int, default=32)
    parser.add_argument("--min-alpha-ratio", type=float, default=0.05)
    parser.add_argument("--max-control-ratio", type=float, default=0.01)
    parser.add_argument("--max-repeated-line-fraction", type=float, default=0.5)
    parser.add_argument("--no-exact-dedup", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train-tokenizer", help="Train a byte-level BPE")
    add_input_arguments(train)
    train.add_argument("--output", required=True)
    train.add_argument("--vocab-size", type=int, default=32_768)
    train.add_argument("--min-frequency", type=int, default=2)
    train.add_argument("--max-training-chars", type=int, default=2_000_000_000)

    existing = commands.add_parser("use-tiktoken", help="Record a named tiktoken artifact")
    existing.add_argument("--encoding", default="gpt2")
    existing.add_argument("--output", required=True)

    tokenize = commands.add_parser("tokenize", help="Build token shards and manifest")
    add_input_arguments(tokenize)
    tokenize.add_argument("--tokenizer", required=True, help="Tokenizer artifact directory")
    tokenize.add_argument("--output", required=True)
    tokenize.add_argument("--max-shard-tokens", type=int, default=10_000_000)
    tokenize.add_argument("--validation-fraction", type=float, default=0.01)
    tokenize.add_argument("--split-seed", type=int, default=42)

    inspect = commands.add_parser("inspect", help="Print a token corpus manifest")
    inspect.add_argument("path")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # The CLI exposes three artifact-producing stages separately so the same
    # tokenizer can be reused for many corpora and each stage is reproducible on
    # its own: train/record tokenizer -> tokenize shards -> inspect manifest.
    if args.command == "train-tokenizer":
        # This generator streams cleaned/chunked text directly into BPE training
        # and stops at max_training_chars without assembling a giant string.
        paths = resolve_inputs(args.inputs)
        texts = tokenizer_training_texts(
            paths,
            text_field=args.text_field,
            cleaning=cleaning_from_args(args),
            max_document_chars=args.max_document_chars,
            max_training_chars=args.max_training_chars,
        )
        tokenizer = ByteLevelBPETokenizer.train_from_iterator(
            texts,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
        metadata_path = tokenizer.save(args.output)
        print(metadata_path)
        print(f"vocab_size={tokenizer.vocab_size} fingerprint={tokenizer.fingerprint}")
    elif args.command == "use-tiktoken":
        # Persist the named encoding and runtime fingerprint as an artifact even
        # though tiktoken itself owns the executable vocabulary.
        tokenizer = TiktokenTokenizer(args.encoding)
        metadata_path = tokenizer.save(args.output)
        print(metadata_path)
        print(f"vocab_size={tokenizer.vocab_size} fingerprint={tokenizer.fingerprint}")
    elif args.command == "tokenize":
        # Corpus construction reuses the recorded tokenizer identity; the
        # resulting manifest links every shard back to that exact fingerprint.
        paths = resolve_inputs(args.inputs)
        tokenizer = load_tokenizer(args.tokenizer)
        manifest_path = prepare_token_shards(
            paths,
            output_dir=args.output,
            tokenizer=tokenizer,
            text_field=args.text_field,
            cleaning=cleaning_from_args(args),
            max_document_chars=args.max_document_chars,
            max_shard_tokens=args.max_shard_tokens,
            validation_fraction=args.validation_fraction,
            split_seed=args.split_seed,
        )
        print(manifest_path)
    else:
        _, manifest = load_manifest(args.path)
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
