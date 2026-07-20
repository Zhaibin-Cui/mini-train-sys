# Data preprocessing implementation plan

> 历史设计记录：实现已经完成。当前行为和命令以
> [`data_pipeline.md`](../data/data_pipeline.md) 为准。

Status: archived design record.

## Objective and non-goals

Build a learnable but production-shaped path from independent raw documents to
the existing MiniTrain causal-LM dataloader. The path must support either a
locally trained byte-level BPE vocabulary or an existing tiktoken vocabulary,
preserve document boundaries, choose token storage width safely, stream data in
bounded memory, and emit enough metadata to reproduce and validate a run.

This iteration does not pretend to reproduce the full distributed FineWeb
pipeline. In particular, MinHash/LSH near-duplicate removal, language models,
PII anonymization, and educational-quality classifiers require separate models
or distributed infrastructure. The local cleaner will provide deterministic,
auditable baseline rules and exact-document deduplication; its manifest will
name every enabled rule.

## Findings that constrain the design

### nanochat

1. Source documents are stored as a `text` column in zstd Parquet shards, with
   row groups used as the DDP sharding unit.
2. Tokenizer training streams documents, caps each at 10,000 characters, and
   stops after a character budget (2B by default).
3. `rustbpe` trains byte-level merge ranks. nanochat then constructs a
   `tiktoken.Encoding` from those exact ranks and the exact regex. It does not
   exchange the trained vocabulary for GPT-4/GPT-2 at inference.
4. The pretraining dataloader tokenizes text online, prepends BOS to every
   document, and best-fit packs documents into `T+1` rows. If no whole document
   fits, it crops one to reach 100% utilization. This improves BOS alignment but
   intentionally discards tails of some documents.
5. The reference repackager shuffles documents and writes about 250M characters
   per Parquet shard, 1024 rows per row group, zstd level 3.

### MiniTrain

MiniTrain currently accepts random tokens or one flat `.pt/.npy/.bin` stream.
Its stable `Tokenizer` interface is intentionally tiny, and training consumes
`DataConfig` through `build_training_dataloader`. The new implementation should
extend these seams rather than create a second training entry point.

## Architecture

```text
txt / jsonl / parquet documents
  -> readers (stable document id + provenance)
  -> deterministic cleaning + rejection counters
  -> boundary-aware long-document chunking
  -> deterministic train/validation assignment
  -> tokenizer adapter
       |-- trained byte-level BPE artifact
       `-- named tiktoken artifact
  -> prepend one document-boundary token per chunk
  -> bounded token shard writer (uint16 or uint32)
  -> manifest + tokenizer fingerprint + statistics
  -> manifest-aware memory-mapped block dataset
  -> existing MiniTrain DataLoader / training loop
```

The tokenizer artifact and token shards are coupled by a SHA-256 fingerprint.
The dataloader validates dtype, shard sizes, and manifest version; callers can
also compare the manifest fingerprint with the model/checkpoint tokenizer.

## Decisions

- A “document” is the unit of cleaning, exact deduplication, split assignment,
  and boundary-token insertion.
- Long-document chunks never overlap by default. Overlap duplicates training
  targets and would need loss masking to be statistically honest.
- Chunks prefer paragraph/newline/whitespace boundaries, then hard-split only
  when no boundary exists. No separator text is invented.
- Independent documents are not joined as raw strings. They become one token
  stream only after each receives a boundary token, so concatenation is
  reversible at the semantic boundary level.
- Vocabularies with maximum token id below 65,536 use little-endian `uint16`;
  larger vocabularies use little-endian `uint32`. The manifest is authoritative;
  a bare `.bin` remains the legacy uint16 path.
- Token shards do not split a chunk unless a single chunk exceeds the shard
  budget. Oversized chunks are split at token level, with continuation metadata
  counted in the manifest.
- Train/validation assignment hashes the stable document id with a seed. It is
  independent of input order and therefore reproducible under parallel readers.
- The first dataloader implementation uses non-overlapping `T+1` blocks within
  each shard. It never silently joins the tail of one physical shard to another.

## Implementation sequence and verification

1. Add tokenizer protocol/adapters, artifact metadata, and fingerprints.
2. Add document readers, cleaner, chunker, deterministic splitter, shard writer,
   and a CLI with `train-tokenizer`, `tokenize`, and `inspect` commands.
3. Add manifest-aware token loading and a sharded block dataset to the existing
   dataloader/config path without breaking `random` and legacy `tokens` sources.
4. Add hermetic tests over tiny txt/jsonl corpora: cleaning counters, stable
   split, boundary insertion, uint width, manifest validation, mmap blocks, and
   custom/existing tokenizer round trips when optional dependencies exist.
5. Write the final architecture/tutorial document, including BPE mechanics,
   byte accounting, nanochat comparison, exact commands, and operational limits.

Acceptance requires unit tests, a tiny end-to-end preprocessing smoke run, a
batch produced through `build_training_dataloader`, and inspection of the final
manifest. Large external datasets are not downloaded as part of the test.
