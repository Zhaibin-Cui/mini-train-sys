"""Turn token storage into deterministic causal-LM batches.

There are two independent decisions in this module:

* the dataset maps logical token spans to tensors, crossing shard files when
  necessary;
* the sampler chooses an epoch order and partitions it across distributed ranks.

Keeping them separate lets persistent workers remain read-only while the main
process changes shuffle order from one epoch to the next.
"""

from __future__ import annotations

import os
from functools import partial
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from minitrain.data.preprocess import load_manifest
from minitrain.runtime.config import DataConfig


# ---- Dataset definitions -------------------------------------------------


@dataclass(frozen=True)
class PackedBlockSpec:
    """Global token spans that form one packed causal-LM block."""

    segments: tuple[tuple[int, int], ...]


class TokenBlockDataset(Dataset[dict[str, torch.Tensor]]):
    """Turn one token stream into non-overlapping causal training blocks.

    If `seq_len=4` and the token stream is `[10, 11, 12, 13, 14]`, the model
    sees `input_ids=[10, 11, 12, 13]` and learns to predict
    `targets=[11, 12, 13, 14]`. This is the standard causal language-model
    training shape used by nanoGPT/nanochat-style pretraining loops.
    """

    def __init__(self, tokens: torch.Tensor, seq_len: int) -> None:
        self.tokens = tokens.long()
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, (self.tokens.numel() - 1) // self.seq_len)

    @property
    def sample_ranges(self) -> tuple[tuple[int, int], ...]:
        return ((0, len(self)),) if len(self) else ()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        # Stride T, not stride 1.  Neighboring T+1 raw windows share only the
        # one boundary token needed for next-token input/target alignment.
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        return {"input_ids": chunk[:-1], "targets": chunk[1:]}

class ShardedTokenBlockDataset(Dataset[dict[str, torch.Tensor]]):
    """Memory-map one logical token stream as non-overlapping causal blocks.

    Physical shards are only an I/O boundary. A block may read across adjacent
    shards, which prevents every shard from permanently losing its short tail.
    Only the final incomplete block of the entire split is omitted.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        seq_len: int,
        max_open_shards: int = 8,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if max_open_shards <= 0:
            raise ValueError("max_open_shards must be positive")
        self.manifest_path, self.manifest = load_manifest(manifest_path)
        self.root = self.manifest_path.parent
        self.dtype = np.dtype(self.manifest["dtype"])
        split_data = self.manifest["splits"].get(split)
        if split_data is None:
            raise ValueError(f"manifest has no split {split!r}")
        self.seq_len = seq_len
        self.max_open_shards = max_open_shards
        # Build a split-wide coordinate system.  _cumulative_tokens maps any
        # global token offset back to its physical shard via binary search.
        self.shards = []
        self._memory_maps: OrderedDict[int, np.memmap] = OrderedDict()
        self._cumulative_tokens = []
        self._sample_ranges = []
        total_tokens = 0
        for record in split_data["shards"]:
            path = self.root / record["path"]
            num_tokens = int(record["num_tokens"])
            expected_bytes = num_tokens * self.dtype.itemsize
            if not path.exists() or path.stat().st_size != expected_bytes:
                raise ValueError(f"token shard size mismatch: {path}")
            if num_tokens <= 0:
                continue
            token_start = total_tokens
            total_tokens += num_tokens
            self._cumulative_tokens.append(total_tokens)

            # Keep the existing three-field shard metadata shape. The final
            # field is filled below with the global block starts owned here.
            self.shards.append((path, num_tokens, 0))

        # A causal sample needs T+1 raw tokens and advances by T.  Group sample
        # indices by the shard containing their start only for I/O-aware shuffle;
        # the actual read is still allowed to cross the next shard boundary.
        self.total_tokens = total_tokens
        self._num_samples = max(0, (total_tokens - 1) // seq_len)
        for shard_index, token_stop in enumerate(self._cumulative_tokens):
            token_start = self._cumulative_tokens[shard_index - 1] if shard_index else 0
            sample_start = min(self._num_samples, (token_start + seq_len - 1) // seq_len)
            sample_stop = min(self._num_samples, (token_stop + seq_len - 1) // seq_len)
            path, num_tokens, _ = self.shards[shard_index]
            self.shards[shard_index] = (
                path,
                num_tokens,
                max(0, sample_stop - sample_start),
            )
            if sample_start < sample_stop:
                self._sample_ranges.append((sample_start, sample_stop))

    def __len__(self) -> int:
        return self._num_samples

    @property
    def sample_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return global sample-index ranges grouped by physical shard."""

        return tuple(self._sample_ranges)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        block = self._read_segments(((index * self.seq_len, self.seq_len + 1),))
        tensor = torch.from_numpy(block)
        return {"input_ids": tensor[:-1], "targets": tensor[1:]}

    def _read_segments(self, segments: Sequence[tuple[int, int]]) -> np.ndarray:
        """Copy one logical block from one or more global token spans."""

        # ``segments`` has one span for contiguous packing and often several
        # spans for randomized-document packing.  Each logical span may itself
        # cross physical files, so the inner loop clips one shard at a time.
        block = np.empty(sum(length for _, length in segments), dtype=np.int64)
        copied = 0
        for segment_start, segment_length in segments:
            segment_copied = 0
            while segment_copied < segment_length:
                global_offset = segment_start + segment_copied
                shard_index = bisect_right(self._cumulative_tokens, global_offset)
                previous = self._cumulative_tokens[shard_index - 1] if shard_index else 0
                local_offset = global_offset - previous
                _, num_tokens, _ = self.shards[shard_index]
                copy_count = min(segment_length - segment_copied, num_tokens - local_offset)
                tokens = self._open_shard(shard_index)
                block[copied : copied + copy_count] = tokens[
                    local_offset : local_offset + copy_count
                ]
                copied += copy_count
                segment_copied += copy_count
        return block

    def _open_shard(self, shard_index: int) -> np.memmap:
        path, num_tokens, _ = self.shards[shard_index]
        tokens = self._memory_maps.get(shard_index)
        if tokens is None:
            # The LRU is per Dataset instance (and therefore per DataLoader
            # worker).  It bounds open mappings without loading shard bytes into
            # RAM; the OS still pages touched regions on demand.
            tokens = np.memmap(path, mode="r", dtype=self.dtype, shape=(num_tokens,))
            self._memory_maps[shard_index] = tokens
            if len(self._memory_maps) > self.max_open_shards:
                _, stale = self._memory_maps.popitem(last=False)
                mmap = getattr(stale, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        else:
            self._memory_maps.move_to_end(shard_index)
        return tokens

class RandomizedDocumentBlockDataset(ShardedTokenBlockDataset):
    """Read fixed-size blocks from an epoch-specific ordering of documents.

    ``documents.idx`` stores physical spans.  The sampler supplies a logical
    block as a list of pieces from those spans; no shard file is rewritten when
    a new epoch chooses a new document order.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        seq_len: int,
        max_open_shards: int = 8,
    ) -> None:
        super().__init__(
            manifest_path,
            split=split,
            seq_len=seq_len,
            max_open_shards=max_open_shards,
        )
        split_data = self.manifest["splits"][split]
        metadata = split_data.get("document_index")
        if metadata is None:
            raise ValueError(
                "data.packing='randomized_documents' requires a manifest with "
                "a document_index; regenerate token shards"
            )
        index_path = self.root / metadata["path"]
        entries = int(metadata["entries"])
        index_dtype = np.dtype(metadata["dtype"])
        expected_bytes = entries * 2 * index_dtype.itemsize
        if metadata.get("columns") != ["token_offset", "token_length"]:
            raise ValueError("unsupported document index columns")
        if (
            index_dtype != np.dtype("<u8")
            or not index_path.exists()
            or index_path.stat().st_size != expected_bytes
        ):
            raise ValueError(f"document index size or dtype mismatch: {index_path}")
        # The index is small fixed-width metadata (16 bytes/document), so load
        # it once while the much larger token payload remains memory-mapped.
        values = np.fromfile(index_path, dtype=index_dtype).reshape(entries, 2)
        self.document_spans = tuple((int(offset), int(length)) for offset, length in values)
        self._validate_document_spans()

    def _validate_document_spans(self) -> None:
        expected_offset = 0
        for offset, length in self.document_spans:
            if offset != expected_offset or length <= 0:
                raise ValueError("document index must contain positive contiguous spans")
            expected_offset += length
        if expected_offset != self.total_tokens:
            raise ValueError("document index does not cover the complete token split")

    def __getitem__(self, spec: PackedBlockSpec) -> dict[str, torch.Tensor]:
        if not isinstance(spec, PackedBlockSpec):
            raise TypeError("randomized document packing requires PackedBlockSpec indices")
        if sum(length for _, length in spec.segments) != self.seq_len + 1:
            raise ValueError("packed block spec must contain exactly seq_len + 1 tokens")
        block = self._read_segments(spec.segments)
        tensor = torch.from_numpy(block)
        return {"input_ids": tensor[:-1], "targets": tensor[1:]}


# ---- Sampler definitions -------------------------------------------------


class ShardAwareBlockSampler(Sampler[int]):
    """Shuffle shard order and bounded block windows deterministically.

    A global permutation over every block causes random reads across the whole
    corpus. This sampler keeps I/O bounded: each epoch shuffles physical shards,
    then shuffles windows and block indices only within one shard. Distributed
    ranks take equal, disjoint positions from the same deterministic stream, so
    every rank executes the same number of optimizer steps without duplicating
    training blocks.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle_window: int = 1024,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        if shuffle_window <= 0:
            raise ValueError("shuffle_window must be positive")
        ranges = getattr(dataset, "sample_ranges", None)
        if ranges is None:
            ranges = ((0, len(dataset)),) if len(dataset) else ()
        self.sample_ranges: tuple[tuple[int, int], ...] = tuple(ranges)
        self.num_samples = len(dataset) // world_size
        self.total_size = self.num_samples * world_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle_window = shuffle_window
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _permutation(length: int, generator: torch.Generator) -> Sequence[int]:
        return torch.randperm(length, generator=generator).tolist()

    def _global_indices(self, generator: torch.Generator) -> Iterator[int]:
        shard_order: Sequence[int] = range(len(self.sample_ranges))
        if self.shuffle:
            shard_order = self._permutation(len(self.sample_ranges), generator)

        # Three bounded levels of ordering: shard-owned ranges, windows within
        # a range, then block indices within a window.  Token order inside a
        # block never changes.
        for shard_index in shard_order:
            start, stop = self.sample_ranges[shard_index]
            if not self.shuffle:
                yield from range(start, stop)
                continue

            windows = [
                (window_start, min(window_start + self.shuffle_window, stop))
                for window_start in range(start, stop, self.shuffle_window)
            ]
            for window_index in self._permutation(len(windows), generator):
                window_start, window_stop = windows[window_index]
                for offset in self._permutation(window_stop - window_start, generator):
                    yield window_start + offset

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        # Partition positions in the common deterministic order, rather than
        # independently shuffling each rank and risking duplicates.
        for position, index in enumerate(self._global_indices(generator)):
            if position >= self.total_size:
                break
            if position % self.world_size == self.rank:
                yield index


class RandomizedDocumentSampler(Sampler[PackedBlockSpec]):
    """Reorder complete documents each epoch, then pack a continuous token stream.

    Stored documents retain their internal token order and boundary token. Only
    their order changes. Fixed-size blocks may cut a document, matching the
    packed 512-token BIO pretraining setup while preventing one permanently cut
    subset across every epoch.
    """

    def __init__(
        self,
        dataset: RandomizedDocumentBlockDataset,
        *,
        shuffle: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        self.dataset = dataset
        self.num_samples = len(dataset) // world_size
        self.total_size = self.num_samples * world_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _ordered_spans(self) -> list[tuple[int, int]]:
        if not self.shuffle:
            return list(self.dataset.document_spans)
        # One without-replacement permutation per epoch.  seed+epoch makes the
        # plan reproducible on every rank while changing which documents meet
        # the fixed block grid in successive epochs.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.dataset.document_spans), generator=generator).tolist()
        return [self.dataset.document_spans[index] for index in order]

    def __iter__(self) -> Iterator[PackedBlockSpec]:
        ordered_spans = self._ordered_spans()

        # Prefix sums convert positions in the newly ordered logical stream to
        # (document index, offset within document) in O(log num_documents).
        stream_stops = []
        total = 0
        for _, length in ordered_spans:
            total += length
            stream_stops.append(total)

        # All ranks enumerate the same global block plan and retain disjoint
        # modulo positions.  total_size truncation gives every rank equal work.
        for block_index in range(self.total_size):
            if block_index % self.world_size != self.rank:
                continue
            stream_offset = block_index * self.dataset.seq_len
            document_index = bisect_right(stream_stops, stream_offset)
            previous_stop = stream_stops[document_index - 1] if document_index else 0
            document_offset = stream_offset - previous_stop
            remaining = self.dataset.seq_len + 1
            segments = []
            # Convert this T+1 logical window back into one or more immutable
            # physical spans.  Crossing a document or shard requires no padding.
            while remaining:
                token_offset, document_length = ordered_spans[document_index]
                take = min(remaining, document_length - document_offset)
                segments.append((token_offset + document_offset, take))
                remaining -= take
                document_index += 1
                document_offset = 0
            yield PackedBlockSpec(tuple(segments))


# ---- Runtime and input helpers ------------------------------------------


def _distributed_rank_info() -> tuple[int, int]:
    """Return torchrun rank metadata without requiring an initialized process group."""

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(
            f"RANK must be in [0, WORLD_SIZE), got rank={rank}, world_size={world_size}"
        )
    return rank, world_size


def resolve_data_workers(cfg: DataConfig) -> int:
    """Resolve a per-rank worker count without oversubscribing one host."""

    if cfg.num_workers is not None:
        return cfg.num_workers
    local_world_size = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cpu_count = os.cpu_count() or 1
    usable = max(1, cpu_count - local_world_size)  # reserve one core per trainer rank
    node_budget = min(cfg.worker_budget or usable, usable)
    base, remainder = divmod(node_budget, local_world_size)
    assigned = base + int(local_rank < remainder)
    return min(cfg.max_workers_per_rank, assigned)


def _initialize_data_worker(
    worker_id: int, *, local_rank: int, local_world_size: int, workers_per_rank: int,
    cpu_affinity: bool,
) -> None:
    """Keep loader workers single-threaded and optionally pin them to host CPUs."""

    torch.set_num_threads(1)
    if not cpu_affinity or not hasattr(os, "sched_getaffinity"):
        return
    allowed = sorted(os.sched_getaffinity(0))
    trainer_slots = min(local_world_size, len(allowed))
    slot = trainer_slots + local_rank * workers_per_rank + worker_id
    if slot < len(allowed):
        os.sched_setaffinity(0, {allowed[slot]})


def _load_tokens_from_file(path: str | Path) -> torch.Tensor:
    """Load a flat token tensor from common pre-tokenized file formats."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            # Many preprocessing scripts save {"tokens": tensor}; accepting this
            # shape keeps the dataloader compatible with small pretrain projects.
            payload = payload.get("tokens", payload.get("input_ids"))
        if payload is None:
            raise ValueError(f"{path} did not contain a 'tokens' or 'input_ids' tensor")
        return torch.as_tensor(payload, dtype=torch.long).flatten()
    if suffix == ".npy":
        return torch.from_numpy(np.load(path)).long().flatten()
    if suffix == ".bin":
        # nanoGPT commonly stores uint16 token ids in .bin files. If your vocab
        # is larger than 65535, save .npy/.pt instead so dtype is unambiguous.
        return torch.from_numpy(np.fromfile(path, dtype=np.uint16)).long()
    raise ValueError(f"Unsupported token file format: {path.suffix}")


# ---- DataLoader construction --------------------------------------------


def build_dataloader(
    tokens: torch.Tensor, seq_len: int, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TokenBlockDataset(tokens, seq_len)
    sampler = ShardAwareBlockSampler(dataset, shuffle=shuffle, seed=0)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def _build_training_loader(
    dataset: Dataset,
    cfg: DataConfig,
    *,
    batch_size: int,
    seed: int,
    rank: int,
    world_size: int,
) -> DataLoader:
    sampler = ShardAwareBlockSampler(
        dataset,
        shuffle=cfg.shuffle,
        seed=seed,
        rank=rank,
        world_size=world_size,
        shuffle_window=cfg.shuffle_window,
    )
    return _loader_from_sampler(dataset, cfg, batch_size=batch_size, sampler=sampler)


def _loader_from_sampler(
    dataset: Dataset,
    cfg: DataConfig,
    *,
    batch_size: int,
    sampler: Sampler,
) -> DataLoader:
    if len(sampler) == 0:
        world_size = getattr(sampler, "world_size", 1)
        raise ValueError(f"Dataset has too few complete blocks for world_size={world_size}")
    if cfg.drop_last and len(sampler) < batch_size:
        raise ValueError(
            f"Dataset provides {len(sampler)} blocks per rank, fewer than "
            f"batch_size={batch_size} with data.drop_last=true"
        )

    # Only pass prefetch_factor when multiprocessing is active; PyTorch rejects
    # that option for the synchronous num_workers=0 path.
    num_workers = resolve_data_workers(cfg)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    loader_options = {
        "batch_size": batch_size,
        "sampler": sampler,
        "drop_last": cfg.drop_last,
        "num_workers": num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers and num_workers > 0,
    }
    if num_workers > 0:
        loader_options["prefetch_factor"] = cfg.prefetch_factor
        loader_options["worker_init_fn"] = partial(
            _initialize_data_worker,
            local_rank=local_rank,
            local_world_size=local_world_size,
            workers_per_rank=num_workers,
            cpu_affinity=cfg.worker_cpu_affinity,
        )
    loader = DataLoader(dataset, **loader_options)
    loader.minitrain_worker_budget = num_workers * local_world_size
    return loader


def build_training_dataloader(
    cfg: DataConfig,
    *,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    seed: int,
) -> DataLoader:
    """Build the training dataloader selected by YAML config.

    The caller passes model-derived `seq_len` and `vocab_size` so data loading
    stays compatible with whichever mini-pretrain model config is active.
    """

    # Source selection: random/tokens become one in-memory logical stream;
    # token_shards keep payloads on disk and select a packing-aware dataset.
    if cfg.source == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        tokens = torch.randint(
            low=0,
            high=vocab_size,
            size=(cfg.num_tokens,),
            dtype=torch.long,
            generator=generator,
        )
    elif cfg.source == "tokens":
        if cfg.path is None:
            raise ValueError("data.path is required when data.source='tokens'")
        tokens = _load_tokens_from_file(cfg.path)
    elif cfg.source == "token_shards":
        if cfg.path is None:
            raise ValueError("data.path is required when data.source='token_shards'")
        # Packing changes logical block construction, not the persisted shard
        # bytes.  randomized_documents additionally requires documents.idx.
        dataset_type = (
            RandomizedDocumentBlockDataset
            if cfg.packing == "randomized_documents"
            else ShardedTokenBlockDataset
        )
        dataset = dataset_type(
            cfg.path, split="train", seq_len=seq_len, max_open_shards=cfg.max_open_shards
        )
        manifest_vocab_size = int(dataset.manifest["vocab_size"])
        if manifest_vocab_size > vocab_size:
            raise ValueError(
                f"dataset vocab_size={manifest_vocab_size} exceeds model vocab_size={vocab_size}"
            )
        if (
            cfg.tokenizer_fingerprint is not None
            and cfg.tokenizer_fingerprint != dataset.manifest["tokenizer_fingerprint"]
        ):
            raise ValueError("data tokenizer fingerprint does not match configured tokenizer")
        rank, world_size = _distributed_rank_info()
        # Randomized documents need span-valued sampler indices.  Contiguous
        # packing uses integer block indices and the I/O-bounded shard sampler.
        if cfg.packing == "randomized_documents":
            sampler = RandomizedDocumentSampler(
                dataset,
                shuffle=cfg.shuffle,
                seed=seed,
                rank=rank,
                world_size=world_size,
            )
            return _loader_from_sampler(dataset, cfg, batch_size=batch_size, sampler=sampler)
        return _build_training_loader(
            dataset,
            cfg,
            batch_size=batch_size,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )
    else:
        raise ValueError(f"Unknown data.source: {cfg.source}")

    rank, world_size = _distributed_rank_info()
    min_tokens = seq_len + 1
    if tokens.numel() < min_tokens:
        raise ValueError(
            f"Need at least {min_tokens} tokens per rank for seq_len={seq_len}, "
            f"got {tokens.numel()}"
        )
    dataset = TokenBlockDataset(tokens, seq_len)
    return _build_training_loader(
        dataset,
        cfg,
        batch_size=batch_size,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
