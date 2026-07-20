import json

import numpy as np
import pytest
import torch

from minitrain.data.dataloader import (
    RandomizedDocumentBlockDataset,
    RandomizedDocumentSampler,
    ShardAwareBlockSampler,
    ShardedTokenBlockDataset,
    TokenBlockDataset,
    build_training_dataloader,
    resolve_data_workers,
)
from minitrain.data.documents import CleaningConfig, Document, DocumentCleaner, chunk_document
from minitrain.data.preprocess import (
    deterministic_split,
    prepare_token_shards,
    token_storage_dtype,
)
from minitrain.data.tokenizer import ByteLevelBPETokenizer, Tokenizer, load_tokenizer
from minitrain.runtime.config import DataConfig


class ByteTokenizer(Tokenizer):
    @property
    def vocab_size(self):
        return 257

    @property
    def boundary_token_id(self):
        return 256

    @property
    def fingerprint(self):
        return "test-byte-tokenizer"

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(token for token in tokens if token < 256).decode("utf-8")

    def save(self, output_dir):
        raise NotImplementedError


def test_cleaner_normalizes_and_exact_deduplicates():
    cleaner = DocumentCleaner(CleaningConfig(min_chars=3, min_alpha_ratio=0.1))
    first = cleaner.clean(Document("a", " Hello\r\n\r\n\r\nworld ", "memory"))
    duplicate = cleaner.clean(Document("b", "Hello\n\nworld", "memory"))

    assert first is not None and first.text == "Hello\n\nworld"
    assert duplicate is None
    assert cleaner.stats.rejected_duplicate == 1


def test_chunk_document_prefers_boundaries_without_overlap():
    document = Document("doc", "alpha beta gamma delta epsilon", "memory")
    chunks = list(chunk_document(document, max_chars=12))

    assert len(chunks) >= 2
    assert "".join(chunk.text for chunk in chunks).replace(" ", "") == document.text.replace(
        " ", ""
    )
    assert chunks[0].metadata["parent_id"] == "doc"


def test_split_is_stable_and_token_dtype_tracks_vocab():
    assert deterministic_split("doc", validation_fraction=0.2, seed=7) == deterministic_split(
        "doc", validation_fraction=0.2, seed=7
    )
    assert token_storage_dtype(65_536) == np.dtype("<u2")
    assert token_storage_dtype(65_537) == np.dtype("<u4")


def test_token_blocks_are_non_overlapping_and_targets_shift_one_token():
    dataset = TokenBlockDataset(torch.arange(13), seq_len=4)

    assert len(dataset) == 3
    assert dataset[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert dataset[0]["targets"].tolist() == [1, 2, 3, 4]
    assert dataset[1]["input_ids"].tolist() == [4, 5, 6, 7]
    assert dataset[1]["targets"].tolist() == [5, 6, 7, 8]


def test_sharded_blocks_cross_physical_boundaries_without_losing_each_tail(tmp_path):
    shard_values = [np.arange(0, 3), np.arange(3, 5), np.arange(5, 14)]
    shard_records = []
    for index, values in enumerate(shard_values):
        path = tmp_path / f"shard_{index:05d}.bin"
        values.astype("<u2").tofile(path)
        shard_records.append({"path": path.name, "num_tokens": len(values)})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "dtype": "<u2",
                "splits": {"train": {"shards": shard_records}},
            }
        ),
        "utf-8",
    )

    dataset = ShardedTokenBlockDataset(
        manifest_path,
        split="train",
        seq_len=4,
        max_open_shards=1,
    )

    # Manifest metadata is enough to construct block ranges; token files are
    # mapped only when __getitem__ actually requests a block.
    assert not dataset._memory_maps
    assert len(dataset) == 3
    assert dataset.sample_ranges == ((0, 1), (1, 2), (2, 3))
    for index in range(len(dataset)):
        expected = list(range(index * 4, index * 4 + 5))
        assert dataset[index]["input_ids"].tolist() == expected[:-1]
        assert dataset[index]["targets"].tolist() == expected[1:]
        assert len(dataset._memory_maps) <= 1
    with pytest.raises(ValueError, match="document_index"):
        RandomizedDocumentBlockDataset(manifest_path, split="train", seq_len=4)


def test_shard_sampler_is_epoch_deterministic_and_partitions_ranks():
    dataset = TokenBlockDataset(torch.arange(41), seq_len=2)
    samplers = [
        ShardAwareBlockSampler(
            dataset,
            shuffle=True,
            seed=7,
            rank=rank,
            world_size=2,
            shuffle_window=4,
        )
        for rank in range(2)
    ]
    epoch_zero = [list(sampler) for sampler in samplers]

    assert len(epoch_zero[0]) == len(epoch_zero[1]) == 10
    assert set(epoch_zero[0]).isdisjoint(epoch_zero[1])
    assert set(epoch_zero[0]) | set(epoch_zero[1]) == set(range(20))

    for sampler in samplers:
        sampler.set_epoch(1)
    epoch_one = [list(sampler) for sampler in samplers]
    assert epoch_one != epoch_zero
    assert set(epoch_one[0]) | set(epoch_one[1]) == set(range(20))


def test_auto_worker_budget_is_bounded_per_rank(monkeypatch):
    monkeypatch.setattr("minitrain.data.dataloader.os.cpu_count", lambda: 64)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "3")
    cfg = DataConfig(num_workers=None, worker_budget=32, max_workers_per_rank=4)
    assert resolve_data_workers(cfg) == 4
    assert resolve_data_workers(DataConfig(num_workers=2)) == 2


def test_shard_sampler_shuffles_complete_ranges_without_global_interleaving():
    class GroupedBlocks:
        sample_ranges = ((0, 4), (4, 8), (8, 12))

        def __len__(self):
            return 12

    sampler = ShardAwareBlockSampler(
        GroupedBlocks(),
        shuffle=True,
        seed=11,
        shuffle_window=2,
    )
    indices = list(sampler)

    # Each range may reorder its windows and blocks, but is exhausted before
    # the sampler advances to another shard-owned range.
    group_order = []
    for index in indices:
        group = index // 4
        if not group_order or group_order[-1] != group:
            group_order.append(group)
    assert sorted(group_order) == [0, 1, 2]
    assert sorted(indices) == list(range(12))


def test_prepare_manifest_and_dataloader_end_to_end(tmp_path, monkeypatch):
    source = tmp_path / "documents.jsonl"
    rows = [
        {"id": "a", "text": "alpha beta gamma delta"},
        {"id": "b", "text": "epsilon zeta eta theta"},
        {"id": "c", "text": "iota kappa lambda mu"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")
    output_dir = tmp_path / "tokens"
    manifest_path = prepare_token_shards(
        [source],
        output_dir=output_dir,
        tokenizer=ByteTokenizer(),
        cleaning=CleaningConfig(min_chars=1, min_alpha_ratio=0.0),
        max_document_chars=100,
        max_shard_tokens=24,
        validation_fraction=0.0,
    )

    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["dtype"] == "<u2"
    assert manifest["tokenizer_fingerprint"] == "test-byte-tokenizer"
    assert manifest["splits"]["train"]["documents"] == 3
    index_metadata = manifest["splits"]["train"]["document_index"]
    assert index_metadata["entries"] == manifest["splits"]["train"]["chunks"]
    document_index = np.fromfile(
        output_dir / index_metadata["path"], dtype=np.dtype(index_metadata["dtype"])
    ).reshape(-1, 2)
    assert document_index[:, 0].tolist() == np.cumsum(np.r_[0, document_index[:-1, 1]]).tolist()
    assert len(manifest["splits"]["train"]["shards"]) >= 2
    for shard in manifest["splits"]["train"]["shards"]:
        tokens = np.fromfile(output_dir / shard["path"], dtype=np.dtype(manifest["dtype"]))
        assert int(tokens[0]) == 256

    dataset = ShardedTokenBlockDataset(manifest_path, split="train", seq_len=4)
    all_tokens = np.concatenate(
        [
            np.fromfile(output_dir / shard["path"], dtype=np.dtype(manifest["dtype"]))
            for shard in manifest["splits"]["train"]["shards"]
        ]
    )
    assert len(dataset) == (len(all_tokens) - 1) // 4
    for index in range(len(dataset)):
        expected = all_tokens[index * 4 : index * 4 + 5].astype(np.int64).tolist()
        assert dataset[index]["input_ids"].tolist() == expected[:-1]
        assert dataset[index]["targets"].tolist() == expected[1:]
    sample = dataset[0]
    assert sample["input_ids"].shape == (4,)
    assert sample["targets"].tolist() == dataset[0]["input_ids"].tolist()[1:] + [
        sample["targets"][-1].item()
    ]

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    loader = build_training_dataloader(
        DataConfig(
            source="token_shards",
            path=str(manifest_path),
            shuffle=False,
            tokenizer_fingerprint="test-byte-tokenizer",
            num_workers=2,
            pin_memory=False,
            persistent_workers=True,
        ),
        seq_len=4,
        batch_size=2,
        vocab_size=257,
        seed=1,
    )
    batch = next(iter(loader))
    assert isinstance(loader.sampler, ShardAwareBlockSampler)
    assert batch["input_ids"].shape == (2, 4)
    assert batch["targets"].shape == (2, 4)

    randomized_loader = build_training_dataloader(
        DataConfig(
            source="token_shards",
            path=str(manifest_path),
            packing="randomized_documents",
            shuffle=True,
            tokenizer_fingerprint="test-byte-tokenizer",
            num_workers=2,
            pin_memory=False,
            persistent_workers=True,
        ),
        seq_len=4,
        batch_size=2,
        vocab_size=257,
        seed=7,
    )
    randomized_batch = next(iter(randomized_loader))
    assert isinstance(randomized_loader.sampler, RandomizedDocumentSampler)
    assert randomized_batch["input_ids"].shape == (2, 4)
    assert randomized_batch["targets"].shape == (2, 4)


def test_randomized_document_packing_matches_epoch_permutation_and_ddp(
    tmp_path,
):
    documents = [
        np.asarray([99, 1, 2], dtype="<u2"),
        np.asarray([99, 3, 4, 5], dtype="<u2"),
        np.asarray([99, 6], dtype="<u2"),
        np.asarray([99, 7, 8, 9, 10], dtype="<u2"),
        np.asarray([99, 11, 12], dtype="<u2"),
    ]
    stored = np.concatenate(documents)
    shard_lengths = [5, 4, len(stored) - 9]
    shard_records = []
    start = 0
    for index, length in enumerate(shard_lengths):
        path = tmp_path / f"shard_{index:05d}.bin"
        stored[start : start + length].tofile(path)
        shard_records.append({"path": path.name, "num_tokens": length})
        start += length

    offsets = np.cumsum([0, *[len(document) for document in documents[:-1]]])
    index_values = np.asarray(list(zip(offsets, map(len, documents))), dtype="<u8")
    index_path = tmp_path / "documents.idx"
    index_values.tofile(index_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "dtype": "<u2",
                "vocab_size": 100,
                "tokenizer_fingerprint": "test",
                "splits": {
                    "train": {
                        "shards": shard_records,
                        "document_index": {
                            "path": index_path.name,
                            "entries": len(documents),
                            "dtype": "<u8",
                            "columns": ["token_offset", "token_length"],
                        },
                    }
                },
            }
        ),
        "utf-8",
    )
    dataset = RandomizedDocumentBlockDataset(
        manifest_path, split="train", seq_len=3, max_open_shards=1
    )

    ordered_sampler = RandomizedDocumentSampler(dataset, shuffle=False, seed=17)
    for block_index, spec in enumerate(ordered_sampler):
        block = dataset[spec]
        expected = stored[block_index * 3 : block_index * 3 + 4].astype(np.int64).tolist()
        assert block["input_ids"].tolist() == expected[:-1]
        assert block["targets"].tolist() == expected[1:]

    def expected_stream(epoch):
        generator = torch.Generator().manual_seed(17 + epoch)
        order = torch.randperm(len(documents), generator=generator).tolist()
        return np.concatenate([documents[index] for index in order]).astype(np.int64)

    sampler = RandomizedDocumentSampler(dataset, shuffle=True, seed=17)
    epoch_specs = []
    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        specs = list(sampler)
        epoch_specs.append(specs)
        stream = expected_stream(epoch)
        for block_index, spec in enumerate(specs):
            block = dataset[spec]
            expected = stream[block_index * 3 : block_index * 3 + 4].tolist()
            assert block["input_ids"].tolist() == expected[:-1]
            assert block["targets"].tolist() == expected[1:]
    assert epoch_specs[0] != epoch_specs[1]

    ranks = [
        RandomizedDocumentSampler(dataset, shuffle=True, seed=17, rank=rank, world_size=2)
        for rank in range(2)
    ]
    rank_specs = [list(rank_sampler) for rank_sampler in ranks]
    assert len(rank_specs[0]) == len(rank_specs[1])
    assert set(rank_specs[0]).isdisjoint(rank_specs[1])
    assert set(rank_specs[0]) | set(rank_specs[1]) == set(epoch_specs[0][:-1])


def test_custom_bpe_artifact_round_trip(tmp_path):
    pytest.importorskip("tokenizers")
    tokenizer = ByteLevelBPETokenizer.train_from_iterator(
        iter(["hello byte BPE", "你好，字节 tokenizer", "hello again"]),
        vocab_size=300,
        min_frequency=1,
    )
    tokenizer.save(tmp_path)
    restored = load_tokenizer(tmp_path)
    text = "hello 你好"

    assert restored.decode(restored.encode(text)) == text
    assert restored.fingerprint == tokenizer.fingerprint
