import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from minitrain.runtime.config import DataConfig


class TokenBlockDataset(Dataset[dict[str, torch.Tensor]]):
    """Turn one long token stream into next-token-prediction samples.

    If `seq_len=4` and the token stream is `[10, 11, 12, 13, 14]`, the model
    sees `input_ids=[10, 11, 12, 13]` and learns to predict
    `targets=[11, 12, 13, 14]`. This is the standard causal language-model
    training shape used by nanoGPT/nanochat-style pretraining loops.
    """

    def __init__(self, tokens: torch.Tensor, seq_len: int) -> None:
        self.tokens = tokens.long()
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, self.tokens.numel() - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return {"input_ids": chunk[:-1], "targets": chunk[1:]}


def build_dataloader(tokens: torch.Tensor, seq_len: int, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TokenBlockDataset(tokens, seq_len), batch_size=batch_size, shuffle=shuffle)


def _distributed_rank_info() -> tuple[int, int]:
    """Return torchrun rank metadata without requiring an initialized process group."""

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"RANK must be in [0, WORLD_SIZE), got rank={rank}, world_size={world_size}")
    return rank, world_size


def _shard_tokens(tokens: torch.Tensor, *, rank: int, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tokens

    base = tokens.numel() // world_size
    extra = tokens.numel() % world_size
    start = rank * base + min(rank, extra)
    length = base + (1 if rank < extra else 0)
    return tokens[start : start + length]


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
    else:
        raise ValueError(f"Unknown data.source: {cfg.source}")

    rank, world_size = _distributed_rank_info()
    tokens = _shard_tokens(tokens, rank=rank, world_size=world_size)

    min_tokens = seq_len + 1
    if tokens.numel() < min_tokens:
        raise ValueError(
            f"Need at least {min_tokens} tokens per rank for seq_len={seq_len}, "
            f"got {tokens.numel()} on rank {rank}/{world_size}"
        )
    return build_dataloader(tokens, seq_len=seq_len, batch_size=batch_size, shuffle=cfg.shuffle)
