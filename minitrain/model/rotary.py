from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Build RoPE accurately once and serve allocation-free cache views."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float,
        *,
        cache_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        angles = torch.cat((freqs, freqs), dim=-1)

        # Compute trigonometric values in fp32, then cast once at construction.
        # The derived buffers are excluded from checkpoints and forward only
        # returns slices, so no per-step dtype conversion or allocation occurs.
        self.register_buffer("cos", angles.cos().to(dtype=cache_dtype), persistent=False)
        self.register_buffer("sin", angles.sin().to(dtype=cache_dtype), persistent=False)

    @property
    def max_seq_len(self) -> int:
        return self.cos.size(0)

    def forward(
        self,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds configured max seq_len={self.max_seq_len}"
            )
        return self.cos[:seq_len], self.sin[:seq_len]
