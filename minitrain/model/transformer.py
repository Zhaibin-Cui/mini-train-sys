from __future__ import annotations

import torch
from torch import nn

from minitrain.model.blocks import RMSNorm, TransformerBlock
from minitrain.model.config import ModelConfig
from minitrain.model.ops import OpsBackend
from minitrain.model.rotary import RotaryEmbedding


class MiniTransformer(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        ops: OpsBackend,
        *,
        activation_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if activation_dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError(f"Unsupported activation dtype: {activation_dtype}")
        self.cfg = cfg
        self.ops = ops
        self.activation_dtype = activation_dtype
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.rotary = RotaryEmbedding(
            cfg.head_dim,
            cfg.seq_len,
            cfg.rope_theta,
            cache_dtype=activation_dtype,
        )
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._init_weights()
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

    def _init_weights(self) -> None:
        base_std = 0.02
        nn.init.normal_(self.embed.weight, mean=0.0, std=base_std)
        if not self.cfg.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=base_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_fused_loss: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        # Embedding is not an autocast-eligible op. Cast it explicitly so the
        # residual stream starts, and remains, in the configured 16-bit dtype.
        x = self.dropout(self.embed(input_ids).to(dtype=self.activation_dtype))
        rope_cos, rope_sin = self.rotary(x.size(1))
        if rope_cos.device != x.device or rope_cos.dtype != x.dtype:
            raise RuntimeError(
                "RoPE cache must already match the residual device and dtype; "
                "per-forward conversion is intentionally disabled"
            )
        for block in self.blocks:
            x = block(x, self.ops, rope_cos, rope_sin)
        x = self.norm(x, self.ops)
        if targets is not None and use_fused_loss:
            loss = self.ops.fused_linear_cross_entropy(
                x.reshape(-1, x.size(-1)),
                self.lm_head.weight,
                targets.reshape(-1),
            )
            return loss, x
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = self.ops.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss, logits
