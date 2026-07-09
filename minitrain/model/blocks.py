import math

import torch
from torch import nn

from minitrain.model.config import ModelConfig
from minitrain.model.ops import OpsBackend


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, ops: OpsBackend) -> torch.Tensor:
        return ops.rmsnorm(x, self.weight, self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.out = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self._init_weights()
        inv_freq = 1.0 / (
            cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim)
        )
        positions = torch.arange(cfg.seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", emb.cos(), persistent=False)
        self.register_buffer("rope_sin", emb.sin(), persistent=False)

    def _init_weights(self) -> None:
        base_std = 0.02
        residual_std = base_std / math.sqrt(2 * self.cfg.n_layers)
        nn.init.normal_(self.qkv.weight, mean=0.0, std=base_std)
        nn.init.normal_(self.out.weight, mean=0.0, std=residual_std)

    def forward(self, x: torch.Tensor, ops: OpsBackend) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        if seq_len > self.rope_cos.size(0):
            raise ValueError(f"seq_len={seq_len} exceeds configured max seq_len={self.rope_cos.size(0)}")
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]
        q, k = ops.rope(q, k, cos, sin)
        dropout_p = self.cfg.dropout if self.training else 0.0
        y = ops.attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.resid_dropout(self.out(y))


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self._init_weights(cfg)

    def _init_weights(self, cfg: ModelConfig) -> None:
        base_std = 0.02
        residual_std = base_std / math.sqrt(2 * cfg.n_layers)
        nn.init.normal_(self.gate.weight, mean=0.0, std=base_std)
        nn.init.normal_(self.up.weight, mean=0.0, std=base_std)
        nn.init.normal_(self.down.weight, mean=0.0, std=residual_std)

    def forward(self, x: torch.Tensor, ops: OpsBackend) -> torch.Tensor:
        return self.dropout(self.down(ops.swiglu(self.gate(x), self.up(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, x: torch.Tensor, ops: OpsBackend) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x, ops), ops)
        x = x + self.mlp(self.mlp_norm(x, ops), ops)
        return x
