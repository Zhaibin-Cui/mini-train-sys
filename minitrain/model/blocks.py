from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from minitrain.model.config import ModelConfig
from minitrain.model.moe_router import TopKRouter
from minitrain.model.ops import OpsBackend


@dataclass
class FeedForwardOutput:
    hidden_states: torch.Tensor
    router_aux_loss: torch.Tensor | None = None
    router_z_loss: torch.Tensor | None = None
    router_metrics: dict[str, torch.Tensor] | None = None


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
        nn.init.normal_(self.qkv.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def forward(self, x, ops, rope_cos, rope_sin):
        batch, seq_len, hidden = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        head_shape = (batch, seq_len, self.cfg.n_heads, self.cfg.head_dim)
        q = q.view(head_shape).transpose(1, 2)
        k = k.view(head_shape).transpose(1, 2)
        v = v.view(head_shape).transpose(1, 2)
        q, k = ops.rope(q, k, rope_cos, rope_sin)
        dropout_p = self.cfg.dropout if self.training else 0.0
        output = ops.attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        return self.resid_dropout(self.out(output))


class DenseFeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def forward(self, x: torch.Tensor, ops: OpsBackend) -> FeedForwardOutput:
        hidden = self.down(ops.swiglu(self.gate(x), self.up(x)))
        return FeedForwardOutput(self.dropout(hidden))


class MoEFeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.router = TopKRouter(cfg)
        self.gate_up_proj = nn.Parameter(
            torch.empty(cfg.num_experts, 2 * cfg.intermediate_size, cfg.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(cfg.num_experts, cfg.hidden_size, cfg.intermediate_size)
        )
        self.dropout = nn.Dropout(cfg.dropout)
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def forward(self, x: torch.Tensor, ops: OpsBackend) -> FeedForwardOutput:
        shape = x.shape
        flat_x = x.reshape(-1, shape[-1])
        route = self.router(flat_x, ops)
        hidden = ops.fused_moe(
            flat_x,
            self.gate_up_proj,
            self.down_proj,
            route.expert_indices,
            route.expert_weights,
        ).view(shape)
        return FeedForwardOutput(
            hidden_states=self.dropout(hidden),
            router_aux_loss=route.auxiliary_loss,
            router_z_loss=route.z_loss,
            router_metrics=route.metrics,
        )


def build_feed_forward(cfg: ModelConfig) -> nn.Module:
    return MoEFeedForward(cfg) if cfg.is_moe else DenseFeedForward(cfg)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.ffn = build_feed_forward(cfg)

    @property
    def mlp_norm(self):
        return self.ffn_norm

    @property
    def mlp(self):
        return self.ffn

    def forward(self, x, ops, rope_cos, rope_sin) -> FeedForwardOutput:
        x = x + self.attn(self.attn_norm(x, ops), ops, rope_cos, rope_sin)
        ffn_output = self.ffn(self.ffn_norm(x, ops), ops)
        return FeedForwardOutput(
            hidden_states=x + ffn_output.hidden_states,
            router_aux_loss=ffn_output.router_aux_loss,
            router_z_loss=ffn_output.router_z_loss,
            router_metrics=ffn_output.router_metrics,
        )


SwiGLUMLP = DenseFeedForward
