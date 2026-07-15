from __future__ import annotations

import torch
from torch import nn

from minitrain.model.blocks import RMSNorm, TransformerBlock
from minitrain.model.config import ModelConfig
from minitrain.model.ops import OpsBackend
from minitrain.model.rotary import RotaryEmbedding


class MiniTransformer(nn.Module):
    """Shared causal LM whose FFN family is selected by ``ModelConfig.ffn_type``."""

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
            cfg.head_dim, cfg.seq_len, cfg.rope_theta, cache_dtype=activation_dtype
        )
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        self.last_moe_metrics: dict[str, torch.Tensor] = {}

    def _base_loss(
        self, hidden: torch.Tensor, targets: torch.Tensor, use_fused_loss: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_fused_loss:
            loss = self.ops.fused_linear_cross_entropy(
                hidden.reshape(-1, hidden.size(-1)),
                self.lm_head.weight,
                targets.reshape(-1),
            )
            return loss, hidden
        logits = self.lm_head(hidden)
        loss = self.ops.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
        return loss, logits

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_fused_loss: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        hidden = self.dropout(self.embed(input_ids).to(dtype=self.activation_dtype))
        rope_cos, rope_sin = self.rotary(hidden.size(1))
        if rope_cos.device != hidden.device or rope_cos.dtype != hidden.dtype:
            raise RuntimeError("RoPE cache must match the residual device and dtype")
        aux_losses = []
        z_losses = []
        router_metrics: dict[str, list[torch.Tensor]] = {}
        for block in self.blocks:
            block_output = block(hidden, self.ops, rope_cos, rope_sin)
            hidden = block_output.hidden_states
            if block_output.router_aux_loss is not None:
                aux_losses.append(block_output.router_aux_loss)
            if block_output.router_z_loss is not None:
                z_losses.append(block_output.router_z_loss)
            for name, value in (block_output.router_metrics or {}).items():
                router_metrics.setdefault(name, []).append(value)
        self.last_moe_metrics = {
            name: torch.stack(values).mean() for name, values in router_metrics.items()
        }
        if aux_losses:
            self.last_moe_metrics["moe/aux_loss"] = torch.stack(aux_losses).mean().detach()
        if z_losses:
            self.last_moe_metrics["moe/z_loss"] = torch.stack(z_losses).mean().detach()
        hidden = self.norm(hidden, self.ops)
        if targets is None:
            return None, self.lm_head(hidden)
        loss, output = self._base_loss(hidden, targets, use_fused_loss)
        if aux_losses and self.cfg.router_aux_loss_coef:
            loss = loss + self.cfg.router_aux_loss_coef * torch.stack(aux_losses).mean()
        if z_losses and self.cfg.router_z_loss_coef:
            loss = loss + self.cfg.router_z_loss_coef * torch.stack(z_losses).mean()
        return loss, output


class MiniMoETransformer(MiniTransformer):
    """Compatibility name for constructing the shared model with an MoE config."""

    def __init__(self, cfg: ModelConfig, ops: OpsBackend, **kwargs) -> None:
        if not cfg.is_moe:
            raise ValueError("MiniMoETransformer requires ffn_type='moe'")
        super().__init__(cfg, ops, **kwargs)
