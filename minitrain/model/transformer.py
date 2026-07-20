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
        self.last_training_metrics: dict[str, torch.Tensor] = {}
        self.last_training_visualizations: dict[str, torch.Tensor] = {}
        # Backward-compatible names retained for existing experiment code.
        self.last_moe_metrics = self.last_training_metrics
        self.last_moe_visualizations = self.last_training_visualizations
        self._last_router_aux_loss: torch.Tensor | None = None
        self._last_router_z_loss: torch.Tensor | None = None

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

    def hidden_states(
        self,
        input_ids: torch.Tensor,
        *,
        embedding_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return final normalized states without materializing vocabulary logits.

        ``embedding_delta`` is intentionally narrow: probes can add a differentiable
        low-rank update while leaving the pretrained embedding and transformer frozen.
        """

        hidden = self.embed(input_ids)
        if embedding_delta is not None:
            if embedding_delta.shape != hidden.shape:
                raise ValueError("embedding_delta must match embedded input shape")
            hidden = hidden + embedding_delta
        hidden = self.dropout(hidden.to(dtype=self.activation_dtype))
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
        scalar_metrics: dict[str, torch.Tensor] = {}
        visualizations: dict[str, torch.Tensor] = {}
        for name, values in router_metrics.items():
            stacked = torch.stack(values)
            if stacked.ndim == 1:
                scalar_metrics[name] = stacked.float().mean()
                continue
            by_layer = stacked.detach()
            visualizations[f"{name}_by_layer"] = by_layer
            for expert_index, value in enumerate(by_layer.mean(dim=0)):
                scalar_metrics[f"{name}/expert_{expert_index:02d}"] = value
        self.last_training_metrics = scalar_metrics
        self.last_training_visualizations = visualizations
        self.last_moe_metrics = self.last_training_metrics
        self.last_moe_visualizations = self.last_training_visualizations
        if aux_losses:
            self.last_training_metrics["moe/aux_loss"] = (
                torch.stack(aux_losses).mean().detach()
            )
        if z_losses:
            self.last_training_metrics["moe/z_loss"] = (
                torch.stack(z_losses).mean().detach()
            )
        self._last_router_aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        self._last_router_z_loss = torch.stack(z_losses).mean() if z_losses else None
        return self.norm(hidden, self.ops)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_fused_loss: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        hidden = self.hidden_states(input_ids)
        if targets is None:
            return None, self.lm_head(hidden)
        lm_loss, output = self._base_loss(hidden, targets, use_fused_loss)
        loss = lm_loss
        self.last_training_metrics["loss/lm_cross_entropy"] = lm_loss.detach()
        moe_regularization = lm_loss.new_zeros(())
        if self._last_router_aux_loss is not None:
            weighted_aux = self.cfg.router_aux_loss_coef * self._last_router_aux_loss
            loss = loss + weighted_aux
            moe_regularization = moe_regularization + weighted_aux
            self.last_training_metrics["loss/moe_aux_weighted"] = weighted_aux.detach()
        if self._last_router_z_loss is not None:
            weighted_z = self.cfg.router_z_loss_coef * self._last_router_z_loss
            loss = loss + weighted_z
            moe_regularization = moe_regularization + weighted_z
            self.last_training_metrics["loss/moe_z_weighted"] = weighted_z.detach()
        if self.cfg.is_moe:
            self.last_training_metrics["loss/moe_regularization_total"] = (
                moe_regularization.detach()
            )
        self.last_training_metrics["loss/total"] = loss.detach()
        return loss, output


class MiniMoETransformer(MiniTransformer):
    """Compatibility name for constructing the shared model with an MoE config."""

    def __init__(self, cfg: ModelConfig, ops: OpsBackend, **kwargs) -> None:
        if not cfg.is_moe:
            raise ValueError("MiniMoETransformer requires ffn_type='moe'")
        super().__init__(cfg, ops, **kwargs)
