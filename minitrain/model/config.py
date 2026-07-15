from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32000
    seq_len: int = 2048
    n_layers: int = 12
    n_heads: int = 12
    hidden_size: int = 768
    intermediate_size: int = 2048
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_word_embeddings: bool = False
    ffn_type: str = "dense"
    num_experts: int = 8
    experts_per_token: int = 2
    router_aux_loss_coef: float = 1e-2
    router_z_loss_coef: float = 1e-3
    router_normalize_topk: bool = True
    router_jitter_noise: float = 0.0
    # Reserved for a future capacity-aware compact dispatch implementation.
    # TopKRouter.forward currently ignores these fields and routes droplessly;
    # the legacy _capacity_mask helper remains available for isolated experiments.
    expert_capacity_factor: float | None = None
    expert_min_capacity: int = 4

    def __post_init__(self) -> None:
        dimensions = {
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
        }
        if any(value <= 0 for value in dimensions.values()):
            raise ValueError("all model dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("norm_eps and rope_theta must be positive")
        if self.ffn_type not in {"dense", "moe"}:
            raise ValueError("ffn_type must be either 'dense' or 'moe'")
        if self.hidden_size % self.n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")
        if self.ffn_type == "moe":
            if self.num_experts <= 0:
                raise ValueError("num_experts must be positive")
            if not 0 < self.experts_per_token <= self.num_experts:
                raise ValueError("experts_per_token must be in [1, num_experts]")
            if self.router_aux_loss_coef < 0:
                raise ValueError("router_aux_loss_coef must be non-negative")
            if self.router_z_loss_coef < 0:
                raise ValueError("router_z_loss_coef must be non-negative")
            if not 0 <= self.router_jitter_noise < 1:
                raise ValueError("router_jitter_noise must be in [0, 1)")
            if self.expert_capacity_factor is not None and self.expert_capacity_factor <= 0:
                raise ValueError("expert_capacity_factor must be positive or null")
            if self.expert_min_capacity <= 0:
                raise ValueError("expert_min_capacity must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    @property
    def is_moe(self) -> bool:
        return self.ffn_type == "moe"


@dataclass(frozen=True)
class MoEModelConfig(ModelConfig):
    """Compatibility config whose only difference is the default FFN type."""

    ffn_type: str = "moe"
