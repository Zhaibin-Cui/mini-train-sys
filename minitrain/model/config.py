from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32000
    seq_len: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    hidden_size: int = 768
    intermediate_size: int = 2048
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")
        return self.hidden_size // self.n_heads

