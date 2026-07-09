import torch

from minitrain.kernels.torch_ops import TorchOpsBackend
from minitrain.kernels.triton.flash_attention import is_flash_attention_supported
from minitrain.model import MiniTransformer, ModelConfig


class RecordingOpsBackend(TorchOpsBackend):
    def __init__(self) -> None:
        self.attention_dropout_p: list[float] = []

    def attention(self, q, k, v, *, is_causal, dropout_p):
        self.attention_dropout_p.append(dropout_p)
        return super().attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)


def _tiny_cfg(dropout: float) -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        seq_len=8,
        n_layers=2,
        n_heads=2,
        hidden_size=16,
        intermediate_size=32,
        dropout=dropout,
    )


def test_attention_dropout_uses_cfg_in_train_and_zero_in_eval() -> None:
    ops = RecordingOpsBackend()
    model = MiniTransformer(_tiny_cfg(dropout=0.25), ops)
    input_ids = torch.randint(0, 64, (2, 8))

    model.train()
    model(input_ids)
    assert ops.attention_dropout_p == [0.25, 0.25]

    ops.attention_dropout_p.clear()
    model.eval()
    model(input_ids)
    assert ops.attention_dropout_p == [0.0, 0.0]


def test_flash_attention_adapter_defers_nonzero_dropout_to_sdpa() -> None:
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)

    assert not is_flash_attention_supported(q, k, v, dropout_p=0.25)
