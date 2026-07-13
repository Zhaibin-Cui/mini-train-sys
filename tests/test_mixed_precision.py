import pytest
import torch

from minitrain.distributed.fsdp import FSDPStrategy
from minitrain.kernels.torch_ops import TorchOpsBackend
from minitrain.model import MiniTransformer, ModelConfig
from minitrain.train.precision import resolve_precision_policy


def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        seq_len=8,
        n_layers=2,
        n_heads=2,
        hidden_size=32,
        intermediate_size=64,
        dropout=0.0,
    )


def test_precision_policy_contract() -> None:
    fp32 = resolve_precision_policy("fp32", torch.device("cpu"))
    bf16 = resolve_precision_policy("bf16", torch.device("cpu"))

    assert fp32.activation_dtype is torch.float32
    assert not fp32.autocast_enabled
    assert not fp32.grad_scaling_enabled
    assert bf16.activation_dtype is torch.bfloat16
    assert bf16.autocast_enabled
    assert not bf16.grad_scaling_enabled

    with pytest.raises(ValueError, match="only on CUDA"):
        resolve_precision_policy("fp16", torch.device("cpu"))
    with pytest.raises(ValueError, match="Unknown precision"):
        resolve_precision_policy("int8", torch.device("cpu"))


def test_rope_cache_is_shared_at_model_level() -> None:
    model = MiniTransformer(tiny_model_config(), TorchOpsBackend())

    buffers = dict(model.named_buffers())
    assert set(buffers) == {"rotary.cos", "rotary.sin"}
    assert buffers["rotary.cos"].dtype is torch.float32
    assert buffers["rotary.sin"].dtype is torch.float32
    assert all(not dict(block.attn.named_buffers()) for block in model.blocks)
    assert "rotary.cos" not in model.state_dict()
    assert "rotary.sin" not in model.state_dict()


def test_bf16_rope_slices_share_preconverted_cache_storage() -> None:
    model = MiniTransformer(
        tiny_model_config(),
        TorchOpsBackend(),
        activation_dtype=torch.bfloat16,
    )

    cos, sin = model.rotary(4)

    assert model.rotary.cos.dtype is torch.bfloat16
    assert model.rotary.sin.dtype is torch.bfloat16
    assert cos.untyped_storage().data_ptr() == model.rotary.cos.untyped_storage().data_ptr()
    assert sin.untyped_storage().data_ptr() == model.rotary.sin.untyped_storage().data_ptr()


def test_bf16_residual_stream_with_fp32_parameters_and_gradients() -> None:
    cfg = tiny_model_config()
    model = MiniTransformer(
        cfg,
        TorchOpsBackend(),
        activation_dtype=torch.bfloat16,
    )
    residual_dtypes: list[torch.dtype] = []
    handles = [
        block.register_forward_pre_hook(
            lambda _module, args: residual_dtypes.append(args[0].dtype)
        )
        for block in model.blocks
    ]
    input_ids = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))

    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, logits = model(input_ids, targets=input_ids)
    assert loss is not None
    loss.backward()

    for handle in handles:
        handle.remove()
    assert residual_dtypes == [torch.bfloat16] * cfg.n_layers
    assert logits.dtype is torch.bfloat16
    assert loss.dtype is torch.float32
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
    assert {
        parameter.grad.dtype
        for parameter in model.parameters()
        if parameter.grad is not None
    } == {torch.float32}


def test_fsdp_mixed_precision_keeps_reductions_and_grads_fp32() -> None:
    strategy = FSDPStrategy(sharding_strategy="full_shard", precision="bf16")

    assert strategy.mixed_precision is not None
    assert strategy.mixed_precision.param_dtype is torch.bfloat16
    assert strategy.mixed_precision.buffer_dtype is torch.bfloat16
    assert strategy.mixed_precision.reduce_dtype is torch.float32
    assert not strategy.mixed_precision.keep_low_precision_grads
