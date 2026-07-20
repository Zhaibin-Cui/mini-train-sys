import torch

from minitrain.model.config import ModelConfig
from minitrain.model.ops import get_ops_backend
from minitrain.model.transformer import MiniTransformer


def _model(*, ffn_type: str) -> MiniTransformer:
    return MiniTransformer(
        ModelConfig(
            vocab_size=64,
            seq_len=16,
            n_layers=2,
            n_heads=4,
            hidden_size=32,
            intermediate_size=32,
            dropout=0.0,
            ffn_type=ffn_type,
            num_experts=4,
            experts_per_token=2,
            router_aux_loss_coef=0.01,
            router_z_loss_coef=0.001,
        ),
        get_ops_backend("torch"),
    )


def test_moe_reports_decomposed_loss_and_layer_expert_distributions():
    model = _model(ffn_type="moe")
    input_ids = torch.randint(0, model.cfg.vocab_size, (2, 8))
    targets = torch.randint(0, model.cfg.vocab_size, (2, 8))

    loss, _ = model(input_ids, targets=targets)
    metrics = model.last_training_metrics
    visualizations = model.last_training_visualizations

    assert torch.allclose(loss, metrics["loss/total"])
    assert torch.allclose(
        metrics["loss/moe_regularization_total"],
        metrics["loss/moe_aux_weighted"] + metrics["loss/moe_z_weighted"],
    )
    assert torch.allclose(
        loss,
        metrics["loss/lm_cross_entropy"]
        + metrics["loss/moe_regularization_total"],
    )
    load = visualizations["moe/expert_load_fraction_by_layer"]
    probability = visualizations["moe/expert_probability_by_layer"]
    assert load.shape == probability.shape == (model.cfg.n_layers, model.cfg.num_experts)
    assert torch.allclose(load.sum(dim=1), torch.ones(model.cfg.n_layers))
    assert torch.allclose(
        probability.sum(dim=1), torch.ones(model.cfg.n_layers), atol=1e-5
    )
    assert all(
        f"moe/expert_load_fraction/expert_{index:02d}" in metrics
        for index in range(model.cfg.num_experts)
    )
    assert model.last_moe_metrics is model.last_training_metrics


def test_dense_reports_only_architecture_independent_loss_metrics():
    model = _model(ffn_type="dense")
    input_ids = torch.randint(0, model.cfg.vocab_size, (2, 8))
    targets = torch.randint(0, model.cfg.vocab_size, (2, 8))

    loss, _ = model(input_ids, targets=targets)

    assert torch.allclose(loss, model.last_training_metrics["loss/lm_cross_entropy"])
    assert torch.allclose(loss, model.last_training_metrics["loss/total"])
    assert not any(key.startswith("moe/") for key in model.last_training_metrics)
    assert model.last_training_visualizations == {}
