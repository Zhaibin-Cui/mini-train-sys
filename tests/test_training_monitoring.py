import pytest
import torch

from minitrain.train.trainer import Trainer


class ScalarLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, targets=None, use_fused_loss=False):
        del targets, use_fused_loss
        loss = self.weight * input_ids.float().sum()
        return loss, input_ids


def test_trainer_reports_global_clip_state():
    model = ScalarLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model,
        optimizer,
        device=torch.device("cpu"),
        precision="fp32",
        grad_clip_norm=5.0,
    )

    trainer.train_step(
        {"input_ids": torch.tensor([10]), "targets": torch.tensor([0])}
    )

    assert float(trainer.last_metrics["grad_norm"]) == pytest.approx(10.0)
    assert float(trainer.last_metrics["grad_clip_threshold"]) == 5.0
    assert float(trainer.last_metrics["grad_clip_coefficient"]) == pytest.approx(0.5)
    assert float(trainer.last_metrics["grad_was_clipped"]) == 1.0
    assert model.weight.item() == pytest.approx(0.5)
