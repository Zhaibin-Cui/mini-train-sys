from dataclasses import dataclass

import torch


@dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0


class Trainer:
    """Minimal trainer that is deliberately unaware of kernels and distribution.

    `model` may already be wrapped by a `ParallelStrategy`, and the model itself
    owns the `OpsBackend`. This separation is the main reason operator and
    distributed experiments can be mixed independently.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        use_fused_loss: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_fused_loss = use_fused_loss
        self.state = TrainState()
        self.optimizer.zero_grad(set_to_none=True)

    def train_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        targets = batch["targets"].to(self.device)
        loss, _ = self.model(input_ids, targets=targets, use_fused_loss=self.use_fused_loss)
        if loss is None:
            raise RuntimeError("Expected loss during training")
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.step += 1
        self.state.tokens_seen += input_ids.numel()
        return loss.detach()
