from dataclasses import dataclass

import torch

from minitrain.train.precision import PrecisionPolicy, resolve_precision_policy


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
        precision: str = "fp32",
        grad_clip_norm: float | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_fused_loss = use_fused_loss
        self.precision: PrecisionPolicy = resolve_precision_policy(precision, device)
        self.grad_clip_norm = grad_clip_norm
        self.grad_scaler = torch.amp.GradScaler(
            device.type,
            enabled=self.precision.grad_scaling_enabled,
        )
        self.state = TrainState()
        self.optimizer.zero_grad(set_to_none=True)

    def _clip_gradients(self) -> None:
        if self.grad_clip_norm is None:
            return
        # FSDP must compute a global norm across shards. Its wrapper supplies a
        # specialized method; single-device and DDP use the standard utility.
        clip_grad_norm = getattr(self.model, "clip_grad_norm_", None)
        if clip_grad_norm is not None:
            clip_grad_norm(self.grad_clip_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

    def train_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        targets = batch["targets"].to(self.device)
        with self.precision.autocast_context(self.device):
            loss, _ = self.model(input_ids, targets=targets, use_fused_loss=self.use_fused_loss)
        if loss is None:
            raise RuntimeError("Expected loss during training")

        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            self._clip_gradients()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self._clip_gradients()
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.step += 1
        self.state.tokens_seen += input_ids.numel()
        return loss.detach()
