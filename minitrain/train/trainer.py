"""One optimizer-step engine shared by single-device, DDP, and FSDP runs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from minitrain.train.precision import PrecisionPolicy, resolve_precision_policy
from minitrain.train.lr_scheduler import LearningRateScheduler


@dataclass
class TrainState:
    step: int = 0
    # LR progress is separate from the lifetime optimizer-update counter so an
    # epoch-boundary elastic resume can remap it to a new steps-per-epoch.
    lr_step: int = 0
    tokens_seen: int = 0
    epoch: int = 0


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
        precision: str = "auto",
        grad_clip_norm: float | None = None,
        check_finite: bool = True,
        lr_scheduler: LearningRateScheduler | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_fused_loss = use_fused_loss
        self.precision: PrecisionPolicy = resolve_precision_policy(precision, device)
        self.grad_clip_norm = grad_clip_norm
        self.check_finite = check_finite
        self.lr_scheduler = lr_scheduler
        self.last_lr = float(optimizer.param_groups[0]["lr"])
        self.last_metrics: dict[str, torch.Tensor] = {}
        self.last_visualizations: dict[str, torch.Tensor] = {}
        self.last_grad_norm: float | None = None
        self.last_grad_clip_coefficient = 1.0
        self.last_grad_was_clipped = False
        self.grad_scaler = torch.amp.GradScaler(
            device.type,
            enabled=self.precision.grad_scaling_enabled,
        )
        self.state = TrainState()
        self.optimizer.zero_grad(set_to_none=True)

    def _clip_gradients(self) -> None:
        if self.grad_clip_norm is None:
            self.last_grad_norm = None
            self.last_grad_clip_coefficient = 1.0
            self.last_grad_was_clipped = False
            return
        # FSDP must compute a global norm across shards. Its wrapper supplies a
        # specialized method; single-device and DDP use the standard utility.
        clip_grad_norm = getattr(self.model, "clip_grad_norm_", None)
        if clip_grad_norm is not None:
            norm = clip_grad_norm(self.grad_clip_norm)
        else:
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.last_grad_norm = float(norm.detach())
        if not math.isfinite(self.last_grad_norm):
            raise FloatingPointError("non-finite global gradient norm")
        self.last_grad_clip_coefficient = min(
            1.0,
            self.grad_clip_norm / (self.last_grad_norm + 1e-6),
        )
        self.last_grad_was_clipped = self.last_grad_norm > self.grad_clip_norm

    def train_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run forward, backward, clipping, optimizer, and scheduler exactly once."""

        self.model.train()

        # DataLoader workers produce CPU tensors.  Pinned-memory batches can use
        # these non-blocking copies to overlap host-to-device transfer with CUDA.
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        targets = batch["targets"].to(self.device, non_blocking=True)

        # Parameters remain fp32; autocast selects bf16/fp16 only for eligible
        # forward operators according to the resolved PrecisionPolicy.
        with self.precision.autocast_context(self.device):
            loss, _ = self.model(input_ids, targets=targets, use_fused_loss=self.use_fused_loss)
        if loss is None:
            raise RuntimeError("Expected loss during training")
        if self.check_finite:
            finite = torch.isfinite(loss.detach())
            if loss.is_cuda and hasattr(torch, "_assert_async"):
                torch._assert_async(finite, "non-finite training loss")
            elif not bool(finite):
                raise FloatingPointError("non-finite training loss")

        # DDP exposes model-owned diagnostics through .module. Copy them
        # before the next forward overwrites the transient metric dictionary.
        metric_source = getattr(self.model, "module", self.model)
        self.last_metrics = dict(
            getattr(
                metric_source,
                "last_training_metrics",
                getattr(metric_source, "last_moe_metrics", {}),
            )
        )
        self.last_visualizations = dict(
            getattr(
                metric_source,
                "last_training_visualizations",
                getattr(metric_source, "last_moe_visualizations", {}),
            )
        )

        self.last_lr = float(self.optimizer.param_groups[0]["lr"])
        # fp16 requires dynamic loss scaling.  Unscale before clipping so the
        # configured norm is expressed in true gradient units.  bf16/fp32 use
        # the direct branch because their exponent range does not need scaling.
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
        # This trainer currently performs one optimizer update per batch (no
        # gradient accumulation), then advances token and LR schedule state.
        self.optimizer.zero_grad(set_to_none=True)
        self.state.step += 1
        self.state.lr_step += 1
        self.state.tokens_seen += input_ids.numel()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.state.lr_step)
        if self.last_grad_norm is not None:
            self.last_metrics["grad_norm"] = torch.tensor(self.last_grad_norm)
            self.last_metrics["grad_clip_threshold"] = torch.tensor(self.grad_clip_norm)
            self.last_metrics["grad_clip_coefficient"] = torch.tensor(
                self.last_grad_clip_coefficient
            )
            self.last_metrics["grad_was_clipped"] = torch.tensor(
                float(self.last_grad_was_clipped)
            )
        if self.grad_scaler.is_enabled():
            self.last_metrics["grad_scale"] = torch.tensor(self.grad_scaler.get_scale())
        return loss.detach()
