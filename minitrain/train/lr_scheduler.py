from __future__ import annotations

import math
from typing import Any

import torch

from minitrain.runtime.config import LRSchedulerConfig


def resolve_total_steps(
    *, max_steps: int | None, epochs: int | None, steps_per_epoch: int
) -> int:
    candidates = []
    if max_steps is not None:
        candidates.append(max_steps)
    if epochs is not None:
        candidates.append(epochs * steps_per_epoch)
    if not candidates:
        raise ValueError("A max_steps or epochs training limit is required")
    return min(candidates)


class LearningRateScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        cfg: LRSchedulerConfig,
        *,
        total_steps: int,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self.optimizer = optimizer
        self.cfg = cfg
        self.total_steps = total_steps
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_step = 0
        self.step(0)

    def _scale(self, step: int) -> float:
        if self.cfg.warmup_steps and step < self.cfg.warmup_steps:
            return (step + 1) / self.cfg.warmup_steps
        if self.cfg.schedule == "constant":
            return 1.0

        decay_end = self.cfg.decay_steps or self.total_steps
        decay_start = self.cfg.warmup_steps
        progress = (step - decay_start) / max(1, decay_end - decay_start)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.cfg.min_lr_ratio + (1.0 - self.cfg.min_lr_ratio) * cosine

    def step(self, completed_steps: int) -> None:
        self.last_step = completed_steps
        scale = self._scale(completed_steps)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {"last_step": self.last_step, "total_steps": self.total_steps}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.last_step = int(state["last_step"])
        self.step(self.last_step)
