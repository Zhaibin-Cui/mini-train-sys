from __future__ import annotations

import os
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from minitrain.distributed.strategy import ParallelStrategy
from minitrain.runtime.config import ExperimentConfig
from minitrain.runtime.logger import EventLogger
from minitrain.train.checkpoint import checkpoint_path, save_checkpoint
from minitrain.train.trainer import Trainer


def _memory_metrics_mb(device: torch.device) -> dict[str, float]:
    if device.type == "cuda":
        return {
            "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / 1024**2, 2),
            "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved(device) / 1024**2, 2),
            "gpu_peak_memory_allocated_mb": round(
                torch.cuda.max_memory_allocated(device) / 1024**2, 2
            ),
        }
    _, peak_bytes = tracemalloc.get_traced_memory()
    return {"host_peak_memory_mb": round(peak_bytes / 1024**2, 2)}


class TrainingRunner:
    def __init__(
        self,
        *,
        cfg: ExperimentConfig,
        trainer: Trainer,
        dataloader: Any,
        strategy: ParallelStrategy,
        logger: EventLogger,
        device: torch.device,
        world_size: int,
    ) -> None:
        self.cfg = cfg
        self.trainer = trainer
        self.dataloader = dataloader
        self.strategy = strategy
        self.logger = logger
        self.device = device
        self.world_size = world_size
        self.rank = int(os.environ.get("RANK", "0"))

    def _log_step(
        self,
        *,
        epoch: int,
        loss: torch.Tensor,
        interval_tokens: int,
        interval_seconds: float,
        session_tokens: int,
        session_seconds: float,
    ) -> None:
        state = self.trainer.state
        payload: dict[str, object] = {
            "event": "train",
            "step": state.step,
            "epoch": epoch,
            "loss": round(float(loss), 6),
            "lr": self.trainer.last_lr,
            "tokens_seen": state.tokens_seen * self.world_size,
            "tokens_per_sec": round(
                interval_tokens * self.world_size / max(interval_seconds, 1e-12), 2
            ),
            "avg_tokens_per_sec": round(
                session_tokens * self.world_size / max(session_seconds, 1e-12), 2
            ),
        }
        payload.update(_memory_metrics_mb(self.device))
        payload.update(
            {name: float(value) for name, value in self.trainer.last_metrics.items()}
        )
        self.logger.log_event(payload)

    def _save(self, *, epoch: int) -> Path:
        state = self.trainer.state
        path = checkpoint_path(
            self.cfg.train.checkpoint_dir,
            self.cfg.run.name,
            epoch=epoch,
            step=state.step,
        )
        save_checkpoint(
            path,
            self.trainer.model,
            self.trainer.optimizer,
            state.step,
            epoch=epoch,
            tokens_seen=state.tokens_seen,
            grad_scaler=self.trainer.grad_scaler,
            lr_scheduler=self.trainer.lr_scheduler,
            precision=self.trainer.precision.name,
            config=asdict(self.cfg),
            write=self.rank == 0,
        )
        if self.rank == 0:
            self.logger.log_event({"event": "checkpoint", "path": str(path)})
        return path

    def run(self, *, max_steps: int | None, resume_path: Path | None = None) -> None:
        if self.device.type != "cuda":
            tracemalloc.start()
        else:
            torch.cuda.reset_peak_memory_stats(self.device)

        self.strategy.barrier()
        started = last_log_at = time.perf_counter()
        initial_tokens = last_log_tokens = self.trainer.state.tokens_seen
        state = self.trainer.state
        stop = max_steps is not None and state.step >= max_steps
        epoch = state.epoch
        last_checkpoint = resume_path

        while not stop and (self.cfg.train.epochs is None or epoch < self.cfg.train.epochs):
            epoch += 1
            completed_epoch = True
            batches_in_epoch = len(self.dataloader)
            for batch_index, batch in enumerate(self.dataloader, start=1):
                loss = self.trainer.train_step(batch)
                should_stop = max_steps is not None and state.step >= max_steps
                should_log = (
                    state.step == 1
                    or state.step % self.cfg.train.log_interval == 0
                    or should_stop
                )
                if should_log:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    now = time.perf_counter()
                    self._log_step(
                        epoch=epoch,
                        loss=loss,
                        interval_tokens=state.tokens_seen - last_log_tokens,
                        interval_seconds=now - last_log_at,
                        session_tokens=state.tokens_seen - initial_tokens,
                        session_seconds=now - started,
                    )
                    last_log_at = now
                    last_log_tokens = state.tokens_seen
                if should_stop:
                    completed_epoch = batch_index == batches_in_epoch
                    stop = True
                    break

            if completed_epoch:
                state.epoch = epoch
            checkpoint_due = (
                completed_epoch
                and self.cfg.train.checkpoint_every_epochs is not None
                and epoch % self.cfg.train.checkpoint_every_epochs == 0
            )
            if checkpoint_due:
                last_checkpoint = self._save(epoch=epoch)
            self.strategy.barrier()

        if self.cfg.train.save_final_checkpoint:
            final_path = checkpoint_path(
                self.cfg.train.checkpoint_dir,
                self.cfg.run.name,
                epoch=state.epoch,
                step=state.step,
            )
            if final_path != last_checkpoint:
                self._save(epoch=state.epoch)

        if self.device.type != "cuda" and tracemalloc.is_tracing():
            tracemalloc.stop()
