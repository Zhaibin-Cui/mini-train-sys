"""Epoch/step orchestration around the single-step :class:`Trainer`."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from minitrain.distributed.strategy import ParallelStrategy
from minitrain.runtime.config import ExperimentConfig
from minitrain.runtime.logger import EventLogger
from minitrain.runtime.monitoring import (
    GpuUtilizationMonitor,
    distributed_mean,
    distributed_mean_tensors,
    distributed_gpu_utilization,
    memory_metrics,
)
from minitrain.train.checkpoint import checkpoint_path, prune_checkpoints, save_checkpoint
from minitrain.train.lr_scheduler import resolve_total_steps
from minitrain.train.trainer import Trainer


class TrainingRunner:
    """Coordinate epochs, sampler state, logging, barriers, and checkpoints."""

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
        self.rank = strategy.rank
        self.gpu_monitor = GpuUtilizationMonitor(device)

    def _log_step(
        self,
        *,
        epoch: int,
        batch_index: int,
        batches_in_epoch: int,
        total_steps: int,
        tokens_goal: int,
        interval_metric_sums: dict[str, torch.Tensor],
        interval_visualization_sums: dict[str, torch.Tensor],
        interval_steps: int,
        interval_tokens: int,
        interval_seconds: float,
        interval_data_wait_seconds: float,
        interval_clipped_steps: int,
        session_tokens: int,
        session_seconds: float,
        initial_step: int,
    ) -> None:
        state = self.trainer.state
        scalars = {
            name: float(value / max(interval_steps, 1))
            for name, value in interval_metric_sums.items()
        }
        scalars["lr"] = self.trainer.last_lr
        scalars["grad_clip_fraction"] = interval_clipped_steps / max(interval_steps, 1)
        scalars = distributed_mean(scalars, self.device)
        visualizations = distributed_mean_tensors(
            {
                name: value / max(interval_steps, 1)
                for name, value in interval_visualization_sums.items()
            },
            self.device,
        )
        completed_this_session = max(state.step - initial_step, 1)
        remaining_steps = max(total_steps - state.step, 0)
        eta_seconds = session_seconds / completed_this_session * remaining_steps
        payload: dict[str, object] = {
            "event": "train",
            "step": state.step,
            "step_total": total_steps,
            "epoch": epoch,
            "batch": batch_index,
            "batches_total": batches_in_epoch,
            "optimizer_step": state.step,
            "loss": round(scalars.pop("loss"), 6),
            "lr": scalars.pop("lr"),
            "tokens_seen": state.tokens_seen * self.world_size,
            "tokens_goal": tokens_goal,
            "progress_percent": round(100 * state.step / max(total_steps, 1), 3),
            "elapsed_seconds": round(session_seconds, 3),
            "eta_seconds": round(eta_seconds, 3),
            "tokens_per_sec": round(
                interval_tokens * self.world_size / max(interval_seconds, 1e-12), 2
            ),
            "avg_tokens_per_sec": round(
                session_tokens * self.world_size / max(session_seconds, 1e-12), 2
            ),
            "step_time_ms": round(1000 * interval_seconds / max(interval_steps, 1), 3),
            "data_wait_ms": round(
                1000 * interval_data_wait_seconds / max(interval_steps, 1), 3
            ),
            "data_wait_percent": round(
                100 * interval_data_wait_seconds / max(interval_seconds, 1e-12), 3
            ),
        }
        if self.cfg.train.epochs is not None:
            payload["epochs_total"] = self.cfg.train.epochs
        payload.update(memory_metrics(self.device, reset_peak_stats=True))
        payload.update(
            distributed_gpu_utilization(self.gpu_monitor.read_interval(), self.device)
        )
        payload.update(scalars)
        payload.update(visualizations)
        self.logger.log_event(payload)

    @staticmethod
    def _accumulate_tensors(
        totals: dict[str, torch.Tensor], values: dict[str, torch.Tensor]
    ) -> None:
        """Accumulate tiny detached diagnostics without synchronizing CUDA each step."""

        for name, value in values.items():
            detached = value.detach()
            if name in totals:
                totals[name].add_(detached)
            else:
                totals[name] = detached.clone()

    def _save(self, *, epoch: int, force_model_export: bool = False) -> Path:
        state = self.trainer.state
        path = checkpoint_path(
            self.cfg.checkpoint.directory,
            self.cfg.run.name,
            epoch=epoch,
            step=state.step,
        )
        export_interval = self.cfg.checkpoint.export_model_every_epochs
        export_model = self.cfg.checkpoint.export_model and (
            force_model_export
            or export_interval is None
            or epoch % export_interval == 0
        )
        checkpoint_started = time.perf_counter()
        save_checkpoint(
            path,
            self.trainer.model,
            self.trainer.optimizer,
            state.step,
            epoch=epoch,
            lr_step=state.lr_step,
            tokens_seen=state.tokens_seen,
            grad_scaler=self.trainer.grad_scaler,
            lr_scheduler=self.trainer.lr_scheduler,
            precision=self.trainer.precision.name,
            config=asdict(self.cfg),
            export_model=export_model,
            cpu_offload=self.cfg.checkpoint.cpu_offload,
        )
        if self.rank == 0:
            checkpoint_seconds = time.perf_counter() - checkpoint_started
            checkpoint_bytes = sum(
                item.stat().st_size for item in path.rglob("*") if item.is_file()
            )
            removed: list[Path] = []
            if self.cfg.checkpoint.keep_last is not None:
                removed = prune_checkpoints(
                    self.cfg.checkpoint.directory,
                    self.cfg.run.name,
                    keep_last=self.cfg.checkpoint.keep_last,
                    keep_safety=self.cfg.checkpoint.keep_safety,
                    safety_every_epochs=self.cfg.checkpoint.safety_every_epochs,
                    keep_model_exports=self.cfg.checkpoint.keep_model_exports,
                )
            self.logger.log_event(
                {
                    "event": "checkpoint",
                    "path": str(path),
                    "includes_optimizer": True,
                    "exported_model": export_model,
                    "checkpoint_seconds": round(checkpoint_seconds, 3),
                    "checkpoint_bytes": checkpoint_bytes,
                    "removed_old_checkpoints": [str(old) for old in removed],
                }
            )
        return path

    def run(self, *, max_steps: int | None, resume_path: Path | None = None) -> None:
        """Run until the configured step or epoch terminal condition is met."""

        # Memory accounting is backend-specific: tracemalloc for host runs and
        # allocator peak counters for CUDA runs.
        if self.device.type != "cuda":
            tracemalloc.start()
        else:
            torch.cuda.reset_peak_memory_stats(self.device)
            self.gpu_monitor.start()

        # Enter timing only after every distributed rank has completed setup.
        self.strategy.barrier()
        started = last_log_at = time.perf_counter()
        initial_tokens = last_log_tokens = self.trainer.state.tokens_seen
        state = self.trainer.state
        initial_step = last_log_step = state.step
        interval_data_wait_seconds = 0.0
        interval_clipped_steps = 0
        interval_metric_sums: dict[str, torch.Tensor] = {}
        interval_visualization_sums: dict[str, torch.Tensor] = {}
        total_steps = resolve_total_steps(
            max_steps=max_steps,
            epochs=self.cfg.train.epochs,
            steps_per_epoch=len(self.dataloader),
        )
        tokens_goal = (
            total_steps
            * self.cfg.train.batch_size
            * int(self.cfg.model["seq_len"])
            * self.world_size
        )
        stop = max_steps is not None and state.step >= max_steps
        epoch = state.epoch
        last_checkpoint = resume_path

        while not stop and (self.cfg.train.epochs is None or epoch < self.cfg.train.epochs):
            epoch += 1
            sampler = getattr(self.dataloader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                # Both bounded block shuffle and randomized-document packing use
                # seed+epoch, so all ranks must receive the same epoch number.
                sampler.set_epoch(epoch)
            completed_epoch = True
            batches_in_epoch = len(self.dataloader)
            iterator = iter(self.dataloader)
            for batch_index in range(1, batches_in_epoch + 1):
                data_wait_started = time.perf_counter()
                batch = next(iterator)
                interval_data_wait_seconds += time.perf_counter() - data_wait_started
                loss = self.trainer.train_step(batch)
                self._accumulate_tensors(interval_metric_sums, {"loss": loss})
                self._accumulate_tensors(interval_metric_sums, self.trainer.last_metrics)
                self._accumulate_tensors(
                    interval_visualization_sums, self.trainer.last_visualizations
                )
                interval_clipped_steps += int(self.trainer.last_grad_was_clipped)
                should_stop = max_steps is not None and state.step >= max_steps
                should_log = (
                    state.step == initial_step + 1
                    or state.step % self.cfg.train.log_interval == 0
                    or batch_index == batches_in_epoch
                    or should_stop
                )
                if should_log:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    now = time.perf_counter()
                    self._log_step(
                        epoch=epoch,
                        batch_index=batch_index,
                        batches_in_epoch=batches_in_epoch,
                        total_steps=total_steps,
                        tokens_goal=tokens_goal,
                        interval_metric_sums=interval_metric_sums,
                        interval_visualization_sums=interval_visualization_sums,
                        interval_steps=state.step - last_log_step,
                        interval_tokens=state.tokens_seen - last_log_tokens,
                        interval_seconds=now - last_log_at,
                        interval_data_wait_seconds=interval_data_wait_seconds,
                        interval_clipped_steps=interval_clipped_steps,
                        session_tokens=state.tokens_seen - initial_tokens,
                        session_seconds=now - started,
                        initial_step=initial_step,
                    )
                    last_log_at = now
                    last_log_tokens = state.tokens_seen
                    last_log_step = state.step
                    interval_data_wait_seconds = 0.0
                    interval_clipped_steps = 0
                    interval_metric_sums.clear()
                    interval_visualization_sums.clear()
                if should_stop:
                    completed_epoch = batch_index == batches_in_epoch
                    stop = True
                    break

            # Only commit epoch state after consuming its final batch.  A run
            # stopped by max_steps mid-epoch keeps the previous complete epoch.
            if completed_epoch:
                state.epoch = epoch
            checkpoint_due = (
                completed_epoch
                and self.cfg.checkpoint.every_epochs is not None
                and epoch % self.cfg.checkpoint.every_epochs == 0
            )
            if checkpoint_due:
                terminal_epoch = (
                    self.cfg.train.epochs is not None and epoch >= self.cfg.train.epochs
                )
                terminal_step = max_steps is not None and state.step >= max_steps
                last_checkpoint = self._save(
                    epoch=epoch,
                    force_model_export=terminal_epoch or terminal_step,
                )
            # Keep ranks aligned around rank-0-only checkpoint publication.
            self.strategy.barrier()

        # Avoid rewriting an identical path when the epoch checkpoint is already
        # the terminal checkpoint for this run.
        if self.cfg.checkpoint.save_final:
            final_path = checkpoint_path(
                self.cfg.checkpoint.directory,
                self.cfg.run.name,
                epoch=state.epoch,
                step=state.step,
            )
            if final_path != last_checkpoint:
                self._save(epoch=state.epoch, force_model_export=True)

        if self.device.type != "cuda" and tracemalloc.is_tracing():
            tracemalloc.stop()
        self.gpu_monitor.close()
