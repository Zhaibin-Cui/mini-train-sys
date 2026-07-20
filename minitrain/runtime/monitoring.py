"""Shared progress, throughput, memory, and distributed metric utilities."""

from __future__ import annotations

import math
import time
import tracemalloc
from dataclasses import dataclass

import torch
import torch.distributed as dist

from minitrain.runtime.logger import EventLogger


def _distributed_device(device: torch.device) -> torch.device:
    """NCCL reductions require CUDA tensors; Gloo reductions use CPU tensors."""

    if dist.is_available() and dist.is_initialized() and dist.get_backend() == "nccl":
        return device
    return torch.device("cpu")


def distributed_mean(values: dict[str, float], device: torch.device) -> dict[str, float]:
    """Return the rank mean for a small, identically keyed scalar dictionary."""

    if not values:
        return {}
    keys = sorted(values)
    tensor = torch.tensor(
        [float(values[key]) for key in keys],
        dtype=torch.float64,
        device=_distributed_device(device),
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return dict(zip(keys, tensor.cpu().tolist()))


def distributed_mean_tensors(
    values: dict[str, torch.Tensor], device: torch.device
) -> dict[str, object]:
    """Average identically shaped visualization tensors and return JSON-safe lists."""

    averaged: dict[str, object] = {}
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    reduction_device = _distributed_device(device)
    for name in sorted(values):
        tensor = values[name].detach().to(device=reduction_device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= world_size
        averaged[name] = tensor.cpu().tolist()
    return averaged


def memory_metrics(device: torch.device) -> dict[str, float]:
    """Collect local plus node-job aggregate memory without hiding the max rank."""

    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if device.type != "cuda":
        _, peak = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)
        return {"host_peak_memory_mb": round(peak / 1024**2, 2)}

    local = torch.tensor(
        [
            torch.cuda.memory_allocated(device),
            torch.cuda.memory_reserved(device),
            torch.cuda.max_memory_allocated(device),
            torch.cuda.get_device_properties(device).total_memory,
        ],
        dtype=torch.float64,
        device=_distributed_device(device),
    )
    maximum = local.clone()
    total = local.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    mib = 1024**2
    return {
        "gpu_memory_allocated_mb": round(local[0].item() / mib, 2),
        "gpu_memory_reserved_mb": round(local[1].item() / mib, 2),
        "gpu_peak_memory_allocated_mb": round(local[2].item() / mib, 2),
        "gpu_memory_allocated_mb_max": round(maximum[0].item() / mib, 2),
        "gpu_memory_reserved_mb_max": round(maximum[1].item() / mib, 2),
        "gpu_peak_memory_allocated_mb_max": round(maximum[2].item() / mib, 2),
        "gpu_memory_allocated_mb_total": round(total[0].item() / mib, 2),
        "gpu_memory_reserved_mb_total": round(total[1].item() / mib, 2),
        "gpu_memory_capacity_mb_max": round(maximum[3].item() / mib, 2),
        "gpu_memory_capacity_mb_total": round(total[3].item() / mib, 2),
        "gpu_memory_utilization_percent_max": round(
            100 * maximum[0].item() / max(maximum[3].item(), 1), 2
        ),
        "monitored_world_size": world_size,
    }


@dataclass
class ProgressReporter:
    """Emit consistent terminal/JSONL/TensorBoard events for bounded tasks."""

    event: str
    total: int
    logger: EventLogger
    device: torch.device
    log_interval: int = 1
    unit: str = "batch"
    tokens_goal: int | None = None

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError("progress total must be positive")
        if self.log_interval <= 0:
            raise ValueError("progress log_interval must be positive")
        self.started_at = time.perf_counter()
        self.last_at = self.started_at
        self.processed_items = 0
        self.processed_tokens = 0
        self.last_payload: dict[str, object] = {}
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        elif not tracemalloc.is_tracing():
            tracemalloc.start()

    def update(
        self,
        completed: int,
        *,
        metrics: dict[str, float] | None = None,
        items: int = 0,
        tokens: int = 0,
        force: bool = False,
    ) -> None:
        self.processed_items += items
        self.processed_tokens += tokens
        if not force and completed != 1 and completed != self.total:
            if completed % self.log_interval:
                return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        elapsed = max(now - self.started_at, 1e-12)
        progress = min(max(completed / self.total, 0.0), 1.0)
        eta = elapsed * (1 - progress) / progress if progress > 0 else math.inf
        payload: dict[str, object] = {
            "event": self.event,
            "step": completed,
            self.unit: completed,
            {"batch": "batches", "example": "examples"}.get(
                self.unit, f"{self.unit}s"
            )
            + "_total": self.total,
            "progress_percent": round(progress * 100, 3),
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": round(eta, 3) if math.isfinite(eta) else -1.0,
            "items_processed": self.processed_items,
            "items_per_sec": round(self.processed_items / elapsed, 3),
            "tokens_processed": self.processed_tokens,
            "tokens_per_sec": round(self.processed_tokens / elapsed, 3),
        }
        if self.tokens_goal is not None:
            payload["tokens_goal"] = self.tokens_goal
        if metrics:
            payload.update(metrics)
        payload.update(memory_metrics(self.device))
        self.last_payload = payload
        self.logger.log_event(payload)
        self.last_at = now

    def summary(self) -> dict[str, object]:
        """Return the last emitted state for durable experiment result JSON."""

        return dict(self.last_payload)
