"""Shared progress, throughput, memory, and distributed metric utilities."""

from __future__ import annotations

import math
import threading
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any

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


def memory_metrics(
    device: torch.device, *, reset_peak_stats: bool = False
) -> dict[str, float]:
    """Collect allocator memory, with unambiguous current and interval-peak ratios."""

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
    metrics = {
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
        "gpu_memory_current_percent_max": round(
            100 * maximum[0].item() / max(maximum[3].item(), 1), 2
        ),
        "gpu_memory_reserved_percent_max": round(
            100 * maximum[1].item() / max(maximum[3].item(), 1), 2
        ),
        "gpu_memory_peak_percent_max": round(
            100 * maximum[2].item() / max(maximum[3].item(), 1), 2
        ),
        "monitored_world_size": world_size,
    }
    if reset_peak_stats:
        torch.cuda.reset_peak_memory_stats(device)
    return metrics


class GpuUtilizationMonitor:
    """Continuously sample the CUDA device through NVML without stalling training.

    A point query at log time commonly lands between kernels and reports 0%.  The
    background sampler preserves the distribution observed throughout each log
    interval.  Every distributed rank samples its own physical device; callers
    combine the local summaries with :func:`distributed_gpu_utilization`.
    """

    def __init__(self, device: torch.device, *, sample_interval_seconds: float = 0.2) -> None:
        self.device = device
        self.sample_interval_seconds = sample_interval_seconds
        self._nvml: Any | None = None
        self._handle: Any | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float, float]] = []

    @property
    def available(self) -> bool:
        return self._handle is not None

    @staticmethod
    def _normalize_uuid(value: object) -> str:
        if isinstance(value, bytes):
            value = value.decode("ascii")
        return str(value).lower().removeprefix("gpu-").replace("-", "")

    def start(self) -> None:
        if self.device.type != "cuda" or self._thread is not None:
            return
        try:
            import pynvml
        except ImportError:
            return
        try:
            pynvml.nvmlInit()
            target_uuid = self._normalize_uuid(
                torch.cuda.get_device_properties(self.device).uuid
            )
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                if self._normalize_uuid(pynvml.nvmlDeviceGetUUID(handle)) == target_uuid:
                    self._handle = handle
                    break
            if self._handle is None:
                pynvml.nvmlShutdown()
                return
            self._nvml = pynvml
            self._sample()
            self._thread = threading.Thread(
                target=self._run,
                name=f"minitrain-nvml-{self.device.index}",
                daemon=True,
            )
            self._thread.start()
        except (pynvml.NVMLError, RuntimeError, OSError, AttributeError):
            self._nvml = None
            self._handle = None

    def _sample(self) -> None:
        if self._nvml is None or self._handle is None:
            return
        try:
            rates = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
        except self._nvml.NVMLError:
            return
        with self._lock:
            self._samples.append((float(rates.gpu), float(rates.memory)))

    def _run(self) -> None:
        while not self._stop.wait(self.sample_interval_seconds):
            self._sample()

    def read_interval(self) -> dict[str, float]:
        """Return this rank's interval distribution and begin a fresh interval."""

        self._sample()
        with self._lock:
            samples, self._samples = self._samples, []
        if not samples:
            return {}
        compute = [sample[0] for sample in samples]
        memory = [sample[1] for sample in samples]
        return {
            "gpu_compute_utilization_percent_local_min": min(compute),
            "gpu_compute_utilization_percent_local_mean": sum(compute) / len(compute),
            "gpu_compute_utilization_percent_local_max": max(compute),
            "gpu_memory_controller_utilization_percent_local_min": min(memory),
            "gpu_memory_controller_utilization_percent_local_mean": sum(memory) / len(memory),
            "gpu_memory_controller_utilization_percent_local_max": max(memory),
            "gpu_utilization_samples_local": float(len(samples)),
        }

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self.sample_interval_seconds))
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except self._nvml.NVMLError:
                pass
        self._thread = None
        self._handle = None
        self._nvml = None


def distributed_gpu_utilization(
    local: dict[str, float], device: torch.device
) -> dict[str, float]:
    """Combine per-rank NVML interval summaries into job-wide min/mean/max."""

    reduction_device = _distributed_device(device)
    available = torch.tensor(
        int(bool(local)), dtype=torch.int32, device=reduction_device
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(available, op=dist.ReduceOp.MIN)
    if not bool(available.item()):
        return {"gpu_utilization_available": 0}
    output: dict[str, float] = {}
    for prefix in ("gpu_compute_utilization_percent", "gpu_memory_controller_utilization_percent"):
        local_values = torch.tensor(
            [
                local[f"{prefix}_local_min"],
                local[f"{prefix}_local_mean"],
                local[f"{prefix}_local_max"],
            ],
            dtype=torch.float64,
            device=reduction_device,
        )
        minimum = local_values[0].clone()
        mean = local_values[1].clone()
        maximum = local_values[2].clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            dist.all_reduce(mean, op=dist.ReduceOp.SUM)
            mean /= dist.get_world_size()
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        output[f"{prefix}_min"] = round(minimum.item(), 2)
        output[f"{prefix}_mean"] = round(mean.item(), 2)
        output[f"{prefix}_max"] = round(maximum.item(), 2)
    sample_count = torch.tensor(
        local["gpu_utilization_samples_local"], dtype=torch.float64, device=reduction_device
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sample_count, op=dist.ReduceOp.MIN)
    output["gpu_utilization_samples_per_rank_min"] = int(sample_count.item())
    output["gpu_utilization_available"] = 1
    return output


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
            # The memory-stat API does not reliably lazy-initialize an explicitly
            # indexed CUDA device. Probe workers construct their reporter before
            # loading the model, so make the requested device current first.
            torch.cuda.set_device(self.device)
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
