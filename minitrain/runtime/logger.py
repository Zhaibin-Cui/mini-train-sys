from __future__ import annotations

import json
import os
import struct
import time
import warnings
import zlib
from pathlib import Path
from typing import Protocol

import torch

from minitrain.runtime.config import LoggingConfig


class EventLogger(Protocol):
    """Small logging contract used by training scripts.

    The trainer should not know whether metrics go to stdout, TensorBoard, or a
    future service. Keeping this interface tiny makes those outputs swappable.
    """

    def log_event(self, payload: dict[str, object]) -> None:
        ...

    def close(self) -> None:
        ...


def is_primary_rank() -> bool:
    """Return True for the one process that should write human-facing logs."""

    return int(os.environ.get("RANK", "0")) == 0


class NullLogger:
    def log_event(self, payload: dict[str, object]) -> None:
        return None

    def close(self) -> None:
        return None


class ConsoleLogger:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def log_event(self, payload: dict[str, object]) -> None:
        if self.enabled:
            print(format_console_event(payload), flush=True)

    def close(self) -> None:
        return None


def _duration(seconds: object) -> str:
    try:
        value = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "?"
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _number(payload: dict[str, object], key: str, digits: int = 3) -> str:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value:,.{digits}f}"


def format_console_event(payload: dict[str, object]) -> str:
    """Render structured events as stable, human-readable one-line progress."""

    event = str(payload.get("event", "event"))
    if event == "probe_pipeline":
        parts = [
            f"[probe_pipeline:{payload.get('phase', '?')}] "
            f"tasks {payload.get('step', '?')}/{payload.get('steps_total', '?')}"
        ]
        parts.append(
            f"running {payload.get('tasks_running', 0)} | "
            f"queued {payload.get('tasks_queued', 0)} | "
            f"failed {payload.get('tasks_failed', 0)}"
        )
        if payload.get("task"):
            parts.append(
                f"{payload.get('action', 'update')} {payload.get('task')} "
                f"on {payload.get('device', '?')}"
            )
        parts.append(f"ETA {_duration(payload.get('eta_seconds'))}")
        return " | ".join(parts)
    if event in {
        "train",
        "probe_train",
        "probe_validation",
        "evaluate",
        "analyze",
        "model_load",
        "prepare",
    }:
        unit = "batch"
        total_keys = {"batch": "batches_total", "step": "steps_total", "example": "examples_total"}
        for candidate in ("batch", "step", "example"):
            if candidate in payload and total_keys[candidate] in payload:
                unit = candidate
                break
        completed = payload.get(unit, payload.get("step", "?"))
        total = payload.get(total_keys[unit], payload.get("step_total", "?"))
        parts = [f"[{event}] {unit} {completed}/{total}"]
        if "epoch" in payload and "epochs_total" in payload:
            parts.append(f"epoch {payload['epoch']}/{payload.get('epochs_total', '?')}")
        if "loss" in payload:
            parts.append(f"loss {_number(payload, 'loss', 5)}")
        if "lr" in payload:
            lr = payload["lr"]
            parts.append(f"lr {float(lr):.3e}" if isinstance(lr, (int, float)) else "lr -")
        accuracy_key = "accuracy" if "accuracy" in payload else "accuracy_running"
        if accuracy_key in payload:
            parts.append(f"acc {_number(payload, accuracy_key, 4)}")
        if "grad_norm" in payload:
            parts.append(f"grad {_number(payload, 'grad_norm', 3)}")
        if "data_wait_percent" in payload:
            parts.append(f"data-wait {_number(payload, 'data_wait_percent', 1)}%")
        if "tokens_per_sec" in payload:
            parts.append(f"tok/s {_number(payload, 'tokens_per_sec', 1)}")
        if "items_per_sec" in payload and not payload.get("tokens_per_sec"):
            parts.append(f"items/s {_number(payload, 'items_per_sec', 1)}")
        if "gpu_peak_memory_allocated_mb_max" in payload:
            used = float(payload["gpu_peak_memory_allocated_mb_max"]) / 1024
            capacity = float(payload.get("gpu_memory_capacity_mb_max", 0)) / 1024
            parts.append(f"gpu-peak {used:.2f}/{capacity:.2f} GiB")
        elif "host_peak_memory_mb" in payload:
            parts.append(f"host-peak {_number(payload, 'host_peak_memory_mb', 1)} MiB")
        if "gpu_compute_utilization_percent_mean" in payload:
            parts.append(
                f"gpu-util {_number(payload, 'gpu_compute_utilization_percent_mean', 1)}%"
            )
        parts.append(f"{_number(payload, 'progress_percent', 1)}%")
        parts.append(f"ETA {_duration(payload.get('eta_seconds'))}")
        return " | ".join(parts)
    if event == "checkpoint":
        return f"[checkpoint] saved {payload.get('path')} (model+optimizer state)"
    if event == "resume":
        return (
            f"[resume] {payload.get('path')} | epoch={payload.get('epoch')} "
            f"step={payload.get('step')} world={payload.get('saved_world_size')}"
        )
    if event == "init":
        return (
            f"[init] run={payload.get('run')} device={payload.get('device')} "
            f"strategy={payload.get('parallel')} world={payload.get('world_size')} "
            f"params={payload.get('params')}"
        )
    return f"[{event}] {json.dumps(payload, ensure_ascii=False, default=str)}"


class JsonlLogger:
    """Append every structured event to a durable, line-oriented run log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")

    def log_event(self, payload: dict[str, object]) -> None:
        self.handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class TensorBoardLogger:
    """Write scalar training events to TensorBoard.

    TensorBoard prefers numeric scalar streams. For an event like
    {"event": "train", "step": 3, "loss": 2.1}, this logger writes
    train/loss at global step 3.
    """

    def __init__(self, *, log_dir: str | Path, flush_secs: int = 10) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TensorBoard logging is enabled, but the 'tensorboard' package is not installed. "
                "Install project dependencies or set logging.tensorboard=false."
            ) from exc

        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=flush_secs)

    def log_event(self, payload: dict[str, object]) -> None:
        event = str(payload.get("event", "event"))
        step = payload.get("step")
        if not isinstance(step, int):
            self._log_text_event(event, payload)
            return None

        for key, value in payload.items():
            if key in {"event", "step"}:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{event}/{key}", value, step)
                continue
            tensor = self._numeric_tensor(value)
            if tensor is not None:
                self._log_tensor(event, key, tensor, step)
        return None

    @staticmethod
    def _numeric_tensor(value: object) -> torch.Tensor | None:
        """Convert JSON-safe numeric arrays while rejecting ragged/non-finite data."""

        if not isinstance(value, (list, tuple)):
            return None
        try:
            tensor = torch.tensor(value, dtype=torch.float32)
        except (TypeError, ValueError):
            return None
        if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
            return None
        return tensor

    def _log_tensor(
        self, event: str, key: str, tensor: torch.Tensor, step: int
    ) -> None:
        tag = f"{event}/{key}"
        if key in {
            "moe/expert_load_fraction_by_layer",
            "moe/expert_probability_by_layer",
        } and tensor.ndim == 2:
            # A fixed 0x..2x-uniform scale makes images comparable over time:
            # blue=under-used, white=balanced, red=over-used. Rows are layers,
            # columns are experts. Per-expert scalar curves retain exact labels.
            expert_ratio = tensor * tensor.shape[1]
            clipped = expert_ratio.clamp(0.0, 2.0)
            ones = torch.ones_like(clipped)
            heatmap = torch.stack(
                (
                    torch.minimum(clipped, ones),
                    1.0 - (clipped - 1.0).abs(),
                    torch.minimum(2.0 - clipped, ones),
                )
            ).clamp(0.0, 1.0)
            self._add_rgb_image(f"{tag}/balance_heatmap", heatmap, step)
            self.writer.add_histogram(
                f"{tag}/ratio_histogram", expert_ratio.flatten(), step
            )
            return
        self.writer.add_histogram(tag, tensor.flatten(), step)

    def _add_rgb_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        """Write a tiny RGB PNG directly, avoiding an optional Pillow dependency."""

        from tensorboard.compat.proto.summary_pb2 import Summary

        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("TensorBoard RGB images must have shape [3, height, width]")
        pixels = (
            image.mul(255)
            .round()
            .to(dtype=torch.uint8, device="cpu")
            .permute(1, 2, 0)
            .contiguous()
        )
        height, width, _ = pixels.shape
        scanlines = b"".join(
            b"\x00" + bytes(pixels[row].flatten().tolist()) for row in range(height)
        )

        def png_chunk(kind: bytes, data: bytes) -> bytes:
            checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

        encoded = b"".join(
            (
                b"\x89PNG\r\n\x1a\n",
                png_chunk(
                    b"IHDR",
                    struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
                ),
                png_chunk(b"IDAT", zlib.compress(scanlines)),
                png_chunk(b"IEND", b""),
            )
        )
        summary = Summary(
            value=[
                Summary.Value(
                    tag=tag,
                    image=Summary.Image(
                        height=height,
                        width=width,
                        colorspace=3,
                        encoded_image_string=encoded,
                    ),
                )
            ]
        )
        self.writer._get_file_writer().add_summary(summary, step)

    def _log_text_event(self, event: str, payload: dict[str, object]) -> None:
        # Init events are useful in TensorBoard's text panel because they record
        # exactly which backend, device, and config were used for this run.
        lines = [f"- {key}: {value}" for key, value in sorted(payload.items())]
        self.writer.add_text(event, "\n".join(lines), global_step=0)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


class CompositeLogger:
    def __init__(self, loggers: list[EventLogger]) -> None:
        self.loggers = loggers

    def log_event(self, payload: dict[str, object]) -> None:
        for logger in self.loggers:
            logger.log_event(payload)

    def close(self) -> None:
        for logger in reversed(self.loggers):
            logger.close()


def make_run_log_dir(base_dir: str | Path, run_name: str) -> Path:
    """Create a stable, readable TensorBoard directory for one training run."""

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path(base_dir) / run_name / timestamp


def get_run_log_dir(cfg: LoggingConfig, *, run_name: str) -> Path | None:
    if not is_primary_rank() or not (cfg.tensorboard or cfg.jsonl):
        return None
    return make_run_log_dir(cfg.log_dir, run_name)


def get_tensorboard_log_dir(cfg: LoggingConfig, *, run_name: str) -> Path | None:
    """Backward-compatible helper for callers interested only in TensorBoard."""

    if not cfg.tensorboard:
        return None
    return get_run_log_dir(cfg, run_name=run_name)


def build_event_logger(
    cfg: LoggingConfig,
    *,
    run_name: str,
    tensorboard_log_dir: str | Path | None = None,
) -> EventLogger:
    """Build the configured logger stack.

    In distributed runs only rank 0 writes TensorBoard/console logs. The other
    ranks get a NullLogger so the training loop can call the logger unconditionally.
    """

    if not is_primary_rank():
        return NullLogger()

    loggers: list[EventLogger] = []
    if cfg.console:
        loggers.append(ConsoleLogger())
    run_log_dir = Path(tensorboard_log_dir) if tensorboard_log_dir else None
    if cfg.jsonl:
        run_log_dir = run_log_dir or make_run_log_dir(cfg.log_dir, run_name)
        loggers.append(JsonlLogger(run_log_dir / "events.jsonl"))
    if cfg.tensorboard:
        log_dir = run_log_dir or make_run_log_dir(cfg.log_dir, run_name)
        try:
            loggers.append(
                TensorBoardLogger(
                    log_dir=log_dir,
                    flush_secs=cfg.flush_secs,
                )
            )
        except RuntimeError as exc:
            warnings.warn(f"{exc} Continuing without TensorBoard.", stacklevel=2)
    if not loggers:
        return NullLogger()
    return CompositeLogger(loggers)
