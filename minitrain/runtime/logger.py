import os
import time
from pathlib import Path
from typing import Protocol

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
            print(payload, flush=True)

    def close(self) -> None:
        return None


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
        return None

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


def get_tensorboard_log_dir(cfg: LoggingConfig, *, run_name: str) -> Path | None:
    if not is_primary_rank() or not cfg.tensorboard:
        return None
    return make_run_log_dir(cfg.log_dir, run_name)


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
    if cfg.tensorboard:
        log_dir = tensorboard_log_dir or make_run_log_dir(cfg.log_dir, run_name)
        loggers.append(
            TensorBoardLogger(
                log_dir=log_dir,
                flush_secs=cfg.flush_secs,
            )
        )
    if not loggers:
        return NullLogger()
    return CompositeLogger(loggers)
