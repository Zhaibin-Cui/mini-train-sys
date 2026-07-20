import json

import torch

from minitrain.runtime.config import LoggingConfig
from minitrain.runtime.logger import build_event_logger, format_console_event
from minitrain.runtime.monitoring import ProgressReporter


def test_jsonl_logger_persists_every_event(tmp_path):
    logger = build_event_logger(
        LoggingConfig(console=False, tensorboard=False, jsonl=True, log_dir=str(tmp_path)),
        run_name="audit",
        tensorboard_log_dir=tmp_path / "audit-run",
    )
    logger.log_event({"event": "init", "seed": 7})
    logger.log_event({"event": "train", "step": 1, "loss": 2.5})
    logger.close()

    rows = [json.loads(line) for line in (tmp_path / "audit-run" / "events.jsonl").read_text().splitlines()]
    assert rows == [
        {"event": "init", "seed": 7},
        {"event": "train", "step": 1, "loss": 2.5},
    ]


def test_console_progress_is_human_readable():
    rendered = format_console_event(
        {
            "event": "train",
            "step": 100,
            "step_total": 10_000,
            "batch": 100,
            "batches_total": 10_000,
            "epoch": 1,
            "epochs_total": 5,
            "loss": 2.125,
            "lr": 3e-4,
            "tokens_per_sec": 12_345,
            "gpu_memory_allocated_mb_max": 12_288,
            "gpu_memory_capacity_mb_max": 24_576,
            "progress_percent": 1.0,
            "eta_seconds": 3661,
        }
    )

    assert "batch 100/10000" in rendered
    assert "lr 3.000e-04" in rendered
    assert "gpu(max) 12.00/24.00 GiB" in rendered
    assert "ETA 1:01:01" in rendered


def test_probe_console_surfaces_health_and_pipeline_state():
    probe = format_console_event(
        {
            "event": "probe_train",
            "step": 10,
            "steps_total": 100,
            "loss": 1.0,
            "accuracy": 0.75,
            "grad_norm": 2.5,
            "data_wait_percent": 12.0,
            "progress_percent": 10.0,
            "eta_seconds": 90,
        }
    )
    assert "acc 0.7500" in probe
    assert "grad 2.500" in probe
    assert "data-wait 12.0%" in probe

    pipeline = format_console_event(
        {
            "event": "probe_pipeline",
            "phase": "training",
            "step": 3,
            "steps_total": 22,
            "tasks_running": 4,
            "tasks_queued": 15,
            "tasks_failed": 0,
            "task": "p_major_first",
            "action": "finished",
            "device": "cuda:0",
            "eta_seconds": 60,
        }
    )
    assert "tasks 3/22" in pipeline
    assert "running 4" in pipeline
    assert "p_major_first on cuda:0" in pipeline


def test_tensorboard_records_full_numeric_state(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dir = tmp_path / "tensorboard-run"
    logger = build_event_logger(
        LoggingConfig(console=False, tensorboard=True, jsonl=False, log_dir=str(tmp_path)),
        run_name="audit",
        tensorboard_log_dir=run_dir,
    )
    logger.log_event(
        {
            "event": "train",
            "step": 7,
            "loss": 1.25,
            "lr": 3e-4,
            "tokens_per_sec": 1234.0,
            "tokens_goal": 10_000,
            "gpu_memory_allocated_mb_max": 4096.0,
            "loss/lm_cross_entropy": 1.23,
            "loss/moe_aux_weighted": 0.01,
            "moe/expert_load_fraction/expert_00": 0.5,
            "moe/expert_load_fraction_by_layer": [[0.5, 0.5], [0.25, 0.75]],
        }
    )
    logger.close()

    accumulator = EventAccumulator(str(run_dir)).Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert {
        "train/loss",
        "train/lr",
        "train/tokens_per_sec",
        "train/tokens_goal",
        "train/gpu_memory_allocated_mb_max",
        "train/loss/lm_cross_entropy",
        "train/loss/moe_aux_weighted",
        "train/moe/expert_load_fraction/expert_00",
    } <= tags
    assert (
        "train/moe/expert_load_fraction_by_layer/balance_heatmap"
        in accumulator.Tags()["images"]
    )
    assert (
        "train/moe/expert_load_fraction_by_layer/ratio_histogram"
        in accumulator.Tags()["histograms"]
    )


def test_progress_reporter_uses_standard_batch_plural():
    class CaptureLogger:
        def __init__(self):
            self.payload = None

        def log_event(self, payload):
            self.payload = payload

        def close(self):
            pass

    logger = CaptureLogger()
    reporter = ProgressReporter("evaluate", 2, logger, torch.device("cpu"), unit="batch")
    reporter.update(1, items=4, tokens=32)

    assert logger.payload["batch"] == 1
    assert logger.payload["batches_total"] == 2
    assert "batchs_total" not in logger.payload
