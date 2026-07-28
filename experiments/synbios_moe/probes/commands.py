"""Build the subprocess commands used by the probe scheduler."""

import sys
from collections.abc import Callable
from pathlib import Path

from experiments.synbios_moe.probes.spec import (
    JobCommand,
    ProbeJob,
    ProbeStepSchedule,
    steps_for_job,
)


def probe_train_command_builder(
    *,
    script: Path,
    data: Path,
    cache: Path,
    model_config: Path,
    checkpoint: Path,
    output_dir: Path,
    steps: ProbeStepSchedule,
    seed: int,
    quiet: bool,
    log_interval: int,
    tensorboard: bool,
    batch_sizes: dict[str, int],
    validation_batch_sizes: dict[str, int],
    checkpoint_interval_steps: int,
    evaluate_train: bool,
    checkpoint_model_sha256: str | None = None,
) -> Callable[[ProbeJob, str], JobCommand]:
    def build(job: ProbeJob, device: str) -> JobCommand:
        output = output_dir / "training" / f"{job.key}.json"
        command = [
            sys.executable,
            str(script),
            "probe",
            "--kind",
            job.kind,
            "--data",
            str(data),
            "--probe-cache",
            str(cache),
            "--model-config",
            str(model_config),
            "--checkpoint",
            str(checkpoint),
            "--attribute",
            job.attribute,
            "--target",
            job.target,
            "--steps",
            str(steps_for_job(steps, job)),
            "--batch-size",
            str(batch_sizes[job.kind]),
            "--evaluation-batch-size",
            str(validation_batch_sizes[job.kind]),
            "--seed",
            str(seed),
            "--device",
            device,
            "--output",
            str(output),
            "--log-interval",
            str(log_interval),
            "--recovery-checkpoint",
            str(output_dir / "recovery" / f"{job.key}.pt"),
            "--checkpoint-interval-steps",
            str(checkpoint_interval_steps),
            "--skip-final-validation",
        ]
        if evaluate_train:
            command.append("--evaluate-train")
        if checkpoint_model_sha256 is not None:
            command.extend(("--checkpoint-model-sha256", checkpoint_model_sha256))
        if quiet:
            command.append("--quiet")
        if not tensorboard:
            command.append("--no-tensorboard")
        return JobCommand(
            command,
            output.with_suffix(".pt"),
            output_dir / "logs" / f"train_{job.key}.log",
            events_root=output.parent / "operation_logs",
        )

    return build


def probe_validation_command_builder(
    *,
    script: Path,
    data: Path,
    cache: Path,
    model_config: Path,
    checkpoint: Path,
    output_dir: Path,
    quiet: bool,
    log_interval: int,
    tensorboard: bool,
    validation_batch_sizes: dict[str, int],
    checkpoint_model_sha256: str | None = None,
) -> Callable[[ProbeJob, str], JobCommand]:
    def build(job: ProbeJob, device: str) -> JobCommand:
        output = output_dir / "validation" / f"{job.key}.json"
        probe_checkpoint = output_dir / "training" / f"{job.key}.pt"
        command = [
            sys.executable,
            str(script),
            "validate-probe",
            "--data",
            str(data),
            "--probe-cache",
            str(cache),
            "--model-config",
            str(model_config),
            "--checkpoint",
            str(checkpoint),
            "--probe-checkpoint",
            str(probe_checkpoint),
            "--device",
            device,
            "--output",
            str(output),
            "--log-interval",
            str(log_interval),
            "--batch-size",
            str(validation_batch_sizes[job.kind]),
        ]
        if quiet:
            command.append("--quiet")
        if checkpoint_model_sha256 is not None:
            command.extend(("--checkpoint-model-sha256", checkpoint_model_sha256))
        if not tensorboard:
            command.append("--no-tensorboard")
        return JobCommand(
            command,
            output,
            output_dir / "logs" / f"validation_{job.key}.log",
            dependencies=(probe_checkpoint,),
            events_root=output.parent / "operation_logs",
        )

    return build
