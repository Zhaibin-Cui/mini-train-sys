"""Task-level scheduling and result reduction for the SynBioS probe experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import torch
import yaml

from experiments.synbios_moe.probe_data import paper_probe_tasks


PIPELINE_PROTOCOL_VERSION = 4
CLOZE_GATE_PROTOCOL = "progressive_original_biography_cloze_greedy"


class PipelineEventLogger(Protocol):
    def log_event(self, payload: dict[str, object]) -> None: ...


@dataclass(frozen=True)
class ProbeJob:
    kind: str
    attribute: str
    target: str

    @property
    def key(self) -> str:
        return f"{self.kind}_{self.attribute}_{self.target}"


@dataclass(frozen=True)
class JobCommand:
    command: list[str]
    output: Path
    log: Path
    dependencies: tuple[Path, ...] = ()
    events_root: Path | None = None


@dataclass(frozen=True)
class ProbeRuntimeConfig:
    """Operational knobs kept separate from the scientific task matrix."""

    p_batch_size: int = 50
    q_batch_size: int = 200
    p_validation_batch_size: int = 50
    q_validation_batch_size: int = 200
    log_interval_steps: int = 100
    heartbeat_seconds: float = 10.0
    checkpoint_interval_steps: int = 1000
    evaluate_train: bool = True

    def __post_init__(self) -> None:
        numeric = (
            self.p_batch_size,
            self.q_batch_size,
            self.p_validation_batch_size,
            self.q_validation_batch_size,
            self.log_interval_steps,
            self.heartbeat_seconds,
            self.checkpoint_interval_steps,
        )
        if any(value <= 0 for value in numeric):
            raise ValueError("probe runtime numeric settings must be positive")

    @classmethod
    def from_config(cls, config: dict) -> "ProbeRuntimeConfig":
        runtime = dict(config.get("runtime", {}))
        training = dict(runtime.pop("training_batch_sizes", {}))
        validation = dict(runtime.pop("validation_batch_sizes", {}))
        unknown_training = sorted(set(training) - {"p", "q"})
        unknown_validation = sorted(set(validation) - {"p", "q"})
        if unknown_training or unknown_validation:
            unknown = [
                *(f"training_batch_sizes.{key}" for key in unknown_training),
                *(f"validation_batch_sizes.{key}" for key in unknown_validation),
            ]
            raise ValueError("unknown probe runtime settings: " + ", ".join(unknown))
        instance = cls(
            p_batch_size=int(training.get("p", cls.p_batch_size)),
            q_batch_size=int(training.get("q", cls.q_batch_size)),
            p_validation_batch_size=int(validation.get("p", cls.p_validation_batch_size)),
            q_validation_batch_size=int(validation.get("q", cls.q_validation_batch_size)),
            log_interval_steps=int(runtime.pop("log_interval_steps", cls.log_interval_steps)),
            heartbeat_seconds=float(runtime.pop("heartbeat_seconds", cls.heartbeat_seconds)),
            checkpoint_interval_steps=int(
                runtime.pop("checkpoint_interval_steps", cls.checkpoint_interval_steps)
            ),
            evaluate_train=bool(runtime.pop("evaluate_train", cls.evaluate_train)),
        )
        if runtime:
            raise ValueError("unknown probe runtime settings: " + ", ".join(sorted(runtime)))
        return instance

    def with_overrides(self, **overrides: object) -> "ProbeRuntimeConfig":
        values = {
            name: getattr(self, name) if value is None else value
            for name, value in overrides.items()
        }
        payload = {**self.__dict__, **values}
        return type(self)(**payload)

    def as_dict(self) -> dict[str, object]:
        return {
            "training_batch_sizes": {"p": self.p_batch_size, "q": self.q_batch_size},
            "validation_batch_sizes": {
                "p": self.p_validation_batch_size,
                "q": self.q_validation_batch_size,
            },
            "log_interval_steps": self.log_interval_steps,
            "heartbeat_seconds": self.heartbeat_seconds,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "evaluate_train": self.evaluate_train,
        }


def all_probe_jobs() -> tuple[ProbeJob, ...]:
    return tuple(
        ProbeJob(kind, task.attribute, task.target)
        for task in paper_probe_tasks()
        for kind in ("p", "q")
    )


def load_pipeline_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("stages"), dict):
        raise ValueError("probe pipeline config must contain a stages mapping")
    return payload


def jobs_for_stage(config: dict, stage: str) -> tuple[int, tuple[ProbeJob, ...], str | None]:
    try:
        stage_cfg = config["stages"][stage]
    except KeyError as exc:
        raise ValueError(f"unknown probe stage: {stage}") from exc
    steps = int(stage_cfg["steps"])
    if steps <= 0:
        raise ValueError("stage steps must be positive")
    selected = stage_cfg.get("tasks", "all")
    if selected == "all":
        jobs = all_probe_jobs()
    else:
        if not isinstance(selected, list):
            raise ValueError("stage tasks must be 'all' or a list")
        jobs = tuple(
            ProbeJob(str(item["kind"]), str(item["attribute"]), str(item["target"]))
            for item in selected
        )
    if not jobs:
        raise ValueError("probe stage must contain at least one job")
    keys = [job.key for job in jobs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError("duplicate probe jobs in stage config: " + ", ".join(duplicates))
    valid = {job.key for job in all_probe_jobs()}
    invalid = [job.key for job in jobs if job.key not in valid]
    if invalid:
        raise ValueError("invalid probe jobs in stage config: " + ", ".join(invalid))
    return steps, jobs, stage_cfg.get("requires")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pipeline_identity(
    *,
    stage: str,
    steps: int,
    jobs: Iterable[ProbeJob],
    seed: int,
    data: str | Path,
    cache: str | Path,
    model_config: str | Path,
    checkpoint: str | Path,
    runtime: dict[str, object] | None = None,
) -> dict[str, object]:
    """Fingerprint every input that changes durable probe outputs."""

    data_root = Path(data).resolve()
    cache_root = Path(cache).resolve()
    model_path = Path(model_config).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    model_export = checkpoint_path if checkpoint_path.is_file() else checkpoint_path / "model.pt"
    required_files = {
        "dataset manifest": data_root / "manifest.json",
        "probe cache manifest": cache_root / "manifest.json",
        "model config": model_path,
        "checkpoint model export": model_export,
    }
    missing = [f"{name}: {path}" for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing pipeline input(s): " + "; ".join(missing))
    identity = {
        "protocol_version": PIPELINE_PROTOCOL_VERSION,
        "stage": stage,
        "steps": int(steps),
        "jobs": [job.key for job in jobs],
        "seed": int(seed),
        "data": str(data_root),
        "data_manifest_sha256": _sha256_file(required_files["dataset manifest"]),
        "probe_cache": str(cache_root),
        "probe_cache_manifest_sha256": _sha256_file(required_files["probe cache manifest"]),
        "model_config": str(model_path),
        "model_config_sha256": _sha256_file(model_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_model_sha256": _sha256_file(model_export),
    }
    if runtime is not None:
        identity["runtime"] = runtime
    return identity


def common_pipeline_identity(identity: dict[str, object]) -> dict[str, object]:
    """Return fields that must match across smoke, pilot, and formal stages."""

    stage_fields = {"stage", "steps", "jobs"}
    return {key: value for key, value in identity.items() if key not in stage_fields}


def reusable_cloze_gate(candidate: dict[str, object], identity: dict[str, object]) -> bool:
    """Return whether a cached gate uses the current strict generation protocol."""

    return (
        candidate.get("protocol") == CLOZE_GATE_PROTOCOL
        and candidate.get("identity") == common_pipeline_identity(identity)
        and isinstance(candidate.get("micro_field_accuracy"), (int, float))
    )


def require_matching_identity(
    existing: dict[str, object], expected: dict[str, object], *, label: str
) -> None:
    actual = existing.get("identity")
    if actual == expected:
        return
    if not isinstance(actual, dict):
        raise ValueError(f"{label} predates pipeline identity tracking and cannot be reused")
    mismatches = sorted(
        key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)
    )
    raise ValueError(f"{label} does not match this run: " + ", ".join(mismatches))


def write_json_atomic(path: str | Path, payload: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


class ProbePipelineState:
    """Own durable pipeline state and task-level monitoring in one place."""

    def __init__(
        self,
        *,
        pipeline_path: str | Path,
        events_path: str | Path,
        base_state: dict[str, object],
        jobs: Iterable[ProbeJob],
        logger: PipelineEventLogger,
    ) -> None:
        self.pipeline_path = Path(pipeline_path)
        self.events_path = Path(events_path)
        self.base_state = dict(base_state)
        self.jobs = tuple(jobs)
        self.logger = logger

    def write(self, status: str, **fields: object) -> None:
        write_json_atomic(
            self.pipeline_path,
            {"status": status, **self.base_state, **fields},
        )

    def monitor_phase(
        self, phase: str, *, extra_state: dict[str, object] | None = None
    ) -> Callable[[dict[str, object]], None]:
        started = time.monotonic()
        task_status = {job.key: "queued" for job in self.jobs}
        total = len(task_status)

        def monitor(event: dict[str, object]) -> None:
            job = str(event["job"])
            action = str(event["action"])
            if action == "started":
                task_status[job] = "running"
            elif action == "finished":
                task_status[job] = str(event["status"])
            completed = sum(
                status in {"completed", "failed", "skipped_existing"}
                for status in task_status.values()
            )
            running = sum(status == "running" for status in task_status.values())
            failed = sum(status == "failed" for status in task_status.values())
            elapsed = time.monotonic() - started
            eta = elapsed / completed * (total - completed) if completed else None
            payload = {
                "event": "probe_pipeline",
                "phase": phase,
                "action": action,
                "task": job,
                "device": event.get("device"),
                "status": event.get("status", "running"),
                "step": completed,
                "steps_total": total,
                "tasks_running": running,
                "tasks_queued": total - completed - running,
                "tasks_failed": failed,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "progress_percent": 100.0 * completed / max(total, 1),
            }
            payload.update(
                {
                    key: value
                    for key, value in event.items()
                    if key.startswith("worker_") and value is not None
                }
            )
            self.logger.log_event(payload)
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({**payload, **event}, default=str) + "\n")
            self.write(
                "running",
                **(extra_state or {}),
                active_phase=phase,
                task_status=task_status,
                live=payload,
            )

        return monitor


def resolve_devices(spec: str, num_gpus: int | None = None) -> tuple[str, ...]:
    """Resolve auto/N-GPU or explicit device lists without changing task semantics."""

    if spec == "auto":
        available = torch.cuda.device_count()
        if available == 0:
            if num_gpus not in (None, 0):
                raise ValueError(f"requested {num_gpus} GPUs, but CUDA is unavailable")
            return ("cpu",)
        count = available if num_gpus is None else num_gpus
        if count <= 0 or count > available:
            raise ValueError(f"num_gpus must be in [1, {available}], got {count}")
        return tuple(f"cuda:{index}" for index in range(count))
    if num_gpus is not None:
        raise ValueError("--num-gpus is only valid with --devices auto")
    devices = tuple(part.strip() for part in spec.split(",") if part.strip())
    if not devices:
        raise ValueError("device list is empty")
    normalized = tuple(f"cuda:{item}" if item.isdigit() else item for item in devices)
    for device in normalized:
        torch.device(device)
    return normalized


def _run_one(
    command: list[str],
    log_path: Path,
    *,
    heartbeat_seconds: float = 30.0,
    on_heartbeat: Callable[[float], None] | None = None,
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        next_heartbeat = started + heartbeat_seconds
        while process.poll() is None:
            now = time.monotonic()
            if on_heartbeat is not None and now >= next_heartbeat:
                on_heartbeat(now - started)
                next_heartbeat = now + heartbeat_seconds
            time.sleep(min(0.5, heartbeat_seconds))
    return process.returncode, time.monotonic() - started


def _latest_worker_progress(events_root: Path | None, job_key: str) -> dict[str, object]:
    """Read the newest structured worker event without coupling scheduling to training."""

    if events_root is None or not events_root.is_dir():
        return {}
    candidates = list(events_root.glob(f"synbios_*_{job_key}/*/events.jsonl"))
    if not candidates:
        return {}
    path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 64 * 1024))
            lines = handle.read().splitlines()
        for line in reversed(lines):
            payload = json.loads(line)
            if payload.get("event") not in {
                "probe_train",
                "probe_train_evaluation",
                "probe_validation",
                "probe_checkpoint",
            }:
                continue
            fields: dict[str, object] = {
                "worker_event": payload.get("event"),
                "worker_step": payload.get("step"),
                "worker_progress_percent": payload.get("progress_percent"),
                "worker_eta_seconds": payload.get("eta_seconds"),
            }
            total = payload.get("steps_total", payload.get("batches_total"))
            if total is not None:
                fields["worker_steps_total"] = total
            for name in (
                "loss",
                "accuracy",
                "accuracy_running",
                "lr",
                "grad_norm",
                "items_per_sec",
                "gpu_peak_memory_allocated_mb_max",
                "gpu_memory_capacity_mb_max",
                "gpu_compute_utilization_percent_local_mean",
            ):
                if name in payload:
                    fields[f"worker_{name}"] = payload[name]
            return fields
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return {}


def schedule_jobs(
    jobs: Iterable[ProbeJob],
    devices: tuple[str, ...],
    command_builder: Callable[[ProbeJob, str], JobCommand],
    *,
    on_event: Callable[[dict[str, object]], None] | None = None,
    heartbeat_seconds: float = 30.0,
    reuse_existing: bool = True,
) -> list[dict[str, object]]:
    """Run at most one independent probe process per configured device."""

    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")

    pending: queue.Queue[ProbeJob] = queue.Queue()
    for job in jobs:
        pending.put(job)
    results: list[dict[str, object]] = []
    result_lock = threading.Lock()

    def worker(device: str) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                spec = command_builder(job, device)
                with result_lock:
                    if on_event is not None:
                        on_event({"action": "started", "job": job.key, "device": device})
                is_current = (
                    reuse_existing
                    and spec.output.is_file()
                    and all(
                        dependency.is_file()
                        and spec.output.stat().st_mtime_ns >= dependency.stat().st_mtime_ns
                        for dependency in spec.dependencies
                    )
                )
                if is_current:
                    record = {
                        "job": job.key,
                        "device": device,
                        "status": "skipped_existing",
                        "output": str(spec.output),
                        "seconds": 0.0,
                    }
                else:
                    previous_output_mtime = (
                        spec.output.stat().st_mtime_ns if spec.output.is_file() else None
                    )
                    missing_dependencies = [
                        str(dependency)
                        for dependency in spec.dependencies
                        if not dependency.is_file()
                    ]
                    if missing_dependencies:
                        raise FileNotFoundError(
                            "missing job dependencies: " + ", ".join(missing_dependencies)
                        )

                    def heartbeat(elapsed: float) -> None:
                        with result_lock:
                            if on_event is not None:
                                on_event(
                                    {
                                        "action": "heartbeat",
                                        "job": job.key,
                                        "device": device,
                                        "seconds": elapsed,
                                        "log": str(spec.log),
                                        **_latest_worker_progress(spec.events_root, job.key),
                                    }
                                )

                    returncode, seconds = _run_one(
                        spec.command,
                        spec.log,
                        heartbeat_seconds=heartbeat_seconds,
                        on_heartbeat=heartbeat,
                    )
                    status = "completed" if returncode == 0 else "failed"
                    error = None
                    if returncode == 0 and not spec.output.is_file():
                        status = "failed"
                        error = "process exited successfully but did not create its output marker"
                    elif (
                        returncode == 0
                        and previous_output_mtime is not None
                        and spec.output.stat().st_mtime_ns <= previous_output_mtime
                    ):
                        status = "failed"
                        error = "process exited successfully but did not refresh its output marker"
                    record = {
                        "job": job.key,
                        "device": device,
                        "status": status,
                        "returncode": returncode,
                        "output": str(spec.output),
                        "log": str(spec.log),
                        "seconds": seconds,
                    }
                    if error is not None:
                        record["error"] = error
            except Exception as exc:
                record = {
                    "job": job.key,
                    "device": device,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds": 0.0,
                }
            finally:
                with result_lock:
                    results.append(record)
                    if on_event is not None:
                        on_event({"action": "finished", **record})
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(device,)) for device in devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sorted(results, key=lambda item: str(item["job"]))


def probe_train_command_builder(
    *,
    script: Path,
    data: Path,
    cache: Path,
    model_config: Path,
    checkpoint: Path,
    output_dir: Path,
    steps: int,
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
            str(steps),
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
        # The .pt file is the durable completion marker required by validation.
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


def summarize_probe_results(
    named_directories: dict[str, Path],
    output_dir: str | Path,
    *,
    expected_jobs: Iterable[ProbeJob] | None = None,
) -> dict[str, object]:
    """Reduce independent validation JSON files into JSON and tidy CSV tables."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not named_directories:
        raise ValueError("at least one probe result directory is required")
    expected_keys = {job.key for job in expected_jobs} if expected_jobs is not None else None
    rows: list[dict[str, object]] = []
    run_payloads: dict[str, dict[str, object]] = {}
    for run_name, directory in named_directories.items():
        task_payloads = {}
        profile_fingerprints = set()
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            required = {"kind", "attribute", "target", "validation_accuracy"}
            if not required.issubset(payload):
                continue
            key = f"{payload['kind']}_{payload['attribute']}_{payload['target']}"
            if key in task_payloads:
                raise ValueError(f"duplicate validation result for task {key!r}")
            expected_positions = 6 if payload["kind"] == "p" else 1
            if len(payload["validation_accuracy"]) != expected_positions:
                raise ValueError(
                    f"task {key!r} has {len(payload['validation_accuracy'])} positions; "
                    f"expected {expected_positions}"
                )
            if int(payload["classes"]) <= 0 or int(payload["examples"]) <= 0:
                raise ValueError(f"task {key!r} has empty classes or examples")
            task_payloads[key] = payload
            dataset_manifest = payload.get("dataset_manifest", {})
            profile_hash = dataset_manifest.get("files", {}).get("profiles.jsonl", {}).get("sha256")
            if profile_hash:
                profile_fingerprints.add(profile_hash)
            for position, accuracy in enumerate(payload["validation_accuracy"]):
                rows.append(
                    {
                        "run": run_name,
                        "task": key,
                        "kind": payload["kind"],
                        "attribute": payload["attribute"],
                        "target": payload["target"],
                        "position": position,
                        "accuracy": float(accuracy),
                        "classes": int(payload["classes"]),
                        "examples": int(payload["examples"]),
                    }
                )
        if len(profile_fingerprints) > 1:
            raise ValueError(f"validation files for run {run_name!r} use different profiles")
        task_keys = set(task_payloads)
        if not task_keys:
            raise ValueError(f"no probe validation results found for run {run_name!r}")
        if expected_keys is not None and task_keys != expected_keys:
            missing = sorted(expected_keys - task_keys)
            unexpected = sorted(task_keys - expected_keys)
            raise ValueError(
                f"incomplete probe results for run {run_name!r}; "
                f"missing={missing}, unexpected={unexpected}"
            )
        run_payloads[run_name] = {
            "directory": str(directory.resolve()),
            "tasks": task_payloads,
            "profiles_sha256": next(iter(profile_fingerprints), None),
        }
    comparable_fingerprints = {
        payload["profiles_sha256"]
        for payload in run_payloads.values()
        if payload["profiles_sha256"] is not None
    }
    if len(comparable_fingerprints) > 1:
        raise ValueError("probe runs cannot be compared: profiles.jsonl fingerprints differ")
    task_sets = {frozenset(payload["tasks"]) for payload in run_payloads.values()}
    if len(task_sets) > 1:
        raise ValueError("probe runs cannot be compared: task sets differ")
    position_sets = {
        run_name: {
            (str(row["task"]), int(row["position"])) for row in rows if row["run"] == run_name
        }
        for run_name in named_directories
    }
    if len({frozenset(value) for value in position_sets.values()}) > 1:
        raise ValueError("probe runs cannot be compared: observation positions differ")
    summary = {"runs": run_payloads, "rows": rows}
    fields = [
        "run",
        "task",
        "kind",
        "attribute",
        "target",
        "position",
        "accuracy",
        "classes",
        "examples",
    ]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    run_names = list(named_directories)
    comparison_path = output / "comparison.csv"
    if len(run_names) >= 2:
        baseline = run_names[0]
        by_key = {(row["run"], row["task"], row["position"]): row for row in rows}
        comparisons = []
        for candidate in run_names[1:]:
            for key, base_row in by_key.items():
                if key[0] != baseline:
                    continue
                candidate_row = by_key.get((candidate, key[1], key[2]))
                if candidate_row is None:
                    continue
                comparisons.append(
                    {
                        "baseline": baseline,
                        "candidate": candidate,
                        "task": key[1],
                        "position": key[2],
                        "baseline_accuracy": base_row["accuracy"],
                        "candidate_accuracy": candidate_row["accuracy"],
                        "delta": float(candidate_row["accuracy"]) - float(base_row["accuracy"]),
                    }
                )
        comparison_fields = [
            "baseline",
            "candidate",
            "task",
            "position",
            "baseline_accuracy",
            "candidate_accuracy",
            "delta",
        ]
        with comparison_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=comparison_fields)
            writer.writeheader()
            writer.writerows(comparisons)
        summary["comparisons"] = comparisons
    elif comparison_path.exists():
        comparison_path.unlink()
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
