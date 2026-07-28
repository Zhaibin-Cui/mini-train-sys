"""Probe task and command value objects shared by the pipeline."""

from dataclasses import dataclass
from pathlib import Path


ProbeStepSchedule = int | dict[str, int]


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


def steps_for_job(schedule: ProbeStepSchedule, job: ProbeJob | str) -> int:
    """Resolve a uniform, per-kind, or per-target update schedule."""

    if not isinstance(schedule, dict):
        return schedule
    kind = job.kind if isinstance(job, ProbeJob) else str(job)
    if kind in schedule:
        return schedule[kind]
    if not isinstance(job, ProbeJob):
        raise ValueError("target-specific step schedule requires a ProbeJob")
    return schedule[f"{job.kind}_{job.target}"]
