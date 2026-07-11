"""Reusable one-line Nsight Compute profiling for notebook benchmarks."""

from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


_NVTX_RANGE = "minitrain_nsight"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_REPORT_DIR = _PROJECT_ROOT / "reports/nsight"


@dataclass(frozen=True)
class ProfileSpec:
    make_case: Callable[[int], Any]
    forward: Callable[[str, dict[str, torch.Tensor]], Any]
    default_size: int
    provider: str = "triton"


_PROFILE_REGISTRY: dict[str, ProfileSpec] = {}


def register_nsight_kernel(
    name: str,
    make_case: Callable[[int], Any],
    forward: Callable[[str, dict[str, torch.Tensor]], Any],
    *,
    default_size: int,
    provider: str = "triton",
) -> None:
    """Register an existing benchmark definition for one-line profiling."""

    if default_size <= 0:
        raise ValueError("default_size must be positive.")
    _PROFILE_REGISTRY[name] = ProfileSpec(make_case, forward, default_size, provider)


def registered_nsight_kernels() -> tuple[str, ...]:
    return tuple(_PROFILE_REGISTRY)


def _cloudpickle():
    try:
        import cloudpickle
    except ImportError as exc:
        raise RuntimeError(
            "Nsight notebook profiling requires cloudpickle. "
            "Install the development dependencies with `pip install -e .[dev]`."
        ) from exc
    return cloudpickle


def _find_ncu() -> Path:
    candidates = [
        os.environ.get("NCU_PATH"),
        shutil.which("ncu"),
        shutil.which("ncu.bat"),
        shutil.which("ncu.exe"),
    ]
    program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
    for executable in ("ncu.bat", "ncu.exe"):
        candidates.extend(
            sorted(
                program_files.glob(f"NVIDIA Corporation/Nsight Compute */{executable}"),
                reverse=True,
            )
        )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise FileNotFoundError("Nsight Compute CLI was not found. Add ncu to PATH or set NCU_PATH.")


def _flatten_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _flatten_tensors(item)
    else:
        raise TypeError(f"Unsupported output type: {type(value)!r}")


def _loss(output: Any) -> torch.Tensor:
    return sum(tensor.float().sum() for tensor in _flatten_tensors(output))


def _prepare_case(spec: ProfileSpec, *, size: int, mode: str):
    case = spec.make_case(size)
    if mode == "bwd":
        for name in case.grad_names:
            case.tensors[name].requires_grad_(True)
    return case


def _zero_grads(case: Any) -> None:
    for name in case.grad_names:
        case.tensors[name].grad = None


def _run_worker(payload: Path, *, mode: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Nsight kernel profiling requires CUDA.")
    with payload.open("rb") as file:
        spec, size = _cloudpickle().load(file)
    case = _prepare_case(spec, size=size, mode=mode)
    call = lambda: spec.forward(spec.provider, case.tensors)

    # Compile and autotune outside the captured range.
    if mode == "fwd":
        with torch.no_grad():
            call()
    else:
        _loss(call()).backward()
        _zero_grads(case)
    torch.cuda.synchronize()

    if mode == "fwd":
        torch.cuda.nvtx.range_push(_NVTX_RANGE)
        with torch.no_grad():
            output = call()
        torch.cuda.nvtx.range_pop()
        del output
    else:
        loss = _loss(call())
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(_NVTX_RANGE)
        loss.backward()
        torch.cuda.nvtx.range_pop()
        del loss
    torch.cuda.synchronize()


def _csv_summary(csv_text: str) -> list[dict[str, str]]:
    lines = csv_text.splitlines()
    header = next((index for index, line in enumerate(lines) if line.startswith('"ID"')), None)
    if header is None:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_PROJECT_ROOT), existing) if value
    )
    return environment


def nsight_kernel(
    name: str,
    *,
    mode: str = "fwd",
    size: int | None = None,
    report_dir: str | Path = _DEFAULT_REPORT_DIR,
    set_name: str = "basic",
) -> dict[str, Any]:
    """Profile one registered benchmark case in a clean Nsight subprocess."""

    if mode not in ("fwd", "bwd"):
        raise ValueError("mode must be 'fwd' or 'bwd'.")
    try:
        spec = _PROFILE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(registered_nsight_kernels()) or "none"
        raise KeyError(f"Unknown Nsight kernel {name!r}; registered kernels: {available}.") from exc

    profile_size = spec.default_size if size is None else size
    if profile_size <= 0:
        raise ValueError("size must be positive.")
    ncu = _find_ncu()
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = report_dir / f"{name}_{mode}_{profile_size}"
    report_path = stem.with_suffix(".ncu-rep")
    csv_path = stem.with_suffix(".csv")
    payload_path = stem.with_suffix(".profile.pkl")
    with payload_path.open("wb") as file:
        _cloudpickle().dump((spec, profile_size), file)

    command = [
        str(ncu),
        "--set",
        set_name,
        "--target-processes",
        "all",
        "--nvtx",
        "--nvtx-include",
        f"{_NVTX_RANGE}/",
        "--force-overwrite",
        "-o",
        str(stem),
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(payload_path),
        "--mode",
        mode,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            env=_subprocess_environment(),
        )
    finally:
        payload_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        output = f"{completed.stdout}\n{completed.stderr}"
        if "ERR_NVGPUCTRPERM" in output:
            raise PermissionError(
                "Nsight Compute cannot access NVIDIA GPU performance counters. Enable "
                "'Developer Settings > Manage GPU Performance Counters > Allow access to "
                "the GPU performance counters to all users' in the NVIDIA Control Panel."
            )
        raise RuntimeError(f"Nsight Compute failed:\n{output}")
    if not report_path.is_file():
        raise RuntimeError("Nsight Compute completed without creating a report.")

    export = subprocess.run(
        [str(ncu), "--import", str(report_path), "--csv", "--page", "details"],
        check=False,
        capture_output=True,
        text=True,
    )
    if export.returncode != 0:
        raise RuntimeError(f"Failed to export {report_path} as CSV:\n{export.stderr}")
    csv_path.write_text(export.stdout, encoding="utf-8")
    summary = _csv_summary(export.stdout)
    result = {
        "kernel": name,
        "mode": mode,
        "size": profile_size,
        "report": str(report_path),
        "csv": str(csv_path),
        "kernels_captured": len(summary),
        "summary": summary,
    }
    try:
        import pandas as pd
        from IPython.display import display

        display(pd.DataFrame([{key: value for key, value in result.items() if key != "summary"}]))
    except ImportError:
        print({key: value for key, value in result.items() if key != "summary"})
    return result


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--mode", choices=("fwd", "bwd"), default="fwd")
    args = parser.parse_args()
    _run_worker(args.worker, mode=args.mode)


if __name__ == "__main__":
    _main()
