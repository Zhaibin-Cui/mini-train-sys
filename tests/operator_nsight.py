"""Reusable one-line Nsight Compute profiling for notebook benchmarks."""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


_NVTX_RANGE = "minitrain_nsight"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_REPORT_DIR = _PROJECT_ROOT / "reports/nsight"
_DEFAULT_TIMEOUT_S = 120
_DEFAULT_SECTIONS = ("LaunchStats",)
_DEFAULT_LAUNCH_COUNT = 128
_DEFAULT_NVTX_INCLUDE = _NVTX_RANGE
_DEFAULT_CAPTURE_MODE = "cuda_profiler"
_CAPTURE_MODES = ("cuda_profiler", "nvtx", "none")


@dataclass(frozen=True)
class ProfileSpec:
    make_case: Callable[[int], Any]
    forward: Callable[[str, dict[str, torch.Tensor]], Any]
    default_size: int
    provider: str = "triton"
    autotune_kernels: Callable[[], dict[str, Any]] | None = None


_PROFILE_REGISTRY: dict[str, ProfileSpec] = {}


def register_nsight_kernel(
    name: str,
    make_case: Callable[[int], Any],
    forward: Callable[[str, dict[str, torch.Tensor]], Any],
    *,
    default_size: int,
    provider: str = "triton",
    autotune_kernels: Callable[[], dict[str, Any]] | None = None,
) -> None:
    """Register an existing benchmark definition for one-line profiling."""

    if default_size <= 0:
        raise ValueError("default_size must be positive.")
    _PROFILE_REGISTRY[name] = ProfileSpec(
        make_case, forward, default_size, provider, autotune_kernels
    )


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
    def resolve(candidate: str | Path | None) -> Path | None:
        if not candidate:
            return None
        path = Path(candidate).resolve()
        if not path.is_file():
            return None
        if path.suffix.lower() != ".bat":
            return path
        exe_matches = sorted(path.parent.glob("target/**/ncu.exe"), reverse=True)
        return exe_matches[0].resolve() if exe_matches else path

    candidates = [
        os.environ.get("NCU_PATH"),
        shutil.which("ncu.exe"),
        shutil.which("ncu"),
        shutil.which("ncu.bat"),
    ]
    if os.name == "nt":
        program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        for executable in ("ncu.exe", "ncu.bat"):
            candidates.extend(
                sorted(
                    program_files.glob(f"NVIDIA Corporation/Nsight Compute */{executable}"),
                    reverse=True,
                )
            )
    for candidate in candidates:
        path = resolve(candidate)
        if path is not None:
            return path
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


def _pin_autotune_configs(spec: ProfileSpec, configs: dict[str, dict[str, Any]]) -> None:
    if not configs or spec.autotune_kernels is None:
        return
    import triton

    for name, kernel in spec.autotune_kernels().items():
        if name not in configs:
            continue
        config = configs[name]
        kernel.configs = [
            triton.Config(
                config["kwargs"],
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
                num_ctas=config["num_ctas"],
                maxnreg=config["maxnreg"],
            )
        ]
        kernel.cache.clear()


def _run_case(spec: ProfileSpec, *, size: int, mode: str) -> None:
    case = _prepare_case(spec, size=size, mode=mode)
    try:
        output = spec.forward(spec.provider, case.tensors)
        if mode == "bwd":
            _loss(output).backward()
        torch.cuda.synchronize()
    finally:
        case.tensors.clear()
        torch.cuda.empty_cache()


def _autotune_snapshot(spec: ProfileSpec, *, size: int, mode: str) -> dict[str, dict[str, Any]]:
    if spec.autotune_kernels is None:
        return {}
    # Tune once at native speed before ncu starts. The worker receives only the
    # winners, avoiding profiler overhead across every candidate configuration.
    _run_case(spec, size=size, mode=mode)
    snapshot = {}
    for name, kernel in spec.autotune_kernels().items():
        config = getattr(kernel, "best_config", None)
        if config is None:
            continue
        snapshot[name] = {
            "kwargs": dict(config.kwargs),
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
            "num_ctas": config.num_ctas,
            "maxnreg": config.maxnreg,
        }
    return snapshot


def _capture_start(capture_mode: str) -> None:
    if capture_mode == "nvtx":
        torch.cuda.nvtx.range_push(_NVTX_RANGE)
    elif capture_mode == "cuda_profiler":
        torch.cuda.cudart().cudaProfilerStart()


def _capture_stop(capture_mode: str) -> None:
    if capture_mode == "nvtx":
        torch.cuda.nvtx.range_pop()
    elif capture_mode == "cuda_profiler":
        torch.cuda.cudart().cudaProfilerStop()


@contextlib.contextmanager
def _capture_range(capture_mode: str):
    _capture_start(capture_mode)
    try:
        yield
    finally:
        _capture_stop(capture_mode)


def _run_worker(payload: Path, *, mode: str, capture_mode: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Nsight kernel profiling requires CUDA.")
    with payload.open("rb") as file:
        spec, size, autotune_configs = _cloudpickle().load(file)
    _pin_autotune_configs(spec, autotune_configs)
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
        with _capture_range(capture_mode), torch.no_grad():
            output = call()
        del output
    else:
        loss = _loss(call())
        torch.cuda.synchronize()
        with _capture_range(capture_mode):
            loss.backward()
        del loss
    torch.cuda.synchronize()


def _csv_summary(csv_text: str) -> list[dict[str, str]]:
    lines = csv_text.splitlines()
    header = next(
        (index for index, line in enumerate(lines) if line.lstrip("\ufeff").startswith('"ID"')),
        None,
    )
    if header is None:
        return []
    lines[header] = lines[header].lstrip("\ufeff")
    return list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))


def _kernel_summary(metric_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Collapse Nsight's metric-per-row CSV into one row per kernel launch."""

    identity = ("ID", "Kernel Name", "Context", "Stream", "Block Size", "Grid Size")
    metric_labels = {
        "gpu__time_duration.sum": "Duration",
        "launch__registers_per_thread": "Registers Per Thread",
        "launch__shared_mem_per_block_static": "Static Shared Memory",
        "launch__shared_mem_per_block_dynamic": "Dynamic Shared Memory",
        "launch__occupancy_limit_registers": "Register Occupancy Limit",
        "launch__occupancy_limit_shared_mem": "Shared-memory Occupancy Limit",
        "launch__occupancy_per_block_size": "Block Occupancy",
    }
    launches: dict[tuple[str, ...], dict[str, str]] = {}
    for metric in metric_rows:
        key = tuple(metric.get(column, "") for column in identity)
        launch = launches.setdefault(key, dict(zip(identity, key)))
        metric_name = metric.get("Metric Name", "")
        if not metric_name:
            continue
        label = metric_labels.get(metric_name, metric_name)
        value = metric.get("Metric Value", "")
        unit = metric.get("Metric Unit", "")
        launch[label] = f"{value} {unit}".strip()
    return list(launches.values())


def _filter_kernel_summary(
    summary: list[dict[str, str]], kernel_filter: str | None
) -> list[dict[str, str]]:
    if not kernel_filter:
        return summary
    pattern = re.compile(kernel_filter, re.IGNORECASE)
    return [
        launch
        for launch in summary
        if pattern.search(launch.get("Kernel Name", ""))
    ]


def _profile_stem(report_dir: Path, name: str, mode: str, size: int) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return report_dir / f"{name}_{mode}_{size}_{timestamp}"


def _run(
    command: list[str],
    *,
    timeout_s: int,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
            )
        else:
            process.kill()
        process.communicate()
        executable = Path(command[0]).name
        raise TimeoutError(f"{executable} exceeded the {timeout_s}s profiling timeout.") from None
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        "" if stdout is None else stdout,
        "" if stderr is None else stderr,
    )


def _format_process_failure(title: str, completed: subprocess.CompletedProcess[str]) -> str:
    command = subprocess.list2cmdline([str(argument) for argument in completed.args])
    stdout = completed.stdout or "<empty>"
    stderr = completed.stderr or "<empty>"
    return (
        f"{title}\n"
        f"returncode: {completed.returncode}\n"
        f"command: {command}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )


def _missing_report_message(
    completed: subprocess.CompletedProcess[str],
    report_path: Path,
    report_dir: Path,
) -> str:
    recent = sorted(
        (path for path in report_dir.glob("*") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:10]
    files = "\n".join(f"  {path.name}" for path in recent) or "  <none>"
    return (
        _format_process_failure("Nsight Compute completed without creating a report.", completed)
        + f"\nexpected report: {report_path}\nrecent files in report_dir:\n{files}"
    )


def _display_result(result: dict[str, Any]) -> None:
    metadata = {
        key: value
        for key, value in result.items()
        if key not in ("summary", "summary_all")
    }
    rows = result["summary"]
    try:
        import pandas as pd
        from IPython.display import display

        display(pd.DataFrame([metadata]))
        if rows:
            preferred = (
                "Kernel Name",
                "Context",
                "Stream",
                "Duration",
                "Registers Per Thread",
                "Static Shared Memory",
                "Dynamic Shared Memory",
                "Block Size",
                "Grid Size",
            )
            columns = [column for column in preferred if column in rows[0]]
            summary = pd.DataFrame(rows)
            display(summary[columns] if columns else summary)
    except ImportError:
        print(metadata)
        if rows:
            print(rows)


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_PROJECT_ROOT), existing) if value
    )
    return environment


def _normalize_capture_mode(capture_mode: str, use_nvtx: bool | None) -> str:
    if capture_mode not in _CAPTURE_MODES:
        choices = ", ".join(repr(mode) for mode in _CAPTURE_MODES)
        raise ValueError(f"capture_mode must be one of: {choices}.")
    if use_nvtx is None:
        return capture_mode
    # Backward-compatible shim for earlier notebook examples.
    return "nvtx" if use_nvtx else "none"


def _metric_selection(
    *,
    set_name: str | None,
    sections: tuple[str, ...] | None,
) -> list[str]:
    if set_name is not None and sections is not None:
        raise ValueError("Pass either set_name or sections, not both.")
    if set_name is not None:
        return ["--set", set_name]
    sections = _DEFAULT_SECTIONS if sections is None else sections
    if not sections:
        raise ValueError("sections must contain at least one Nsight section.")
    return [argument for section in sections for argument in ("--section", section)]


def _build_ncu_command(
    *,
    ncu: Path,
    stem: Path,
    payload_path: Path,
    mode: str,
    capture_mode: str,
    nvtx_include: str,
    metric_selection: list[str],
    launch_count: int | None,
    launch_skip_before_match: int,
    capture_kernel_filter: str | None,
) -> list[str]:
    command = [
        str(ncu),
        "--target-processes",
        "all",
        "--force-overwrite",
        "-o",
        str(stem),
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(payload_path),
        "--mode",
        mode,
        "--capture-mode",
        capture_mode,
    ]
    if capture_mode == "cuda_profiler":
        command[1:1] = ["--profile-from-start", "off"]
    elif capture_mode == "nvtx":
        command[1:1] = ["--nvtx", "--nvtx-include", nvtx_include]

    selection = list(metric_selection)
    if launch_count is not None:
        selection.extend(["--launch-count", str(launch_count)])
    if launch_skip_before_match:
        selection.extend(["--launch-skip-before-match", str(launch_skip_before_match)])
    if capture_kernel_filter:
        selection.extend(
            [
                "--kernel-name-base",
                "demangled",
                "--kernel-name",
                f"regex:{capture_kernel_filter}",
            ]
        )
    command[1:1] = selection
    return command


def nsight_kernel(
    name: str,
    *,
    mode: str = "fwd",
    size: int | None = None,
    report_dir: str | Path = _DEFAULT_REPORT_DIR,
    set_name: str | None = None,
    sections: tuple[str, ...] | None = None,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    launch_count: int | None = _DEFAULT_LAUNCH_COUNT,
    launch_skip_before_match: int = 0,
    kernel_filter: str | None = None,
    capture_kernel_filter: str | None = None,
    use_nvtx: bool | None = None,
    nvtx_include: str = _DEFAULT_NVTX_INCLUDE,
    capture_mode: str = _DEFAULT_CAPTURE_MODE,
) -> dict[str, Any]:
    """Profile one registered benchmark case in a clean Nsight subprocess.

    The .ncu-rep and .csv files preserve every kernel captured inside the NVTX
    range. kernel_filter only filters the notebook display and result["summary"];
    result["summary_all"] remains the full launch summary.

    capture_kernel_filter is different: it is passed to Nsight Compute as a
    demangled kernel-name regex, so the raw .ncu-rep contains only matching
    profiled launches. Use it when a backward range starts with PyTorch helper
    kernels and you want a report for the real target kernel.

    Nsight Compute may otherwise stop after the first launch in a backward
    range, which is often a tiny PyTorch autograd helper kernel. launch_count
    raises that collection limit while keeping the raw report complete.

    use_nvtx is a backward-compatible shortcut for old notebook examples:
    True maps to capture_mode="nvtx", False maps to capture_mode="none".
    Prefer capture_mode for new calls.

    nvtx_include is passed directly to Nsight Compute's --nvtx-include. The
    default matches the range name pushed by the worker. Some Nsight versions
    are sensitive to stack-syntax suffixes such as '/', so callers can override
    this without changing the profiling wrapper.

    capture_mode selects how the worker marks the measured region:
    "cuda_profiler" uses cudaProfilerStart/Stop and avoids NVTX include syntax;
    "nvtx" uses torch.cuda.nvtx.range_push/pop plus --nvtx-include; "none"
    leaves ncu to profile from process start.
    """

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
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive.")
    if launch_count is not None and launch_count <= 0:
        raise ValueError("launch_count must be positive or None.")
    if launch_skip_before_match < 0:
        raise ValueError("launch_skip_before_match must be non-negative.")
    capture_mode = _normalize_capture_mode(capture_mode, use_nvtx)
    if capture_mode == "nvtx" and not nvtx_include:
        raise ValueError("nvtx_include must be non-empty when capture_mode='nvtx'.")
    metric_selection = _metric_selection(set_name=set_name, sections=sections)
    if kernel_filter is not None:
        re.compile(kernel_filter)
    if capture_kernel_filter is not None:
        re.compile(capture_kernel_filter)
    ncu = _find_ncu()
    autotune_configs = _autotune_snapshot(spec, size=profile_size, mode=mode)
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = _profile_stem(report_dir, name, mode, profile_size)
    report_path = stem.with_suffix(".ncu-rep")
    csv_path = stem.with_suffix(".csv")
    descriptor, payload_name = tempfile.mkstemp(
        prefix=f".{stem.name}_", suffix=".profile.pkl", dir=report_dir
    )
    os.close(descriptor)
    payload_path = Path(payload_name)
    with payload_path.open("wb") as file:
        _cloudpickle().dump((spec, profile_size, autotune_configs), file)

    command = _build_ncu_command(
        ncu=ncu,
        stem=stem,
        payload_path=payload_path,
        mode=mode,
        capture_mode=capture_mode,
        nvtx_include=nvtx_include,
        metric_selection=metric_selection,
        launch_count=launch_count,
        launch_skip_before_match=launch_skip_before_match,
        capture_kernel_filter=capture_kernel_filter,
    )
    try:
        completed = _run(
            command,
            timeout_s=timeout_s,
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
        raise RuntimeError(_format_process_failure("Nsight Compute failed.", completed))
    if not report_path.is_file():
        raise RuntimeError(_missing_report_message(completed, report_path, report_dir))

    export = _run(
        [str(ncu), "--import", str(report_path), "--csv", "--page", "details"],
        timeout_s=timeout_s,
    )
    if export.returncode != 0:
        raise RuntimeError(_format_process_failure(f"Failed to export {report_path} as CSV.", export))
    csv_path.write_text(export.stdout, encoding="utf-8")
    summary = _kernel_summary(_csv_summary(export.stdout))
    display_summary = _filter_kernel_summary(summary, kernel_filter)
    result = {
        "kernel": name,
        "mode": mode,
        "size": profile_size,
        "report_prefix": str(stem),
        "report": str(report_path),
        "csv": str(csv_path),
        "command": command,
        "kernels_captured": len(summary),
        "kernels_displayed": len(display_summary),
        "launch_count": launch_count,
        "launch_skip_before_match": launch_skip_before_match,
        "kernel_filter": kernel_filter,
        "capture_kernel_filter": capture_kernel_filter,
        "capture_mode": capture_mode,
        "use_nvtx": capture_mode == "nvtx",
        "nvtx_include": nvtx_include if capture_mode == "nvtx" else None,
        "summary_all": summary,
        "summary": display_summary,
    }
    _display_result(result)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--mode", choices=("fwd", "bwd"), default="fwd")
    parser.add_argument(
        "--capture-mode",
        choices=("cuda_profiler", "nvtx", "none"),
        default=_DEFAULT_CAPTURE_MODE,
    )
    args = parser.parse_args()
    _run_worker(args.worker, mode=args.mode, capture_mode=args.capture_mode)


if __name__ == "__main__":
    _main()
