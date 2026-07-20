"""Preflight checks for the Linux/NVIDIA experiment server."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist


REQUIRED_MODULES = (
    "cloudpickle",
    "IPython",
    "jupyterlab",
    "matplotlib",
    "numpy",
    "pandas",
    "pyarrow",
    "pytest",
    "tensorboard",
    "tiktoken",
    "tokenizers",
    "triton",
    "yaml",
)


def _capture(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        return 127, f"{type(exc).__name__}: {exc}"
    output = completed.stdout.strip() or completed.stderr.strip()
    return completed.returncode, output


def _find_checkout(start: str | Path | None = None) -> Path | None:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "configs" / "server" / "rtx4090_24gb"
        ).is_dir():
            return candidate
    package_root = Path(__file__).resolve().parents[2]
    return package_root if (package_root / "pyproject.toml").is_file() else None


def collect_server_status(
    *,
    expected_gpus: int,
    min_free_disk_gb: float,
    require_nvcc: bool,
) -> dict[str, object]:
    """Collect actionable environment checks without initializing a process group."""

    errors: list[str] = []
    warnings: list[str] = []
    modules = {
        name: importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES
    }
    missing_modules = [name for name, available in modules.items() if not available]
    if missing_modules:
        errors.append("missing Python modules: " + ", ".join(missing_modules))

    if platform.system() != "Linux":
        errors.append(f"formal server runs require Linux, found {platform.system()}")
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        errors.append(
            f"Python 3.10-3.12 is required, found {sys.version_info.major}.{sys.version_info.minor}"
        )

    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    if not cuda_available:
        errors.append("torch.cuda.is_available() is false")
    elif gpu_count < expected_gpus:
        errors.append(f"expected at least {expected_gpus} visible GPUs, found {gpu_count}")
    if not dist.is_available() or not dist.is_nccl_available():
        errors.append("PyTorch NCCL distributed backend is unavailable")

    gpu_rows = []
    for index in range(gpu_count):
        properties = torch.cuda.get_device_properties(index)
        gpu_rows.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "memory_gb": round(properties.total_memory / 1024**3, 2),
            }
        )
    bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
    if cuda_available and not bf16_supported:
        errors.append("the configured BF16 experiments require torch.cuda.is_bf16_supported()")

    smi_code, smi = _capture(["nvidia-smi", "-L"])
    topo_code, topology = _capture(["nvidia-smi", "topo", "-m"])
    if smi_code != 0:
        errors.append("nvidia-smi failed: " + smi)
    if topo_code != 0:
        warnings.append("nvidia-smi topology query failed: " + topology)

    nvcc_path = shutil.which("nvcc")
    nvcc_code, nvcc_version = (
        _capture([nvcc_path, "--version"]) if nvcc_path else (127, "nvcc not found")
    )
    if require_nvcc and nvcc_code != 0:
        errors.append(
            "nvcc is required for the optional CUDA C++ extension; install a matching CUDA "
            "toolkit or omit --require-nvcc when using the Triton backend"
        )
    elif nvcc_code != 0:
        warnings.append("nvcc is absent; Triton works, but cuda_ext cannot be JIT-compiled")

    checkout = _find_checkout()
    if checkout is None:
        errors.append("run the check from a mini-train-sys source checkout")
        disk_root = Path.cwd()
    else:
        disk_root = checkout
        required_paths = (
            checkout / "scripts" / "train.py",
            checkout / "scripts" / "bash" / "synbios_moe.sh",
            checkout / "scripts" / "bash" / "synbios_probes.sh",
            checkout / "configs" / "synbios_moe" / "model.yaml",
            checkout / "tests" / "distributed_server_benchmark.ipynb",
        )
        missing_paths = [str(path) for path in required_paths if not path.is_file()]
        if missing_paths:
            errors.append("checkout is incomplete: " + ", ".join(missing_paths))
        if not os.access(checkout, os.W_OK):
            errors.append(f"checkout is not writable: {checkout}")

    disk = shutil.disk_usage(disk_root)
    free_disk_gb = disk.free / 1024**3
    if free_disk_gb < min_free_disk_gb:
        warnings.append(
            f"only {free_disk_gb:.1f} GiB is free; requested warning threshold is "
            f"{min_free_disk_gb:.1f} GiB"
        )

    shm_gb = None
    shm = Path("/dev/shm")
    if shm.is_dir():
        shm_gb = shutil.disk_usage(shm).total / 1024**3
        if shm_gb < 8:
            warnings.append(
                f"/dev/shm is only {shm_gb:.1f} GiB; DataLoader workers may need more"
            )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checkout": str(checkout) if checkout is not None else None,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "nccl_available": bool(dist.is_available() and dist.is_nccl_available()),
        "bf16_supported": bf16_supported,
        "visible_gpus": gpu_count,
        "gpus": gpu_rows,
        "nvidia_smi": smi,
        "topology": topology,
        "nvcc": nvcc_version,
        "free_disk_gb": round(free_disk_gb, 2),
        "dev_shm_gb": round(shm_gb, 2) if shm_gb is not None else None,
        "modules": modules,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--expected-gpus", type=int, default=8)
    root.add_argument("--min-free-disk-gb", type=float, default=100.0)
    root.add_argument("--require-nvcc", action="store_true")
    root.add_argument("--output", help="optional JSON output path")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.expected_gpus <= 0 or args.min_free_disk_gb < 0:
        raise SystemExit("expected-gpus must be positive and min-free-disk-gb non-negative")
    status = collect_server_status(
        expected_gpus=args.expected_gpus,
        min_free_disk_gb=args.min_free_disk_gb,
        require_nvcc=args.require_nvcc,
    )
    rendered = json.dumps(status, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if not status["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
