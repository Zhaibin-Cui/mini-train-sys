from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import torch


def collect_provenance(repository: str | Path | None = None) -> dict[str, object]:
    """Collect compact run metadata without failing outside a Git checkout."""

    root = Path(repository or Path.cwd())

    def git(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "git_revision": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_dirty": bool(status) if status is not None else None,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }
