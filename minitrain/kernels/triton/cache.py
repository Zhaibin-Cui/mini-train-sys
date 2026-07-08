import os

from pathlib import Path


def project_root() -> Path:
    """Return the repository root for cache paths owned by this project."""

    return Path(__file__).resolve().parents[3]


def default_triton_cache_dir() -> Path:
    """Return the per-project Triton cache directory.

    Triton keys compiled kernels by device, source, constexpr meta-parameters,
    and a few compiler/runtime details. Keeping that cache under the repository
    makes the generated JIT artifacts easy to inspect and easy to clean without
    searching the user's global cache directory.

    `MINITRAIN_TRITON_CACHE_DIR` is intentionally project-specific: users can
    redirect only mini-train-sys without changing unrelated Triton projects.
    """

    override = os.environ.get("MINITRAIN_TRITON_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / ".triton_cache"


def configure_triton_cache() -> Path:
    """Configure Triton to use the project's cache directory.

    Set this before importing modules that define `@triton.jit` kernels. Triton
    also keeps an in-process cache, so the first call for a new signature may
    compile, while repeated calls with the same constexpr configuration usually
    go straight to launching the cached kernel.
    """

    cache_dir = default_triton_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_dir))
    return cache_dir
