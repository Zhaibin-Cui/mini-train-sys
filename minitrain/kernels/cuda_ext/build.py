"""Build and load MiniTrain's CUDA FlashAttention extension.

The expensive kernel implementation is inherited from FlashAttention 2.8.4.
This module controls only the explicit-instantiation matrix, CUDA architectures,
compiler flags, build parallelism, and PyTorch JIT cache identity.

Three profiles make the same source tree practical on different machines:

``minimal``
    fp16, head-dim bucket 32. Intended for CI and compiler smoke tests.
``workstation`` (default)
    fp16/bf16, buckets 32/64/128. Covers common small-model configurations.
``full``
    fp16/bf16, all upstream buckets through 256. Intended for build servers.

Environment variables ``MINITRAIN_CUDA_HEAD_DIMS`` and
``MINITRAIN_CUDA_DTYPES`` override the profile matrix explicitly.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch


# -----------------------------------------------------------------------------
# User-editable build defaults
# -----------------------------------------------------------------------------
# Change these values when this repository normally runs on a different class
# of machine. Environment variables still take precedence, which lets a build
# server override the local defaults without editing this file.
_DEFAULT_BUILD_PROFILE = "full"
_DEFAULT_CUDA_ARCHS = "86"  # Semicolon-separated values, for example "80;90".
_DEFAULT_MAX_JOBS = 2
_DEFAULT_VERBOSE = True  # Print nvcc command lines and verbose ninja output.

# Each profile selects the dtype/head-dimension kernel matrix. Modify these
# tuples when the project's normal model shapes change. A larger matrix takes
# more compilation time, host memory, and extension disk space.
_HEAD_DIM_BUCKETS = (32, 64, 96, 128, 192, 256)
_DTYPES = ("fp16", "bf16")
_PROFILES = {
    "minimal": ((32,), ("fp16",)),
    "workstation": ((32, 64, 128), _DTYPES),
    "full": (_HEAD_DIM_BUCKETS, _DTYPES),
}


@dataclass(frozen=True)
class CudaBuildConfig:
    """Immutable build matrix used for source selection and cache identity."""

    profile: str
    archs: tuple[str, ...]
    head_dims: tuple[int, ...]
    dtypes: tuple[str, ...]

    @property
    def cache_key(self) -> str:
        """Return a short stable suffix accepted as a Python module name."""

        raw = f"{self.profile}|{self.archs}|{self.head_dims}|{self.dtypes}|fa2_8_4"
        return hashlib.sha256(raw.encode("ascii")).hexdigest()[:12]


def _parse_list(raw: str) -> tuple[str, ...]:
    """Parse semicolon-separated build values while preserving order."""

    return tuple(value.strip() for value in raw.split(";") if value.strip())


@lru_cache(maxsize=1)
def get_build_config() -> CudaBuildConfig:
    """Resolve and validate the active kernel matrix from the environment."""

    profile = os.getenv("MINITRAIN_CUDA_BUILD_PROFILE", _DEFAULT_BUILD_PROFILE).lower()
    if profile not in _PROFILES:
        raise ValueError(
            f"Unknown MINITRAIN_CUDA_BUILD_PROFILE={profile!r}; "
            f"expected one of {sorted(_PROFILES)}."
        )
    profile_head_dims, profile_dtypes = _PROFILES[profile]

    archs = _parse_list(os.getenv("MINITRAIN_CUDA_ARCHS", _DEFAULT_CUDA_ARCHS))
    if not archs or any(not arch.isdigit() or int(arch) < 80 for arch in archs):
        raise ValueError("MINITRAIN_CUDA_ARCHS must contain sm80+ values such as '80;86;90'.")

    raw_head_dims = os.getenv("MINITRAIN_CUDA_HEAD_DIMS")
    head_dims = (
        tuple(int(value) for value in _parse_list(raw_head_dims))
        if raw_head_dims
        else tuple(profile_head_dims)
    )
    if not head_dims or any(value not in _HEAD_DIM_BUCKETS for value in head_dims):
        raise ValueError(f"Head-dim buckets must be selected from {_HEAD_DIM_BUCKETS}.")

    raw_dtypes = os.getenv("MINITRAIN_CUDA_DTYPES")
    dtypes = _parse_list(raw_dtypes) if raw_dtypes else tuple(profile_dtypes)
    if not dtypes or any(value not in _DTYPES for value in dtypes):
        raise ValueError(f"CUDA dtypes must be selected from {_DTYPES}.")

    # Sorting makes equivalent environment strings share one build cache.
    return CudaBuildConfig(
        profile=profile,
        archs=tuple(sorted(set(archs), key=int)),
        head_dims=tuple(sorted(set(head_dims))),
        dtypes=tuple(dtype for dtype in _DTYPES if dtype in dtypes),
    )


def _package_dir() -> Path:
    """Return the Python package directory containing this build module."""

    return Path(__file__).resolve().parent


def _source_dir() -> Path:
    """Return the root of MiniTrain-owned C++/CUDA sources."""

    return _package_dir() / "csrc"


def _third_party_dir() -> Path:
    """Return the root of vendored upstream FlashAttention and CUTLASS files."""

    return _source_dir() / "third_party"


def _extension_name(config: CudaBuildConfig) -> str:
    """Use a configuration-specific name so incompatible DLLs never collide."""

    return f"minitrain_cuda_flash_{config.cache_key}"


def _build_dir(config: CudaBuildConfig) -> Path:
    """Return a repo-local ninja cache isolated by build configuration."""

    path = _package_dir() / "build" / _extension_name(config)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _instantiation_sources(config: CudaBuildConfig) -> list[Path]:
    """Select the thin forward/backward `.cu` files for this build matrix."""

    directory = _source_dir() / "instantiations"
    sources: list[Path] = []
    for direction in ("fwd", "bwd"):
        for dtype in config.dtypes:
            for head_dim in config.head_dims:
                for is_causal in (False, True):
                    causal = "_causal" if is_causal else ""
                    path = directory / f"flash_{direction}_hdim{head_dim}_{dtype}{causal}_sm80.cu"
                    if not path.exists():
                        raise RuntimeError(
                            f"Missing generated kernel {path}. Run "
                            "`python minitrain/kernels/cuda_ext/generate_kernels.py`."
                        )
                    sources.append(path)
    return sources


def _preprocessor_definitions(config: CudaBuildConfig, *, for_nvcc: bool) -> list[str]:
    """Keep API dispatch and linked template symbols on the same matrix.

    MSVC spells definitions as ``/DNAME`` when compiling the C++ bridge. nvcc
    still parses its own command line with ``-DNAME`` on Windows before passing
    host options through to cl.exe.
    """

    definitions = [
        "FLASH_NAMESPACE=minitrain_flash",
        "FLASHATTENTION_DISABLE_ALIBI",
        "FLASHATTENTION_DISABLE_LOCAL",
        "FLASHATTENTION_DISABLE_SOFTCAP",
    ]
    definitions.extend(f"MINITRAIN_FLASH_HDIM_{head_dim}" for head_dim in config.head_dims)
    definitions.extend(f"MINITRAIN_FLASH_ENABLE_{dtype.upper()}" for dtype in config.dtypes)
    prefix = "-D" if for_nvcc or os.name != "nt" else "/D"
    return [f"{prefix}{definition}" for definition in definitions]


def _extra_cuda_cflags(config: CudaBuildConfig) -> list[str]:
    """Return nvcc flags shared by every generated translation unit."""

    windows_compat = (
        ["-allow-unsupported-compiler", "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"]
        if os.name == "nt"
        else []
    )
    return [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--ptxas-options=-v",
        *_preprocessor_definitions(config, for_nvcc=True),
        *windows_compat,
    ]


def _extra_cflags(config: CudaBuildConfig) -> list[str]:
    """Return host flags in MSVC or GCC spelling as appropriate."""

    compiler_flags = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]
    return [*compiler_flags, *_preprocessor_definitions(config, for_nvcc=False)]


def _torch_arch_list(config: CudaBuildConfig) -> str:
    """Translate compact SM names to PyTorch's canonical architecture syntax."""

    values = [f"{arch[0]}.{arch[1:]}" for arch in config.archs]
    # Retain PTX for the newest selected architecture to allow forward JIT on a
    # compatible newer GPU when a server-specific cubin was not precompiled.
    values[-1] += "+PTX"
    return ";".join(values)


def compiled_head_dims() -> tuple[int, ...]:
    """Expose selected buckets to Python's no-load support predicate."""

    return get_build_config().head_dims


def compiled_dtypes() -> tuple[str, ...]:
    """Expose selected dtypes to Python's no-load support predicate."""

    return get_build_config().dtypes


@lru_cache(maxsize=1)
def load_cuda_extension():
    """Compile and import the selected extension exactly once per process."""

    from torch.utils.cpp_extension import CUDA_HOME, load

    if not torch.cuda.is_available():
        raise RuntimeError("The mini-train-sys CUDA backend requires a CUDA device.")
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME/nvcc was not found; install a CUDA toolkit to build cuda_ext.")

    config = get_build_config()
    os.environ.setdefault("MAX_JOBS", os.getenv("MINITRAIN_CUDA_MAX_JOBS", str(_DEFAULT_MAX_JOBS)))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _torch_arch_list(config))

    src = _source_dir()
    third_party = _third_party_dir()
    sources = [src / "flash_api_upstream.cpp", *_instantiation_sources(config)]
    return load(
        name=_extension_name(config),
        sources=[str(path) for path in sources],
        build_directory=str(_build_dir(config)),
        extra_cflags=_extra_cflags(config),
        extra_cuda_cflags=_extra_cuda_cflags(config),
        extra_include_paths=[
            str(src),
            str(third_party / "flash_attn" / "src"),
            str(third_party / "cutlass" / "include"),
        ],
        with_cuda=True,
        verbose=os.getenv("MINITRAIN_CUDA_VERBOSE", "1" if _DEFAULT_VERBOSE else "0") == "1",
    )
