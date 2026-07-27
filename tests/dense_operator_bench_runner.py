"""Crash-isolated industrial benchmark suite for the six dense kernels.

The formal profile reproduces the per-rank shapes of the 293M SynBioS MoE.
Each shape runs in a fresh CUDA process so an OOM is retained as evidence and
cannot poison later cases in the notebook.
"""


import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from minitrain.kernels.triton.flash_attention import (  # noqa: E402
    flash_attention_autotune_kernels,
)
from minitrain.model.ops import get_ops_backend  # noqa: E402
from operator_bench_utils import (  # noqa: E402
    BenchCase,
    bench_sweep,
    benchmark_step,
    plot_kernel_grid,
)


@dataclass(frozen=True)
class DenseProfile:
    name: str
    hidden: int
    intermediate: int
    heads: int
    head_dim: int
    vocab: int
    sequence_length: int | None
    token_sizes: tuple[int, ...]


PROFILES = {
    "project_formal": DenseProfile(
        "project_formal",
        hidden=768,
        intermediate=1024,
        heads=12,
        head_dim=64,
        vocab=50257,
        sequence_length=512,
        token_sizes=(512, 2048, 8192, 32768, 57344),
    ),
    "mixtral_dense": DenseProfile(
        "mixtral_dense",
        hidden=4096,
        intermediate=14336,
        heads=32,
        head_dim=128,
        vocab=32000,
        sequence_length=None,
        token_sizes=(16, 64, 256, 512, 1024),
    ),
}
KERNELS = (
    "rmsnorm",
    "rope",
    "swiglu",
    "cross_entropy",
    "fused_linear_cross_entropy",
    "attention",
)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _rope_cache(sequence: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, head_dim, 2, device="cuda").float() / head_dim)
    )
    frequencies = torch.outer(torch.arange(sequence, device="cuda").float(), inv_freq)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos().to(_dtype()), embedding.sin().to(_dtype())


def _attention_shape(profile: DenseProfile, tokens: int) -> tuple[int, int, int, int]:
    if profile.sequence_length is None:
        return 1, profile.heads, tokens, profile.head_dim
    batch, remainder = divmod(tokens, profile.sequence_length)
    if remainder or batch < 1:
        raise ValueError(
            f"tokens={tokens} must be a positive multiple of "
            f"sequence_length={profile.sequence_length}"
        )
    return batch, profile.heads, profile.sequence_length, profile.head_dim


def _make_case(kernel: str, profile: DenseProfile, tokens: int) -> BenchCase:
    dtype = _dtype()
    if kernel == "rmsnorm":
        return BenchCase(
            tensors={
                "x": torch.randn(tokens, profile.hidden, device="cuda", dtype=dtype),
                "weight": torch.ones(profile.hidden, device="cuda", dtype=dtype),
            },
            grad_names=("x", "weight"),
        )
    if kernel == "rope":
        batch, heads, sequence, head_dim = _attention_shape(profile, tokens)
        cos, sin = _rope_cache(sequence, head_dim)
        return BenchCase(
            tensors={
                "q": torch.randn(
                    batch, heads, sequence, head_dim, device="cuda", dtype=dtype
                ),
                "k": torch.randn(
                    batch, heads, sequence, head_dim, device="cuda", dtype=dtype
                ),
                "cos": cos,
                "sin": sin,
            },
            grad_names=("q", "k"),
        )
    if kernel == "swiglu":
        return BenchCase(
            tensors={
                "gate": torch.randn(
                    tokens, profile.intermediate, device="cuda", dtype=dtype
                ),
                "up": torch.randn(
                    tokens, profile.intermediate, device="cuda", dtype=dtype
                ),
            },
            grad_names=("gate", "up"),
        )
    if kernel == "cross_entropy":
        return BenchCase(
            tensors={
                "logits": torch.randn(
                    tokens, profile.vocab, device="cuda", dtype=dtype
                ),
                "targets": torch.randint(profile.vocab, (tokens,), device="cuda"),
            },
            grad_names=("logits",),
        )
    if kernel == "fused_linear_cross_entropy":
        return BenchCase(
            tensors={
                "x": torch.randn(tokens, profile.hidden, device="cuda", dtype=dtype),
                "weight": torch.randn(
                    profile.vocab, profile.hidden, device="cuda", dtype=dtype
                ),
                "targets": torch.randint(profile.vocab, (tokens,), device="cuda"),
            },
            grad_names=("x", "weight"),
        )
    if kernel == "attention":
        shape = _attention_shape(profile, tokens)
        return BenchCase(
            tensors={
                name: torch.randn(shape, device="cuda", dtype=dtype)
                for name in ("q", "k", "v")
            },
            grad_names=("q", "k", "v"),
        )
    raise ValueError(f"unknown dense kernel: {kernel}")


def _forward(kernel: str, provider: str, tensors: dict[str, torch.Tensor]):
    backend = get_ops_backend(provider)
    if kernel == "rmsnorm":
        return backend.rmsnorm(tensors["x"], tensors["weight"], 1e-5)
    if kernel == "rope":
        return backend.rope(
            tensors["q"], tensors["k"], tensors["cos"], tensors["sin"]
        )
    if kernel == "swiglu":
        return backend.swiglu(tensors["gate"], tensors["up"])
    if kernel == "cross_entropy":
        with torch.autocast("cuda", dtype=_dtype()):
            return backend.cross_entropy(tensors["logits"], tensors["targets"])
    if kernel == "fused_linear_cross_entropy":
        with torch.autocast("cuda", dtype=_dtype()):
            return backend.fused_linear_cross_entropy(
                tensors["x"], tensors["weight"], tensors["targets"]
            )
    if kernel == "attention":
        return backend.attention(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            is_causal=True,
            dropout_p=0.0,
        )
    raise ValueError(f"unknown dense kernel: {kernel}")


def worker(
    *,
    kernel: str,
    profile_name: str,
    tokens: int,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("the dense operator benchmark requires CUDA")
    torch.manual_seed(17)
    profile = PROFILES[profile_name]
    providers = ("torch", "triton")
    autotune_kernels = None
    if kernel == "attention":
        providers = (
            ("torch", "triton", "cuda")
            if profile.name == "project_formal"
            else ("torch", "triton")
        )
        autotune_kernels = flash_attention_autotune_kernels()
    rows = bench_sweep(
        kernel=kernel,
        providers=providers,
        sizes=(tokens,),
        size_label=(
            f"tokens; H={profile.hidden}, I={profile.intermediate}, "
            f"heads={profile.heads}, D={profile.head_dim}, vocab={profile.vocab}"
        ),
        make_case=lambda size: _make_case(kernel, profile, size),
        forward=lambda provider, tensors: _forward(kernel, provider, tensors),
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
        atol=5e-2 if kernel in ("attention", "fused_linear_cross_entropy") else 2e-2,
        rtol=8e-2 if kernel in ("attention", "fused_linear_cross_entropy") else 2e-2,
        autotune_kernels=autotune_kernels,
    )
    shape = {
        "tokens": tokens,
        "hidden": profile.hidden,
        "intermediate": profile.intermediate,
        "heads": profile.heads,
        "head_dim": profile.head_dim,
        "vocab": profile.vocab,
    }
    if kernel in ("rope", "attention"):
        batch, heads, sequence, head_dim = _attention_shape(profile, tokens)
        shape.update(
            {
                "batch": batch,
                "heads": heads,
                "sequence_length": sequence,
                "head_dim": head_dim,
            }
        )
    for row in rows:
        row["profile"] = profile_name
        row["shape"] = shape
        row["dtype"] = str(_dtype())
    return {
        "kernel": kernel,
        "profile": asdict(profile),
        "tokens": tokens,
        "dtype": str(_dtype()),
        "gpu": torch.cuda.get_device_name(),
        "total_memory_mb": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "rows": rows,
    }


def capacity_worker(
    *,
    profile_name: str,
    tokens: int,
    warmup_ms: int,
    rep_ms: int,
    validation_tokens: int,
) -> dict[str, object]:
    """Measure fused loss where a same-shape Torch correctness copy cannot fit.

    Correctness must already have passed at ``validation_tokens``. This worker
    deliberately creates no Torch reference and therefore measures the
    optimized kernel's real capacity instead of the benchmark harness's
    two-backend comparison capacity.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("the dense operator capacity benchmark requires CUDA")
    profile = PROFILES[profile_name]
    kernel = "fused_linear_cross_entropy"
    provider = "triton"

    def make_case(size: int) -> BenchCase:
        return _make_case(kernel, profile, size)

    def forward(selected: str, tensors: dict[str, torch.Tensor]):
        return _forward(kernel, selected, tensors)

    row: dict[str, object] = {
        "kernel": kernel,
        "provider": provider,
        "profile": profile_name,
        "size": tokens,
        "size_label": (
            f"tokens; H={profile.hidden}, vocab={profile.vocab}; "
            "performance-only capacity"
        ),
        "status": "ok",
        "measurement_scope": "performance_only_after_smaller_shape_correctness",
        "validation_tokens": validation_tokens,
        "fwd_correct": None,
        "bwd_correct": None,
        "correctness_inherited": True,
        "dtype": str(_dtype()),
    }
    for mode in ("fwd", "full", "bwd"):
        row.update(
            benchmark_step(
                make_case,
                tokens,
                provider,
                forward,
                mode=mode,
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
            )
        )
    row["shape"] = {
        "tokens": tokens,
        "hidden": profile.hidden,
        "vocab": profile.vocab,
    }
    return {
        "kernel": kernel,
        "profile": asdict(profile),
        "tokens": tokens,
        "validation_tokens": validation_tokens,
        "dtype": str(_dtype()),
        "gpu": torch.cuda.get_device_name(),
        "total_memory_mb": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "rows": [row],
    }


def _run_process(command: list[str], timeout_seconds: int) -> tuple[int, str, str, bool]:
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return int(process.returncode), stdout, stderr, False
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            stdout, stderr = process.communicate()
        return -1, stdout, stderr, True


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "shape"}
        flat.update({f"shape_{key}": value for key, value in row.get("shape", {}).items()})
        flat_rows.append(_json_safe(flat))
    columns = sorted({key for row in flat_rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat_rows)
    temporary.replace(path)


class DenseOperatorBenchmark:
    """Notebook-friendly orchestrator for isolated dense-kernel scans."""

    def __init__(
        self,
        output_dir: str | Path = "artifacts/operator_benchmark/rtx4090_24gb/dense",
        *,
        warmup_ms: int = 50,
        rep_ms: int = 200,
        case_timeout_seconds: int = 1200,
    ) -> None:
        self.output_dir = (
            (ROOT / output_dir).resolve()
            if not Path(output_dir).is_absolute()
            else Path(output_dir)
        )
        self.warmup_ms = warmup_ms
        self.rep_ms = rep_ms
        self.case_timeout_seconds = case_timeout_seconds
        self.rows: list[dict[str, object]] = []
        self.failures: list[dict[str, object]] = []

    def preflight(self) -> dict[str, object]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the selected notebook kernel")
        properties = torch.cuda.get_device_properties(0)
        if properties.total_memory < 20 * 2**30:
            raise RuntimeError("the industrial scan requires a GPU with at least 20 GiB")
        return {
            "python": sys.executable,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": properties.name,
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability())),
            "memory_gib": properties.total_memory / 2**30,
            "profiles": {name: asdict(profile) for name, profile in PROFILES.items()},
            "kernels": KERNELS,
        }

    def _case(self, kernel: str, profile: str, tokens: int) -> None:
        case_name = f"{kernel}_{profile}_{tokens}"
        raw_dir = self.output_dir / "raw"
        log_dir = self.output_dir / "logs"
        raw_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        result_path = raw_dir / f"{case_name}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_worker",
            "--kernel",
            kernel,
            "--profile",
            profile,
            "--tokens",
            str(tokens),
            "--warmup-ms",
            str(self.warmup_ms),
            "--rep-ms",
            str(self.rep_ms),
            "--output",
            str(result_path),
        ]
        code, stdout, stderr, timed_out = _run_process(command, self.case_timeout_seconds)
        log_path = log_dir / f"{case_name}.log"
        log_path.write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8")
        if code == 0 and result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.rows.extend(payload["rows"])
            return
        combined = stdout + stderr
        self.failures.append(
            {
                "kernel": kernel,
                "profile": profile,
                "tokens": tokens,
                "returncode": code,
                "timed_out": timed_out,
                "oom": "out of memory" in combined.lower(),
                "log": str(log_path),
                "error_tail": combined[-2000:],
            }
        )

    def run(self) -> dict[str, object]:
        inventory = self.preflight()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for profile_name, profile in PROFILES.items():
            for kernel in KERNELS:
                for tokens in profile.token_sizes:
                    self._case(kernel, profile_name, tokens)
        _write_csv(self.output_dir / "summary.csv", self.rows)

        figure_dir = self.output_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        figures = []
        for profile in PROFILES:
            for kernel in KERNELS:
                selected = [
                    row
                    for row in self.rows
                    if row["kernel"] == kernel and row["profile"] == profile
                ]
                if not selected:
                    continue
                destination = figure_dir / f"{kernel}_{profile}.png"
                figure = plot_kernel_grid(selected, save_path=destination)
                import matplotlib.pyplot as plt

                plt.close(figure)
                figures.append(str(destination.relative_to(self.output_dir)))
        correctness_failures = [
            row
            for row in self.rows
            if row.get("status") == "ok"
            and (
                not row.get("fwd_correct", False)
                or not row.get("bwd_correct", False)
            )
        ]
        summary = {
            "schema_version": 1,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(self.output_dir),
            "inventory": inventory,
            "settings": {
                "warmup_ms": self.warmup_ms,
                "rep_ms": self.rep_ms,
                "case_timeout_seconds": self.case_timeout_seconds,
                "isolation": "one CUDA process per shape",
            },
            "rows": self.rows,
            "case_failures": self.failures,
            "correctness_failures": correctness_failures,
            "unavailable_rows": [
                row for row in self.rows if row.get("status") != "ok"
            ],
            "figures": figures,
            "quality_gate_passed": not self.failures and not correctness_failures,
        }
        _atomic_json(self.output_dir / "summary.json", summary)
        return summary

    def display(self, summary: dict[str, object]) -> None:
        try:
            import pandas as pd
            from IPython.display import Image, display

            display(pd.DataFrame(summary["rows"]))
            for path in summary["figures"]:
                display(Image(filename=str(self.output_dir / path)))
        except ImportError:
            print(json.dumps(summary, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    worker_parser = commands.add_parser("_worker")
    worker_parser.add_argument("--kernel", choices=KERNELS, required=True)
    worker_parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    worker_parser.add_argument("--tokens", type=int, required=True)
    worker_parser.add_argument("--warmup-ms", type=int, required=True)
    worker_parser.add_argument("--rep-ms", type=int, required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    capacity_parser = commands.add_parser("_capacity_worker")
    capacity_parser.add_argument(
        "--profile", choices=tuple(PROFILES), default="project_formal"
    )
    capacity_parser.add_argument("--tokens", type=int, required=True)
    capacity_parser.add_argument("--validation-tokens", type=int, required=True)
    capacity_parser.add_argument("--warmup-ms", type=int, required=True)
    capacity_parser.add_argument("--rep-ms", type=int, required=True)
    capacity_parser.add_argument("--output", type=Path, required=True)
    suite = commands.add_parser("run")
    suite.add_argument(
        "--output", default="artifacts/operator_benchmark/rtx4090_24gb/dense"
    )
    suite.add_argument("--warmup-ms", type=int, default=50)
    suite.add_argument("--rep-ms", type=int, default=200)
    suite.add_argument("--case-timeout-seconds", type=int, default=1200)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "_worker":
        result = worker(
            kernel=args.kernel,
            profile_name=args.profile,
            tokens=args.tokens,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
        _atomic_json(args.output, result)
        return
    if args.command == "_capacity_worker":
        result = capacity_worker(
            profile_name=args.profile,
            tokens=args.tokens,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
            validation_tokens=args.validation_tokens,
        )
        _atomic_json(args.output, result)
        return
    benchmark = DenseOperatorBenchmark(
        args.output,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        case_timeout_seconds=args.case_timeout_seconds,
    )
    print(json.dumps(benchmark.run(), indent=2))


if __name__ == "__main__":
    main()
