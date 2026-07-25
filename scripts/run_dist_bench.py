"""Single-node DDP/FSDP benchmark runner for the 24 GB RTX 4090 server."""

# ruff: noqa: E402 -- direct script execution needs the repository root on sys.path.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from minitrain.data.dataloader import build_training_dataloader
from minitrain.model.ops import get_ops_backend
from minitrain.runtime.config import experiment_config_from_dict, load_yaml_dict
from minitrain.runtime.factory import build_model, build_ops_backend, build_parallel_strategy
from minitrain.train.optim import build_optimizer
from minitrain.train.precision import resolve_precision_policy
from minitrain.train.trainer import Trainer
from minitrain.utils.seed import seed_everything


CONFIGS = {
    (strategy, world): f"configs/server/rtx4090_24gb/runs/{strategy}_{world}gpu.yaml"
    for strategy in ("ddp", "fsdp")
    for world in (1, 4, 8)
}
CONFIGS[("single", 1)] = "configs/server/rtx4090_24gb/runs/single_1gpu.yaml"


def benchmark_source_fingerprint() -> str:
    """Bind reusable raw cases to the training and kernel implementation."""

    digest = hashlib.sha256()
    candidates = [Path(__file__), ROOT / "pyproject.toml"]
    for path in (ROOT / "minitrain").rglob("*"):
        if path.is_file() and path.suffix in {".py", ".cu", ".cpp", ".h", ".cuh"}:
            candidates.append(path)
    for path in sorted(candidates):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def worker(args: argparse.Namespace) -> None:
    payload = load_yaml_dict(args.config)
    payload["model"] = load_yaml_dict(args.model_config)["model"]
    payload.setdefault("train", {})["batch_size"] = args.batch_size
    if args.ops_backend is not None:
        payload.setdefault("backend", {})["ops"] = args.ops_backend
    cfg = experiment_config_from_dict(payload)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if cfg.parallel.expected_world_size != world_size:
        raise RuntimeError(
            f"config expects {cfg.parallel.expected_world_size} ranks, got {world_size}"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    seed_everything(cfg.run.seed)
    precision = resolve_precision_policy(cfg.train.precision, device)
    ops = build_ops_backend(cfg.backend)
    model = build_model(cfg.model, ops, activation_dtype=precision.activation_dtype).to(device)
    model_seq_len = model.cfg.seq_len
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    loader = build_training_dataloader(
        cfg.data,
        seq_len=model.cfg.seq_len,
        batch_size=args.batch_size,
        vocab_size=model.cfg.vocab_size,
        seed=cfg.run.seed,
    )
    strategy = build_parallel_strategy(cfg, resolved_precision=precision.name)
    try:
        strategy.setup()
        model = strategy.wrap_model(model)
        optimizer = build_optimizer(model, cfg=cfg.optimizer)
        trainer = Trainer(
            model,
            optimizer,
            device=device,
            use_fused_loss=cfg.train.use_fused_loss,
            precision=precision.name,
            grad_clip_norm=cfg.train.grad_clip_norm,
        )
        iterator = iter(loader)
        for _ in range(args.warmup_steps):
            batch, iterator = next_batch(iterator, loader)
            trainer.train_step(batch)
        torch.cuda.synchronize(device)
        strategy.barrier()
        torch.cuda.reset_peak_memory_stats(device)

        step_ms, data_ms = [], []
        last_loss = 0.0
        for _ in range(args.measure_steps):
            started = time.perf_counter()
            batch, iterator = next_batch(iterator, loader)
            data_ready = time.perf_counter()
            loss = trainer.train_step(batch)
            torch.cuda.synchronize(device)
            finished = time.perf_counter()
            data_ms.append((data_ready - started) * 1000)
            step_ms.append((finished - started) * 1000)
            last_loss = float(loss)

        timings = torch.tensor(list(zip(step_ms, data_ms)), device=device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(timings, op=dist.ReduceOp.MAX)
        memory_max = torch.tensor(
            [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
            device=device,
            dtype=torch.float64,
        )
        memory_sum = memory_max.clone()
        if dist.is_initialized():
            dist.all_reduce(memory_max, op=dist.ReduceOp.MAX)
            dist.all_reduce(memory_sum, op=dist.ReduceOp.SUM)
        if strategy.rank == 0:
            # The parallel strategy may replace the model with DDP/FSDP, whose
            # wrapper does not expose MiniTransformer.cfg. The resolved config is
            # the stable source for model dimensions after wrapping.
            global_tokens = args.batch_size * world_size * model_seq_len
            global_step_ms = timings[:, 0].cpu().tolist()
            global_data_ms = timings[:, 1].cpu().tolist()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            result = {
                "schema_version": 1,
                "case_identity": args.case_identity,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "strategy": cfg.parallel.strategy,
                "world_size": world_size,
                "local_batch_size": args.batch_size,
                "global_batch_size": args.batch_size * world_size,
                "seq_len": model_seq_len,
                "global_tokens_per_step": global_tokens,
                "warmup_steps": args.warmup_steps,
                "measure_steps": args.measure_steps,
                "repeat": args.repeat,
                "parameter_count": parameter_count,
                "precision": precision.name,
                "workers_per_rank": loader.num_workers,
                "workers_per_node": loader.num_workers * world_size,
                "prefetch_factor": cfg.data.prefetch_factor,
                "step_time_ms_mean": statistics.mean(global_step_ms),
                "step_time_ms_p50": percentile(global_step_ms, 0.50),
                "step_time_ms_p95": percentile(global_step_ms, 0.95),
                "throughput_tokens_per_sec": statistics.mean(
                    global_tokens / (value / 1000) for value in global_step_ms
                ),
                "data_wait_ms_mean": statistics.mean(global_data_ms),
                "data_wait_ms_p95": percentile(global_data_ms, 0.95),
                "data_stall_percent": 100 * sum(global_data_ms) / sum(global_step_ms),
                "peak_memory_allocated_mb": memory_max[0].item() / 1024**2,
                "peak_memory_reserved_mb": memory_max[1].item() / 1024**2,
                "system_peak_memory_allocated_mb": memory_sum[0].item() / 1024**2,
                "system_peak_memory_reserved_mb": memory_sum[1].item() / 1024**2,
                "gpu_memory_total_mb": total_memory / 1024**2,
                "system_gpu_memory_total_mb": world_size * total_memory / 1024**2,
                "memory_utilization_percent": 100 * memory_max[0].item() / total_memory,
                "system_memory_utilization_percent": (
                    100 * memory_sum[0].item() / (world_size * total_memory)
                ),
                "last_loss": last_loss,
                "ops_backend": cfg.backend.ops,
                "backend_dispatch": (
                    ops.dispatch_summary() if hasattr(ops, "dispatch_summary") else None
                ),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device),
            }
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result))
    finally:
        strategy.teardown()


def inventory(output_dir: Path) -> dict:
    def capture(command: list[str]) -> str:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        return completed.stdout.strip() or completed.stderr.strip()

    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nvidia_smi": capture(["nvidia-smi", "-L"]),
        "gpu_inventory": capture(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,pci.bus_id",
                "--format=csv,noheader",
            ]
        ),
        "topology": capture(["nvidia-smi", "topo", "-m"]),
        "git_commit": capture(["git", "rev-parse", "HEAD"]),
        "git_status": capture(["git", "status", "--short"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inventory.json").write_text(json.dumps(data, indent=2) + "\n", "utf-8")
    print(json.dumps(data, indent=2))
    return data


def run_case(command: list[str], *, timeout_seconds: int) -> tuple[str, str, int, bool]:
    """Run one torchrun process group and tear down the whole group on timeout."""

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
        return stdout, stderr, int(process.returncode), False
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
        return stdout, stderr + "\ncase timed out", -1, True


def suite(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    inventory_data = inventory(output_dir)
    model_config = Path(args.model_config)
    if not model_config.is_absolute():
        model_config = ROOT / model_config
    batches = [args.local_batch] if args.suite in {"weak", "backend"} else args.batch_sizes
    ops_backends = args.ops_backends or [None]
    rows, failures = [], []
    raw_dir = output_dir / "raw"
    source_fingerprint = benchmark_source_fingerprint()
    custom_config = Path(args.case_config).resolve() if args.case_config else None
    if custom_config is not None:
        if len(args.strategies) != 1 or len(args.world_sizes) != 1:
            raise ValueError("--case-config requires exactly one strategy and one world size")
        if not custom_config.is_file():
            raise FileNotFoundError(custom_config)

    def config_for(strategy: str, world: int) -> Path | None:
        if custom_config is not None:
            return custom_config
        relative = CONFIGS.get((strategy, world))
        return None if relative is None else ROOT / relative

    def write_report() -> None:
        report = {
            "schema_version": 2,
            "suite": args.suite,
            "settings": {
                "strategies": args.strategies,
                "world_sizes": args.world_sizes,
                "requested_cases": [
                    {"strategy": strategy, "world_size": world}
                    for strategy in args.strategies
                    for world in args.world_sizes
                    if config_for(strategy, world) is not None
                ],
                "local_batch": args.local_batch,
                "batch_sizes": args.batch_sizes,
                "warmup_steps": args.warmup_steps,
                "measure_steps": args.measure_steps,
                "repeats": args.repeats,
                "ops_backends": [
                    backend if backend is not None else "from_config" for backend in ops_backends
                ],
                "model_config": str(model_config.resolve()),
                "case_config": str(custom_config) if custom_config is not None else None,
                "case_timeout_seconds": args.case_timeout_seconds,
            },
            "results": rows,
            "failures": failures,
        }
        (output_dir / f"{args.suite}_summary.json").write_text(
            json.dumps(report, indent=2) + "\n", "utf-8"
        )

    for strategy in args.strategies:
        for world in args.world_sizes:
            config_path = config_for(strategy, world)
            if config_path is None:
                continue
            for ops_backend in ops_backends:
                for batch in batches:
                    for repeat in range(args.repeats):
                        backend_suffix = "" if ops_backend is None else f"_{ops_backend}"
                        name = (
                            f"{args.suite}_{strategy}_{world}gpu_b{batch}"
                            f"{backend_suffix}_r{repeat}.json"
                        )
                        result_path = raw_dir / name
                        identity_payload = {
                            "strategy": strategy,
                            "world_size": world,
                            "batch_size": batch,
                            "ops_backend": ops_backend,
                            "repeat": repeat,
                            "warmup_steps": args.warmup_steps,
                            "measure_steps": args.measure_steps,
                            "source_fingerprint": source_fingerprint,
                            "environment": {
                                "torch": inventory_data["torch"],
                                "cuda": inventory_data["cuda"],
                                "gpu_inventory": inventory_data["gpu_inventory"],
                                "topology": inventory_data["topology"],
                            },
                            "config": load_yaml_dict(config_path),
                            "model": load_yaml_dict(model_config),
                        }
                        case_identity = hashlib.sha256(
                            json.dumps(identity_payload, sort_keys=True).encode("utf-8")
                        ).hexdigest()
                        if result_path.is_file() and not args.rerun_existing:
                            try:
                                existing = json.loads(result_path.read_text("utf-8"))
                            except (OSError, json.JSONDecodeError):
                                existing = {}
                            if existing.get("case_identity") == case_identity:
                                print(f"REUSE {result_path}", flush=True)
                                rows.append(existing)
                                write_report()
                                continue
                        if result_path.is_file():
                            result_path.unlink()
                        command = [
                            sys.executable,
                            "-m",
                            "torch.distributed.run",
                            "--standalone",
                            "--nproc_per_node",
                            str(world),
                            str(Path(__file__).relative_to(ROOT)),
                            "_worker",
                            "--config",
                            str(config_path),
                            "--model-config",
                            str(model_config),
                            "--batch-size",
                            str(batch),
                            "--warmup-steps",
                            str(args.warmup_steps),
                            "--measure-steps",
                            str(args.measure_steps),
                            "--repeat",
                            str(repeat),
                            "--case-identity",
                            case_identity,
                            "--output",
                            str(result_path),
                        ]
                        if ops_backend is not None:
                            command.extend(["--ops-backend", ops_backend])
                        print("RUN", " ".join(command), flush=True)
                        stdout, stderr, returncode, timed_out = run_case(
                            command, timeout_seconds=args.case_timeout_seconds
                        )
                        (output_dir / "logs").mkdir(parents=True, exist_ok=True)
                        log_path = output_dir / "logs" / name.replace(".json", ".log")
                        log_path.write_text(stdout + "\n--- STDERR ---\n" + stderr, "utf-8")
                        if returncode == 0 and result_path.is_file():
                            rows.append(json.loads(result_path.read_text("utf-8")))
                        else:
                            combined = stdout + stderr
                            failures.append(
                                {
                                    "strategy": strategy,
                                    "world_size": world,
                                    "batch_size": batch,
                                    "ops_backend": ops_backend,
                                    "repeat": repeat,
                                    "oom": "out of memory" in combined.lower(),
                                    "timed_out": timed_out,
                                    "returncode": returncode,
                                    "log": str(log_path),
                                    "error_tail": combined[-2000:],
                                }
                            )
                        write_report()
    if args.suite == "weak":
        for strategy in args.strategies:
            for ops_backend in {r["ops_backend"] for r in rows if r["strategy"] == strategy}:
                baselines = [
                    r
                    for r in rows
                    if r["strategy"] == strategy
                    and r["ops_backend"] == ops_backend
                    and r["world_size"] == 1
                ]
                if not baselines:
                    continue
                baseline = statistics.mean(r["throughput_tokens_per_sec"] for r in baselines)
                for row in rows:
                    if row["strategy"] == strategy and row["ops_backend"] == ops_backend:
                        row["weak_scaling_efficiency_percent"] = (
                            100 * row["throughput_tokens_per_sec"] / (baseline * row["world_size"])
                        )
    write_report()
    print(f"completed={len(rows)} failed={len(failures)}")


def present_backend(args: argparse.Namespace) -> None:
    """Aggregate repeated Torch/CUDA formal-launch cases and render one comparison."""

    source = Path(args.input)
    report = json.loads(source.read_text(encoding="utf-8"))
    validation_path = Path(args.validation)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("quality_gate_passed"):
        raise ValueError("CUDA backend correctness validation did not pass")
    if report.get("suite") != "backend":
        raise ValueError(f"{source} is not a backend benchmark")
    if report.get("failures"):
        raise ValueError("backend benchmark contains failed process cases")
    rows = report.get("results", [])
    backend_order = [
        backend
        for backend in ("torch", "triton", "cuda")
        if any(row.get("ops_backend") == backend for row in rows)
    ]
    grouped = {
        backend: [row for row in rows if row.get("ops_backend") == backend]
        for backend in backend_order
    }
    if any(backend not in grouped for backend in ("torch", "cuda")) or any(
        len(values) < 2 for values in grouped.values()
    ):
        raise ValueError("backend comparison requires at least two Torch and two CUDA repeats")
    identities = {
        (
            row["strategy"],
            row["world_size"],
            row["local_batch_size"],
            row["seq_len"],
            row["precision"],
            row["parameter_count"],
        )
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError("backend cases do not share the same formal launch shape")
    for row in grouped["cuda"]:
        dispatch = (row.get("backend_dispatch") or {}).get("attention", {})
        if int(dispatch.get("native_cuda", 0)) <= 0 or int(dispatch.get("fallback", 0)) != 0:
            raise ValueError("CUDA case did not exclusively use native CUDA attention")

    metrics = (
        "throughput_tokens_per_sec",
        "step_time_ms_mean",
        "peak_memory_allocated_mb",
        "peak_memory_reserved_mb",
    )
    aggregates = []
    for backend, values in grouped.items():
        aggregate = {"ops_backend": backend, "repeats": len(values)}
        for metric in metrics:
            samples = [float(row[metric]) for row in values]
            aggregate[f"{metric}_mean"] = statistics.mean(samples)
            aggregate[f"{metric}_stdev"] = statistics.stdev(samples)
        aggregates.append(aggregate)
    by_backend = {row["ops_backend"]: row for row in aggregates}
    comparison = {}
    for candidate, baseline in (
        ("triton", "torch"),
        ("cuda", "torch"),
        ("cuda", "triton"),
    ):
        if candidate not in by_backend or baseline not in by_backend:
            continue
        candidate_row = by_backend[candidate]
        baseline_row = by_backend[baseline]
        suffix = f"{candidate}_vs_{baseline}"
        comparison[f"throughput_speedup_{suffix}"] = (
            candidate_row["throughput_tokens_per_sec_mean"]
            / baseline_row["throughput_tokens_per_sec_mean"]
        )
        comparison[f"step_time_reduction_percent_{suffix}"] = 100 * (
            baseline_row["step_time_ms_mean_mean"] - candidate_row["step_time_ms_mean_mean"]
        ) / baseline_row["step_time_ms_mean_mean"]
        comparison[f"allocated_memory_reduction_percent_{suffix}"] = 100 * (
            baseline_row["peak_memory_allocated_mb_mean"]
            - candidate_row["peak_memory_allocated_mb_mean"]
        ) / baseline_row["peak_memory_allocated_mb_mean"]
        comparison[f"reserved_memory_reduction_percent_{suffix}"] = 100 * (
            baseline_row["peak_memory_reserved_mb_mean"]
            - candidate_row["peak_memory_reserved_mb_mean"]
        ) / baseline_row["peak_memory_reserved_mb_mean"]
    # Preserve the original CUDA-vs-Torch keys for existing report consumers.
    comparison["step_time_reduction_percent"] = comparison[
        "step_time_reduction_percent_cuda_vs_torch"
    ]
    comparison["allocated_memory_reduction_percent"] = comparison[
        "allocated_memory_reduction_percent_cuda_vs_torch"
    ]
    comparison["reserved_memory_reduction_percent"] = comparison[
        "reserved_memory_reduction_percent_cuda_vs_torch"
    ]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "backend_aggregates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), constrained_layout=True)
    color_map = {"torch": "#3B6EA8", "triton": "#E87722", "cuda": "#2E8B57"}
    labels = [backend.title() for backend in backend_order]
    colors = [color_map[backend] for backend in backend_order]
    panels = (
        ("throughput_tokens_per_sec_mean", "Throughput", "tokens/s"),
        ("step_time_ms_mean_mean", "Training step", "ms"),
        ("peak_memory_allocated_mb_mean", "Peak allocated memory", "MB / GPU"),
    )
    for axis, (metric, title, ylabel) in zip(axes, panels):
        values = [by_backend[backend][metric] for backend in backend_order]
        bars = axis.bar(labels, values, color=colors, width=0.62)
        axis.set_title(title, weight="bold")
        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:,.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle("Formal FSDP launch · Torch vs native-CUDA attention backend", fontsize=14)
    figure_path = output / "backend_comparison.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    result = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "validation": str(validation_path.resolve()),
        "condition": {
            "identity": list(identities)[0],
            "comparison": (
                "Torch vs Triton vs CUDA backend; CUDA-vs-Triton isolates native attention, "
                "while CUDA-vs-Torch measures the complete backend stack"
            ),
        },
        "aggregates": aggregates,
        "comparison": comparison,
        "quality_gate_passed": True,
        "figure": str(figure_path),
    }
    (output / "backend_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


def present_backend_capacity(args: argparse.Namespace) -> None:
    """Select each backend's fastest repeated case under one reserved-VRAM cap."""

    source = Path(args.input)
    report = json.loads(source.read_text(encoding="utf-8"))
    if report.get("suite") != "capacity":
        raise ValueError(f"{source} is not a capacity benchmark")
    rows = report.get("results", [])
    if not rows:
        raise ValueError("capacity benchmark contains no successful cases")
    expected_repeats = int(report.get("settings", {}).get("repeats", args.min_repeats))
    required_repeats = max(args.min_repeats, expected_repeats)
    tested_batches = sorted(
        {
            int(batch)
            for batch in report.get("settings", {}).get("batch_sizes", [])
        }
        or {int(row["local_batch_size"]) for row in rows}
    )
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        backend = str(row.get("ops_backend"))
        if backend not in {"torch", "triton", "cuda"}:
            continue
        grouped.setdefault((backend, int(row["local_batch_size"])), []).append(row)

    aggregates = []
    for (backend, batch_size), values in sorted(grouped.items()):
        reserved_percent = [
            100
            * float(row["peak_memory_reserved_mb"])
            / float(row["gpu_memory_total_mb"])
            for row in values
        ]
        dispatch_valid = True
        if backend == "cuda":
            dispatch_valid = all(
                int(((row.get("backend_dispatch") or {}).get("attention") or {}).get(
                    "native_cuda", 0
                ))
                > 0
                and int(((row.get("backend_dispatch") or {}).get("attention") or {}).get(
                    "fallback", 0
                ))
                == 0
                for row in values
            )
        throughputs = [float(row["throughput_tokens_per_sec"]) for row in values]
        aggregate = {
            "ops_backend": backend,
            "local_batch_size": batch_size,
            "global_batch_size": int(values[0]["global_batch_size"]),
            "repeats_successful": len(values),
            "throughput_tokens_per_sec_mean": statistics.mean(throughputs),
            "throughput_tokens_per_sec_stdev": (
                statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
            ),
            "peak_memory_allocated_mb_mean": statistics.mean(
                float(row["peak_memory_allocated_mb"]) for row in values
            ),
            "peak_memory_reserved_mb_mean": statistics.mean(
                float(row["peak_memory_reserved_mb"]) for row in values
            ),
            "peak_reserved_percent_max": max(reserved_percent),
            "native_cuda_dispatch_valid": dispatch_valid,
        }
        aggregate["eligible"] = (
            len(values) >= required_repeats
            and max(reserved_percent) <= args.memory_limit_percent
            and dispatch_valid
        )
        aggregates.append(aggregate)

    selected = {}
    for backend in ("torch", "triton", "cuda"):
        candidates = [
            row
            for row in aggregates
            if row["ops_backend"] == backend and row["eligible"]
        ]
        if not candidates:
            raise ValueError(
                f"{backend} has no case with {required_repeats} repeats under "
                f"{args.memory_limit_percent:.1f}% reserved VRAM"
            )
        selected[backend] = max(
            candidates, key=lambda row: float(row["throughput_tokens_per_sec_mean"])
        )
    torch_throughput = float(selected["torch"]["throughput_tokens_per_sec_mean"])
    selected_rows = []
    for backend in ("torch", "triton", "cuda"):
        row = dict(selected[backend])
        row["throughput_speedup_vs_torch"] = (
            float(row["throughput_tokens_per_sec_mean"]) / torch_throughput
        )
        row["selected_is_largest_tested_batch"] = (
            int(row["local_batch_size"]) == max(tested_batches)
        )
        selected_rows.append(row)
    eligible_batches_by_backend = {
        backend: {
            int(row["local_batch_size"])
            for row in aggregates
            if row["ops_backend"] == backend and row["eligible"]
        }
        for backend in ("torch", "triton", "cuda")
    }
    common_eligible_batches = sorted(
        set.intersection(*eligible_batches_by_backend.values())
    )
    if not common_eligible_batches:
        raise ValueError(
            "Torch, Triton, and CUDA have no common repeated batch under the VRAM limit"
        )
    common_fixed_batch = max(common_eligible_batches)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("capacity_aggregates.csv", aggregates),
        ("capacity_selected.csv", selected_rows),
    ):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["ops_backend"]).title() for row in selected_rows]
    throughputs = [
        float(row["throughput_tokens_per_sec_mean"]) for row in selected_rows
    ]
    reserved = [float(row["peak_reserved_percent_max"]) for row in selected_rows]
    colors = ["#3B6EA8", "#E87722", "#2E8B57"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    axes[0].bar(labels, throughputs, color=colors)
    axes[0].set_title("Best throughput under fixed VRAM cap")
    axes[0].set_ylabel("tokens/s")
    axes[1].bar(labels, reserved, color=colors)
    axes[1].axhline(args.memory_limit_percent, color="#b91c1c", linestyle="--")
    axes[1].set_title("Peak reserved VRAM")
    axes[1].set_ylabel("% per GPU")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure_path = output / "capacity_backend_comparison.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    result = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "selection_rule": {
            "memory_metric": "peak_memory_reserved_mb / gpu_memory_total_mb",
            "memory_limit_percent": args.memory_limit_percent,
            "minimum_successful_repeats": required_repeats,
            "objective": "maximum mean training throughput among eligible cases",
        },
        "tested_batches": tested_batches,
        "common_eligible_batches": common_eligible_batches,
        "common_fixed_batch": common_fixed_batch,
        "selected": selected_rows,
        "boundary_recommendations": [
            row["ops_backend"]
            for row in selected_rows
            if row["selected_is_largest_tested_batch"]
        ],
        "quality_gate_passed": not any(
            row["selected_is_largest_tested_batch"] for row in selected_rows
        ),
        "figure": str(figure_path),
    }
    (output / "capacity_backend_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


def validate_backend(args: argparse.Namespace) -> None:
    """Compare formal D64 CUDA attention against Torch and smoke dropout backward."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA backend validation requires a CUDA GPU")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    shape = (args.batch_size, 12, args.sequence_length, 64)
    torch.manual_seed(17)
    base = [torch.randn(shape, device=device, dtype=dtype) for _ in range(3)]
    gradient = torch.randn(shape, device=device, dtype=dtype)

    def evaluate(backend_name: str):
        tensors = [tensor.detach().clone().requires_grad_(True) for tensor in base]
        backend = get_ops_backend(backend_name)
        output = backend.attention(
            *tensors,
            is_causal=True,
            dropout_p=0.0,
        )
        (output * gradient).sum().backward()
        return output.detach(), [tensor.grad.detach() for tensor in tensors], backend

    reference, reference_grads, _ = evaluate("torch")
    candidate, candidate_grads, cuda_backend = evaluate("cuda")

    def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
        difference = (actual.float() - expected.float()).abs()
        relative = difference / expected.float().abs().clamp_min(1e-8)
        return {
            "max_abs": float(difference.max()),
            "max_rel": float(relative.max()),
        }

    checks = {"forward": error_stats(candidate, reference)}
    for name, actual, expected in zip(("dq", "dk", "dv"), candidate_grads, reference_grads):
        checks[name] = error_stats(actual, expected)
    torch.testing.assert_close(candidate, reference, atol=5e-2, rtol=5e-2)
    for actual, expected in zip(candidate_grads, reference_grads):
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)

    dropout_inputs = [
        torch.randn(shape, device=device, dtype=dtype, requires_grad=True) for _ in range(3)
    ]
    dropout_output = cuda_backend.attention(
        *dropout_inputs,
        is_causal=True,
        dropout_p=0.1,
    )
    dropout_output.float().square().mean().backward()
    dropout_finite = bool(
        torch.isfinite(dropout_output).all()
        and all(
            tensor.grad is not None and torch.isfinite(tensor.grad).all()
            for tensor in dropout_inputs
        )
    )
    dispatch = cuda_backend.dispatch_summary()
    native_calls = int(dispatch["attention"]["native_cuda"])
    fallback_calls = int(dispatch["attention"]["fallback"])
    result = {
        "schema_version": 1,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "shape": shape,
        "causal": True,
        "no_dropout_torch_reference": checks,
        "dropout_probability": 0.1,
        "dropout_forward_backward_finite": dropout_finite,
        "backend_dispatch": dispatch,
        "quality_gate_passed": dropout_finite and native_calls == 2 and fallback_calls == 0,
    }
    if not result["quality_gate_passed"]:
        raise RuntimeError(f"CUDA backend validation failed: {result}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory")
    inv.add_argument("--output", default="artifacts/distributed_benchmark")
    presentation = commands.add_parser("present-backend")
    presentation.add_argument("--input", required=True)
    presentation.add_argument("--validation", required=True)
    presentation.add_argument("--output", required=True)
    capacity_presentation = commands.add_parser("present-capacity")
    capacity_presentation.add_argument("--input", required=True)
    capacity_presentation.add_argument("--output", required=True)
    capacity_presentation.add_argument("--memory-limit-percent", type=float, default=92.0)
    capacity_presentation.add_argument("--min-repeats", type=int, default=2)
    validation = commands.add_parser("validate-backend")
    validation.add_argument("--device", default="cuda:0")
    validation.add_argument("--batch-size", type=int, default=2)
    validation.add_argument("--sequence-length", type=int, default=512)
    validation.add_argument("--output", required=True)
    run = commands.add_parser("run")
    run.add_argument("--suite", choices=("weak", "capacity", "backend"), required=True)
    run.add_argument(
        "--strategies",
        nargs="+",
        choices=("single", "ddp", "fsdp"),
        default=["single", "ddp", "fsdp"],
    )
    run.add_argument("--world-sizes", nargs="+", type=int, choices=(1, 4, 8), default=[1, 4, 8])
    run.add_argument("--local-batch", type=int, default=4)
    run.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    run.add_argument("--warmup-steps", type=int, default=10)
    run.add_argument("--measure-steps", type=int, default=30)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--model-config", default="configs/model_125m_moe.yaml")
    run.add_argument("--case-config")
    run.add_argument("--ops-backends", nargs="+", choices=("torch", "triton", "cuda"))
    run.add_argument("--output", default="artifacts/distributed_benchmark")
    run.add_argument("--case-timeout-seconds", type=int, default=1800)
    run.add_argument("--rerun-existing", action="store_true")
    hidden = commands.add_parser("_worker")
    for name, kwargs in (
        ("--config", {"required": True}),
        ("--model-config", {"required": True}),
        ("--batch-size", {"type": int, "required": True}),
        ("--warmup-steps", {"type": int, "required": True}),
        ("--measure-steps", {"type": int, "required": True}),
        ("--repeat", {"type": int, "required": True}),
        ("--output", {"required": True}),
        ("--case-identity", {"required": True}),
        ("--ops-backend", {"choices": ("torch", "triton", "cuda")}),
    ):
        hidden.add_argument(name, **kwargs)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "_worker":
        worker(args)
    elif args.command == "inventory":
        inventory(Path(args.output))
    elif args.command == "present-backend":
        present_backend(args)
    elif args.command == "present-capacity":
        present_backend_capacity(args)
    elif args.command == "validate-backend":
        validate_backend(args)
    else:
        suite(args)


if __name__ == "__main__":
    main()
