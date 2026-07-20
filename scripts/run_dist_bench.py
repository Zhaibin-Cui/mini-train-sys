"""Single-node DDP/FSDP benchmark runner for the 24 GB RTX 4090 server."""

# ruff: noqa: E402 -- direct script execution needs the repository root on sys.path.

from __future__ import annotations

import argparse
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
    model = build_model(
        cfg.model, build_ops_backend(cfg.backend), activation_dtype=precision.activation_dtype
    ).to(device)
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
        "gpu_inventory": capture([
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,pci.bus_id",
            "--format=csv,noheader",
        ]),
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
    batches = [args.local_batch] if args.suite == "weak" else args.batch_sizes
    rows, failures = [], []
    raw_dir = output_dir / "raw"
    source_fingerprint = benchmark_source_fingerprint()

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
                    if (strategy, world) in CONFIGS
                ],
                "local_batch": args.local_batch,
                "batch_sizes": args.batch_sizes,
                "warmup_steps": args.warmup_steps,
                "measure_steps": args.measure_steps,
                "repeats": args.repeats,
                "model_config": str(model_config.resolve()),
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
            if (strategy, world) not in CONFIGS:
                continue
            for batch in batches:
                for repeat in range(args.repeats):
                    name = f"{args.suite}_{strategy}_{world}gpu_b{batch}_r{repeat}.json"
                    result_path = raw_dir / name
                    config_path = ROOT / CONFIGS[(strategy, world)]
                    identity_payload = {
                        "strategy": strategy,
                        "world_size": world,
                        "batch_size": batch,
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
                        sys.executable, "-m", "torch.distributed.run", "--standalone",
                        "--nproc_per_node", str(world),
                        str(Path(__file__).relative_to(ROOT)), "_worker",
                        "--config", CONFIGS[(strategy, world)],
                        "--model-config", str(model_config),
                        "--batch-size", str(batch), "--warmup-steps", str(args.warmup_steps),
                        "--measure-steps", str(args.measure_steps), "--repeat", str(repeat),
                        "--case-identity", case_identity,
                        "--output", str(result_path),
                    ]
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
                        failures.append({
                            "strategy": strategy, "world_size": world, "batch_size": batch,
                            "repeat": repeat,
                            "oom": "out of memory" in combined.lower(),
                            "timed_out": timed_out,
                            "returncode": returncode,
                            "log": str(log_path),
                            "error_tail": combined[-2000:],
                        })
                    write_report()
    if args.suite == "weak":
        for strategy in args.strategies:
            baselines = [r for r in rows if r["strategy"] == strategy and r["world_size"] == 1]
            if not baselines:
                continue
            baseline = statistics.mean(r["throughput_tokens_per_sec"] for r in baselines)
            for row in rows:
                if row["strategy"] == strategy:
                    row["weak_scaling_efficiency_percent"] = (
                        100 * row["throughput_tokens_per_sec"] / (baseline * row["world_size"])
                    )
    write_report()
    print(f"completed={len(rows)} failed={len(failures)}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory")
    inv.add_argument("--output", default="artifacts/distributed_benchmark")
    run = commands.add_parser("run")
    run.add_argument("--suite", choices=("weak", "capacity"), required=True)
    run.add_argument(
        "--strategies",
        nargs="+",
        choices=("single", "ddp", "fsdp"),
        default=["single", "ddp", "fsdp"],
    )
    run.add_argument("--world-sizes", nargs="+", type=int, choices=(1, 4, 8), default=[1, 4, 8])
    run.add_argument("--local-batch", type=int, default=4)
    run.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128]
    )
    run.add_argument("--warmup-steps", type=int, default=10)
    run.add_argument("--measure-steps", type=int, default=30)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--model-config", default="configs/model_125m_moe.yaml")
    run.add_argument("--output", default="artifacts/distributed_benchmark")
    run.add_argument("--case-timeout-seconds", type=int, default=1800)
    run.add_argument("--rerun-existing", action="store_true")
    hidden = commands.add_parser("_worker")
    for name, kwargs in (
        ("--config", {"required": True}), ("--model-config", {"required": True}),
        ("--batch-size", {"type": int, "required": True}),
        ("--warmup-steps", {"type": int, "required": True}),
        ("--measure-steps", {"type": int, "required": True}),
        ("--repeat", {"type": int, "required": True}), ("--output", {"required": True}),
        ("--case-identity", {"required": True}),
    ):
        hidden.add_argument(name, **kwargs)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "_worker":
        worker(args)
    elif args.command == "inventory":
        inventory(Path(args.output))
    else:
        suite(args)


if __name__ == "__main__":
    main()
