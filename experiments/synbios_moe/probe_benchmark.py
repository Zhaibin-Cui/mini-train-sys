"""Capacity and throughput regression for paper-style probe batches."""

from __future__ import annotations

import json
import time
from pathlib import Path
from collections.abc import Sequence
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from experiments.synbios_moe.probes import AttributeProbe, ProbeBatchItem, collate_probe
from minitrain.runtime.monitoring import GpuUtilizationMonitor


PAPER_BATCH_SIZES = {"p": 50, "q": 200}


def parse_batch_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("batch sizes must be a comma-separated list of positive integers")
    return tuple(sorted(sizes))


def _benchmark_items(dataset: Dataset, count: int) -> list[ProbeBatchItem]:
    longest = getattr(dataset, "longest_items", None)
    if callable(longest):
        return longest(count)
    return [dataset[index] for index in range(min(count, len(dataset)))]


def benchmark_probe_batches(
    backbone: torch.nn.Module,
    dataset: Dataset,
    *,
    kind: str,
    num_classes: int,
    rank: int,
    batch_sizes: Sequence[int],
    device: torch.device,
    mode: str = "training",
    warmup_steps: int = 3,
    measure_steps: int = 10,
    memory_limit_percent: float = 92.0,
    on_result: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Measure worst-length train or validation steps under a memory cap."""

    if kind not in PAPER_BATCH_SIZES:
        raise ValueError("kind must be p or q")
    if mode not in {"training", "validation"}:
        raise ValueError("mode must be training or validation")
    if warmup_steps < 0 or measure_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and measure_steps must be positive")
    if not 0 < memory_limit_percent <= 100:
        raise ValueError("memory_limit_percent must be in (0, 100]")
    candidates = tuple(sorted(set(int(size) for size in batch_sizes)))
    if not candidates or candidates[0] <= 0:
        raise ValueError("batch_sizes must contain positive integers")

    items = _benchmark_items(dataset, max(candidates))
    if len(items) < max(candidates):
        raise ValueError("dataset is smaller than the largest benchmark batch")
    results: list[dict[str, object]] = []
    backbone.to(device)

    for batch_size in candidates:
        monitor = GpuUtilizationMonitor(device)
        monitor.start()
        record: dict[str, object] = {"batch_size": batch_size, "status": "completed"}
        probe = optimizer = input_ids = positions = labels = None
        logits = loss = expanded = None
        try:
            probe = AttributeProbe(backbone, num_classes, rank=rank, kind=kind).to(device)
            probe.train(mode == "training")
            optimizer = (
                torch.optim.AdamW(
                    (parameter for parameter in probe.parameters() if parameter.requires_grad),
                    lr=1e-3,
                    weight_decay=0.3,
                    eps=1e-6,
                )
                if mode == "training"
                else None
            )
            input_ids, positions, labels = collate_probe(items[:batch_size])
            input_ids = input_ids.to(device, non_blocking=True)
            positions = positions.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            timings = []
            for step in range(warmup_steps + measure_steps):
                started = time.perf_counter()
                with torch.set_grad_enabled(mode == "training"):
                    logits = probe(input_ids, positions)
                    expanded = labels[:, None].expand(-1, logits.shape[1])
                    loss = F.cross_entropy(logits.flatten(0, 1), expanded.reshape(-1))
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if step >= warmup_steps:
                    timings.append(time.perf_counter() - started)
            mean_seconds = sum(timings) / len(timings)
            if device.type == "cuda":
                peak_allocated = torch.cuda.max_memory_allocated(device)
                peak_reserved = torch.cuda.max_memory_reserved(device)
                capacity = torch.cuda.get_device_properties(device).total_memory
                record.update(
                    {
                        # Reserved memory is the physical VRAM safety boundary;
                        # allocated memory is retained for diagnosis.
                        "peak_memory_mb": round(peak_reserved / 1024**2, 2),
                        "peak_memory_allocated_mb": round(peak_allocated / 1024**2, 2),
                        "peak_memory_reserved_mb": round(peak_reserved / 1024**2, 2),
                        "memory_capacity_mb": round(capacity / 1024**2, 2),
                        "peak_memory_percent": round(100 * peak_reserved / capacity, 2),
                    }
                )
            record.update(
                {
                    "step_time_ms": round(1000 * mean_seconds, 3),
                    "examples_per_second": round(batch_size / mean_seconds, 3),
                    "sequence_length": int(input_ids.shape[1]),
                    "loss": float(loss.detach()),
                    **monitor.read_interval(),
                }
            )
        except torch.OutOfMemoryError as exc:
            record.update({"status": "oom", "error": str(exc)})
        finally:
            monitor.close()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            del logits, loss, expanded
            del optimizer, probe, input_ids, positions, labels
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results.append(record)
        if on_result is not None:
            on_result(dict(record))

    safe = [
        result
        for result in results
        if result["status"] == "completed"
        and float(result.get("peak_memory_percent", 0.0)) <= memory_limit_percent
    ]
    recommended = max(safe, key=lambda item: float(item["examples_per_second"])) if safe else None
    paper_batch = PAPER_BATCH_SIZES[kind]
    paper_result = next((item for item in results if item["batch_size"] == paper_batch), None)
    return {
        "kind": kind,
        "mode": mode,
        "device": str(device),
        "paper_training_batch_size": paper_batch,
        "paper_batch_safe": bool(paper_result and paper_result in safe),
        "memory_limit_percent": memory_limit_percent,
        "recommended_capacity_batch_size": (
            int(recommended["batch_size"]) if recommended is not None else None
        ),
        "recommendation_scope": f"{mode}_capacity_profile",
        "results": results,
    }


def summarize_probe_benchmarks(paths: Sequence[str | Path]) -> dict[str, object]:
    """Choose batches that are safe on every measured GPU replica."""

    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if not payloads:
        raise ValueError("at least one benchmark result is required")
    summary: dict[str, object] = {"runs": [str(Path(path).resolve()) for path in paths]}
    recommendations = {}

    def aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
        candidate_sets = [
            tuple(int(result["batch_size"]) for result in run["results"]) for run in runs
        ]
        if any(candidates != candidate_sets[0] for candidates in candidate_sets[1:]):
            raise ValueError("benchmark replicas used different batch-size candidates")
        limits = {float(run["memory_limit_percent"]) for run in runs}
        if len(limits) != 1:
            raise ValueError("benchmark replicas used different memory limits")
        for key in ("attribute", "target", "rank", "checkpoint", "probe_cache"):
            values = {str(run.get(key)) for run in runs if key in run}
            if len(values) > 1 or (values and any(key not in run for run in runs)):
                raise ValueError(f"benchmark replicas disagree on {key}")
        safe_maps = []
        for run in runs:
            limit = float(run["memory_limit_percent"])
            safe_maps.append(
                {
                    int(result["batch_size"]): result
                    for result in run["results"]
                    if result["status"] == "completed"
                    and float(result.get("peak_memory_percent", 0.0)) <= limit
                }
            )
        common = set(safe_maps[0])
        for safe in safe_maps[1:]:
            common &= set(safe)
        aggregate = []
        for batch_size in sorted(common):
            replicas = [safe[batch_size] for safe in safe_maps]
            aggregate.append(
                {
                    "batch_size": batch_size,
                    "examples_per_second_mean": sum(
                        float(item["examples_per_second"]) for item in replicas
                    )
                    / len(replicas),
                    "peak_memory_percent_max": max(
                        float(item["peak_memory_percent"]) for item in replicas
                    ),
                }
            )
        selected = (
            max(aggregate, key=lambda item: float(item["examples_per_second_mean"]))
            if aggregate
            else None
        )
        return {
            "replicas": len(runs),
            "tested_batch_sizes": list(candidate_sets[0]),
            "largest_tested_batch_size": max(candidate_sets[0]),
            "recommended_batch_size": (
                int(selected["batch_size"]) if selected is not None else None
            ),
            "recommended_is_search_boundary": bool(
                selected is not None
                and int(selected["batch_size"]) == max(candidate_sets[0])
            ),
            "safe_candidates": aggregate,
        }

    for kind in PAPER_BATCH_SIZES:
        kind_runs = [payload for payload in payloads if payload.get("kind") == kind]
        if not kind_runs:
            continue
        training_runs = [
            payload for payload in kind_runs if payload.get("mode", "training") == "training"
        ]
        validation_runs = [
            payload for payload in kind_runs if payload.get("mode") == "validation"
        ]
        training = aggregate_runs(training_runs) if training_runs else None
        validation = aggregate_runs(validation_runs) if validation_runs else training
        training_safe = {
            int(item["batch_size"])
            for item in (training or {}).get("safe_candidates", [])
        }
        recommendations[kind] = {
            "training_replicas": len(training_runs),
            "validation_replicas": len(validation_runs),
            "paper_training_batch_size": PAPER_BATCH_SIZES[kind],
            "paper_batch_safe_on_all": PAPER_BATCH_SIZES[kind] in training_safe,
            "recommended_training_batch_size": (
                training.get("recommended_batch_size") if training is not None else None
            ),
            "recommended_validation_batch_size": (
                validation.get("recommended_batch_size") if validation is not None else None
            ),
            "training_safe_candidates": (
                training.get("safe_candidates", []) if training is not None else []
            ),
            "validation_safe_candidates": (
                validation.get("safe_candidates", []) if validation is not None else []
            ),
            "training_recommended_is_search_boundary": (
                bool(training.get("recommended_is_search_boundary"))
                if training is not None
                else False
            ),
            "validation_recommended_is_search_boundary": (
                bool(validation.get("recommended_is_search_boundary"))
                if validation is not None
                else False
            ),
        }
    expected_matrix = {(kind, mode) for kind in PAPER_BATCH_SIZES for mode in ("training", "validation")}
    actual_matrix = {
        (str(payload.get("kind")), str(payload.get("mode", "training")))
        for payload in payloads
    }
    missing_matrix = sorted(f"{kind}/{mode}" for kind, mode in expected_matrix - actual_matrix)
    insufficient_replicas = sorted(
        f"{kind}/{mode}"
        for kind, mode in expected_matrix
        if sum(
            payload.get("kind") == kind and payload.get("mode", "training") == mode
            for payload in payloads
        )
        < 2
    )
    missing_recommendations = sorted(
        f"{kind}/{mode}"
        for kind, recommendation in recommendations.items()
        for mode in ("training", "validation")
        if recommendation.get(f"recommended_{mode}_batch_size") is None
    )
    boundary_recommendations = sorted(
        f"{kind}/{mode}"
        for kind, recommendation in recommendations.items()
        for mode in ("training", "validation")
        if recommendation.get(f"{mode}_recommended_is_search_boundary")
    )
    summary.update(
        {
            "recommendations": recommendations,
            "missing_matrix": missing_matrix,
            "insufficient_replicas": insufficient_replicas,
            "missing_recommendations": missing_recommendations,
            "boundary_recommendations": boundary_recommendations,
            "ready_for_formal": not any(
                (
                    missing_matrix,
                    insufficient_replicas,
                    missing_recommendations,
                    boundary_recommendations,
                )
            ),
        }
    )
    return summary


def probe_batch_environment(summary: dict[str, object]) -> str:
    """Render a sourceable shell file only for a complete, bracketed benchmark."""

    if not summary.get("ready_for_formal"):
        raise ValueError("probe batch benchmark is not ready for formal use")
    recommendations = summary["recommendations"]
    values = {
        "P_BATCH_SIZE": recommendations["p"]["recommended_training_batch_size"],
        "Q_BATCH_SIZE": recommendations["q"]["recommended_training_batch_size"],
        "P_VALIDATION_BATCH_SIZE": recommendations["p"][
            "recommended_validation_batch_size"
        ],
        "Q_VALIDATION_BATCH_SIZE": recommendations["q"][
            "recommended_validation_batch_size"
        ],
    }
    return "".join(f"export {name}={int(value)}\n" for name, value in values.items())
