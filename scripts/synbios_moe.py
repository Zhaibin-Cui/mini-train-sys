"""Prepare, probe, and analyze the Allen-Zhu bioS MoE reproduction."""

# ruff: noqa: E402 -- direct script execution needs the repository root on sys.path.

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader
from torch.torch_version import TorchVersion

from experiments.synbios_moe.data import ATTRIBUTES, write_dataset
from experiments.synbios_moe.cloze_evaluation import (
    evaluate_progressive_biography_cloze,
    summarize_progressive_cloze_results,
)
from experiments.synbios_moe.evaluation import evaluate_attribute_tokens
from experiments.synbios_moe.diagnostic_report import build_diagnostic_report_artifacts
from experiments.synbios_moe.formal_report import build_formal_report_artifacts
from experiments.synbios_moe.probe_data import (
    CachedProbeDataset,
    build_probe_cache,
    validate_probe_cache,
)
from experiments.synbios_moe.probe_diagnostics import (
    WHOLE_ATTRIBUTES,
    bad_case_route_validation,
    oracle_first_token_validation,
)
from experiments.synbios_moe.repository_audit import build_repository_audit
from experiments.synbios_moe.probe_checkpoint import save_probe_result
from experiments.synbios_moe.probe_benchmark import (
    benchmark_probe_batches,
    parse_batch_sizes,
    probe_batch_environment,
    summarize_probe_benchmarks,
)
from experiments.synbios_moe.probe_pipeline import (
    ProbePipelineState,
    ProbeRuntimeConfig,
    build_pipeline_identity,
    common_pipeline_identity,
    estimate_phase_durations,
    jobs_for_stage,
    load_pipeline_config,
    probe_train_command_builder,
    probe_validation_command_builder,
    reusable_cloze_gate,
    require_matching_identity,
    resolve_devices,
    schedule_jobs,
    summarize_probe_results,
    write_json_atomic,
)
from experiments.synbios_moe.probes import (
    AttributeProbe,
    PProbeDataset,
    QProbeDataset,
    active_parameter_estimate,
    collate_probe,
    evaluate as evaluate_probe,
    train_probe,
)
from experiments.synbios_moe.router_analysis import analyze_batch
from minitrain.data.documents import CleaningConfig
from minitrain.data.preprocess import prepare_token_shards
from minitrain.data.tokenizer import TiktokenTokenizer
from minitrain.model import ModelConfig
from minitrain.model.transformer import MiniTransformer
from minitrain.train.checkpoint import load_model_state_dict_from_checkpoint
from minitrain.model.ops import get_ops_backend
from minitrain.runtime.provenance import collect_provenance
from minitrain.runtime.config import LoggingConfig
from minitrain.runtime.logger import build_event_logger, get_run_log_dir
from minitrain.runtime.monitoring import ProgressReporter


def checkpoint_size_bytes(path: str | Path) -> int:
    checkpoint = Path(path)
    if checkpoint.is_file():
        return checkpoint.stat().st_size
    return sum(item.stat().st_size for item in checkpoint.rglob("*") if item.is_file())


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_model_path(path: str | Path) -> Path:
    checkpoint = Path(path)
    return checkpoint if checkpoint.is_file() else checkpoint / "model.pt"


@contextmanager
def command_monitor(args: argparse.Namespace, name: str):
    """Give every experiment stage the same console/JSONL/TensorBoard contract."""

    output = Path(args.output)
    default_log_dir = output.parent / "operation_logs"
    cfg = LoggingConfig(
        console=not getattr(args, "quiet", False),
        tensorboard=getattr(args, "tensorboard", False),
        jsonl=True,
        log_dir=str(getattr(args, "log_dir", None) or default_log_dir),
        flush_secs=5,
    )
    run_name = f"synbios_{name}_{output.stem}"
    log_dir = get_run_log_dir(cfg, run_name=run_name)
    logger = build_event_logger(cfg, run_name=run_name, tensorboard_log_dir=log_dir)
    try:
        yield logger, log_dir
    finally:
        logger.close()


def load_model(
    model_config: str,
    checkpoint: str,
    device: torch.device,
    *,
    logger=None,
) -> MiniTransformer:
    progress = (
        ProgressReporter("model_load", 1, logger, device, unit="step")
        if logger is not None
        else None
    )
    payload = yaml.safe_load(Path(model_config).read_text(encoding="utf-8"))
    model = MiniTransformer(ModelConfig(**payload["model"]), get_ops_backend("torch"))
    # Training checkpoints also contain Adam/scheduler/RNG state for exact
    # resume. Probes and evaluation deliberately load only the model weights.
    state = load_model_state_dict_from_checkpoint(checkpoint)
    model.load_state_dict(state)
    model = model.to(device)
    if progress is not None:
        progress.update(
            1,
            metrics={
                "parameters": float(sum(parameter.numel() for parameter in model.parameters())),
                "checkpoint_mb": checkpoint_size_bytes(checkpoint) / 1024**2,
            },
        )
    return model


def command_prepare(args: argparse.Namespace) -> None:
    """Generate symbolic biographies, then build training token shards."""

    # Keep experiment records (profiles/spans) alongside the separately
    # optimized token-shard representation consumed by pretraining.
    with command_monitor(args, "prepare") as (logger, log_dir):
        progress = ProgressReporter("prepare", 2, logger, torch.device("cpu"), unit="step")
        manifest = write_dataset(
            args.output, num_people=args.num_people, variant=args.variant, seed=args.seed
        )
        progress.update(1, items=args.num_people, metrics={"phase": 1.0})
        tokenizer = TiktokenTokenizer("gpt2")
        token_manifest = prepare_token_shards(
            [Path(args.output) / "biographies.jsonl"],
            output_dir=Path(args.output) / "token_shards",
            tokenizer=tokenizer,
            cleaning=CleaningConfig(min_chars=1),
            max_document_chars=100_000,
            max_shard_tokens=args.max_shard_tokens,
            validation_fraction=0.0,
            split_seed=args.seed,
        )
        progress.update(2, metrics={"phase": 2.0})
        print(
            json.dumps(
                {
                    "dataset_manifest": str(manifest),
                    "token_manifest": str(token_manifest),
                    "log_dir": str(log_dir) if log_dir is not None else None,
                }
            )
        )


def command_cache_probes(args: argparse.Namespace) -> None:
    """Materialize all P/Q inputs and labels once for every independent task."""

    last_reported = 0

    def report(examples: int) -> None:
        nonlocal last_reported
        if examples > last_reported:
            print(json.dumps({"stage": "probe_cache", "p_examples": examples}), flush=True)
            last_reported = examples

    manifest = build_probe_cache(
        args.data,
        args.output,
        force=args.force,
        require_coverage=args.require_coverage,
        progress=report,
    )
    result = validate_probe_cache(manifest.parent)
    result["manifest"] = str(manifest.resolve())
    print(json.dumps(result))


def command_validate_cache(args: argparse.Namespace) -> None:
    print(json.dumps(validate_probe_cache(args.probe_cache, args.data), indent=2))


def build_probe_dataset(
    *,
    data: str | Path,
    cache: str | Path | None,
    kind: str,
    attribute: str,
    target: str,
    split: str,
):
    """Construct one explicit probe dataset without coupling it to CLI state."""

    if cache:
        validate_probe_cache(cache, data, include_missing_classes=False)
        return CachedProbeDataset(
            cache,
            kind=kind,
            attribute=attribute,
            target=target,
            split=split,
        )
    dataset_type = PProbeDataset if kind == "p" else QProbeDataset
    return dataset_type(data, attribute=attribute, target=target, split=split)


def command_probe(args: argparse.Namespace) -> None:
    """Train one paper-style P/Q probe against a frozen checkpoint."""

    device = torch.device(args.device)
    with command_monitor(args, f"{args.kind}_probe") as (logger, log_dir):
        model = load_model(args.model_config, args.checkpoint, device, logger=logger)
        train_data = build_probe_dataset(
            data=args.data,
            cache=args.probe_cache,
            kind=args.kind,
            attribute=args.attribute,
            target=args.target,
            split="train",
        )
        validation_data = build_probe_dataset(
            data=args.data,
            cache=args.probe_cache,
            kind=args.kind,
            attribute=args.attribute,
            target=args.target,
            split="validation",
        )
        rank = args.rank or (2 if args.kind == "p" else 16)
        probe = AttributeProbe(model, len(train_data.class_names), rank=rank, kind=args.kind)
        batch_size = args.batch_size or (50 if args.kind == "p" else 200)
        checkpoint_model_sha256 = args.checkpoint_model_sha256 or _sha256_file(
            _checkpoint_model_path(args.checkpoint)
        )
        cache_manifest_sha256 = (
            _sha256_file(Path(args.probe_cache) / "manifest.json") if args.probe_cache else None
        )
        recovery_metadata = {
            "kind": args.kind,
            "attribute": args.attribute,
            "target": args.target,
            "rank": rank,
            "batch_size": batch_size,
            "steps": args.steps,
            "seed": args.seed,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_model_sha256": checkpoint_model_sha256,
            "probe_cache": str(Path(args.probe_cache).resolve()) if args.probe_cache else None,
            "probe_cache_manifest_sha256": cache_manifest_sha256,
        }
        result = train_probe(
            probe,
            train_data,
            validation_data,
            device=device,
            batch_size=batch_size,
            steps=args.steps,
            seed=args.seed,
            logger=logger,
            log_interval=args.log_interval,
            recovery_path=args.recovery_checkpoint,
            checkpoint_interval_steps=args.checkpoint_interval_steps,
            recovery_metadata=recovery_metadata,
            resume=args.resume_probe,
            evaluate_train=args.evaluate_train,
            evaluate_validation=not args.skip_final_validation,
            evaluation_batch_size=args.evaluation_batch_size,
        )
        result.update(
            {
                "kind": args.kind,
                "attribute": args.attribute,
                "target": args.target,
                "rank": rank,
                "classes": len(train_data.class_names),
                "class_names": list(train_data.class_names),
                "model_parameters": active_parameter_estimate(model),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_model_sha256": checkpoint_model_sha256,
                "checkpoint_bytes": checkpoint_size_bytes(args.checkpoint),
                "dataset_manifest": json.loads(
                    (Path(args.data) / "manifest.json").read_text(encoding="utf-8")
                ),
                "probe_cache": str(Path(args.probe_cache).resolve()) if args.probe_cache else None,
                "probe_cache_manifest_sha256": cache_manifest_sha256,
                "provenance": collect_provenance(ROOT),
                "log_dir": str(log_dir) if log_dir is not None else None,
            }
        )
        output = Path(args.output)
        write_json_atomic(output, result)
        save_probe_result(output.with_suffix(".pt"), probe=probe, result=result)
        print(json.dumps(result))


def command_validate_probe(args: argparse.Namespace) -> None:
    """Re-evaluate a saved probe on the held-out person split."""

    # Probe checkpoints are produced locally by this CLI. TorchVersion is a str
    # subclass embedded by provenance and must be explicitly allowlisted.
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(args.probe_checkpoint, map_location="cpu", weights_only=True)
    metadata = payload["result"]
    for name in ("kind", "attribute", "target", "rank"):
        if name not in metadata:
            raise ValueError(f"probe checkpoint is missing result.{name}")
    kind = str(metadata["kind"])
    attribute = str(metadata["attribute"])
    target = str(metadata["target"])
    saved_backbone = metadata.get("checkpoint")
    requested_backbone = str(Path(args.checkpoint).resolve())
    if (
        saved_backbone
        and str(Path(saved_backbone).resolve()) != requested_backbone
        and not args.allow_checkpoint_mismatch
    ):
        raise SystemExit(
            "probe was trained against a different backbone checkpoint; "
            "pass --allow-checkpoint-mismatch only for an intentional ablation"
        )
    device = torch.device(args.device)
    with command_monitor(args, f"{kind}_probe_validation") as (logger, log_dir):
        dataset = build_probe_dataset(
            data=args.data,
            cache=args.probe_cache,
            kind=kind,
            attribute=attribute,
            target=target,
            split="validation",
        )
        saved_class_names = metadata.get("class_names")
        if saved_class_names is not None and list(saved_class_names) != list(dataset.class_names):
            raise SystemExit("probe checkpoint class mapping does not match the validation cache")
        saved_cache_sha = metadata.get("probe_cache_manifest_sha256")
        if args.probe_cache and saved_cache_sha is not None:
            current_cache_sha = _sha256_file(Path(args.probe_cache) / "manifest.json")
            if saved_cache_sha != current_cache_sha:
                raise SystemExit("probe checkpoint was trained with a different probe cache manifest")
        saved_model_sha = metadata.get("checkpoint_model_sha256")
        if saved_model_sha is not None:
            current_model_sha = args.checkpoint_model_sha256 or _sha256_file(
                _checkpoint_model_path(args.checkpoint)
            )
            if saved_model_sha != current_model_sha and not args.allow_checkpoint_mismatch:
                raise SystemExit("probe checkpoint was trained with different backbone weights")
        model = load_model(args.model_config, args.checkpoint, device, logger=logger)
        probe = AttributeProbe(
            model,
            len(dataset.class_names),
            rank=int(metadata["rank"]),
            kind=kind,
        )
        incompatible = probe.load_state_dict(payload["probe"], strict=False)
        if incompatible.unexpected_keys or any(
            not key.startswith("backbone.") for key in incompatible.missing_keys
        ):
            raise ValueError(f"incompatible probe state: {incompatible}")
        batch_size = args.batch_size or (50 if kind == "p" else 200)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_probe,
            pin_memory=device.type == "cuda",
        )
        progress = ProgressReporter(
            "probe_validation",
            len(loader),
            logger,
            device,
            log_interval=max(1, min(args.log_interval, len(loader))),
            unit="batch",
        )
        accuracy = evaluate_probe(probe.to(device), loader, device, progress=progress)
        result = {
            "kind": kind,
            "attribute": attribute,
            "target": target,
            "rank": int(metadata["rank"]),
            "classes": len(dataset.class_names),
            "class_names": list(dataset.class_names),
            "examples": len(dataset),
            "validation_accuracy": accuracy,
            "probe_checkpoint": str(Path(args.probe_checkpoint).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "probe_cache": str(Path(args.probe_cache).resolve()) if args.probe_cache else None,
            "dataset_manifest": json.loads(
                (Path(args.data) / "manifest.json").read_text(encoding="utf-8")
            ),
            "monitoring": progress.summary(),
            "log_dir": str(log_dir) if log_dir is not None else None,
        }
        output = Path(args.output)
        write_json_atomic(output, result)
        print(json.dumps(result))


def command_probe_benchmark(args: argparse.Namespace) -> None:
    """Benchmark conservative P/Q batch candidates on one assigned GPU."""

    device = torch.device(args.device)
    sizes = parse_batch_sizes(args.batch_sizes)
    with command_monitor(args, f"{args.kind}_probe_batch_benchmark") as (logger, log_dir):
        model = load_model(args.model_config, args.checkpoint, device, logger=logger)
        dataset = build_probe_dataset(
            data=args.data,
            cache=args.probe_cache,
            kind=args.kind,
            attribute=args.attribute,
            target=args.target,
            split="train" if args.mode == "training" else "validation",
        )
        rank = args.rank or (2 if args.kind == "p" else 16)
        progress = ProgressReporter(
            "probe_batch_benchmark",
            len(sizes),
            logger,
            device,
            unit="batch",
        )
        completed = 0

        def report(record: dict[str, object]) -> None:
            nonlocal completed
            completed += 1
            metrics = {
                key: float(value)
                for key, value in record.items()
                if isinstance(value, (int, float)) and key != "batch_size"
            }
            metrics["candidate_batch_size"] = float(record["batch_size"])
            metrics["candidate_completed"] = float(record["status"] == "completed")
            progress.update(completed, metrics=metrics)

        result = benchmark_probe_batches(
            model,
            dataset,
            kind=args.kind,
            num_classes=len(dataset.class_names),
            rank=rank,
            batch_sizes=sizes,
            device=device,
            mode=args.mode,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            memory_limit_percent=args.memory_limit_percent,
            on_result=report,
        )
        result.update(
            {
                "attribute": args.attribute,
                "target": args.target,
                "rank": rank,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "probe_cache": str(Path(args.probe_cache).resolve()),
                "monitoring": progress.summary(),
                "provenance": collect_provenance(ROOT),
                "log_dir": str(log_dir) if log_dir is not None else None,
            }
        )
        write_json_atomic(args.output, result)
        print(json.dumps(result))


def command_summarize_probe_benchmarks(args: argparse.Namespace) -> None:
    result = summarize_probe_benchmarks(args.run)
    write_json_atomic(args.output, result)
    if args.require_complete_search and not result["ready_for_formal"]:
        reasons = {
            key: result[key]
            for key in (
                "missing_matrix",
                "insufficient_replicas",
                "missing_recommendations",
                "boundary_recommendations",
            )
            if result[key]
        }
        raise SystemExit(
            "probe batch search is not ready for formal use; inspect/expand candidates: "
            + json.dumps(reasons, sort_keys=True)
        )
    if args.env_output:
        destination = Path(args.env_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(probe_batch_environment(result), encoding="utf-8")
        temporary.replace(destination)
    print(json.dumps(result, indent=2))


def command_analyze(args: argparse.Namespace) -> None:
    """Measure whether top-1 expert choices correlate with an attribute label."""

    device = torch.device(args.device)
    with command_monitor(args, "analyze") as (logger, log_dir):
        model = load_model(args.model_config, args.checkpoint, device, logger=logger)
        dataset = build_probe_dataset(
            data=args.data,
            cache=args.probe_cache,
            kind="p",
            attribute=args.attribute,
            target=args.target,
            split="validation",
        )
        items = [dataset[index] for index in range(min(args.examples, len(dataset)))]
        input_ids, positions, labels = collate_probe(items)
        progress = ProgressReporter("analyze", 1, logger, device, unit="batch")
        result = analyze_batch(
            model,
            input_ids.to(device, non_blocking=device.type == "cuda"),
            positions,
            labels,
        )
        progress.update(1, items=len(items), tokens=input_ids.numel())
        result["monitoring"] = progress.summary()
        result["log_dir"] = str(log_dir) if log_dir is not None else None
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result))


def _diagnostic_attributes(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(args.attribute or WHOLE_ATTRIBUTES)


def command_validate_probe_oracle_first_token(args: argparse.Namespace) -> None:
    """Measure Q-whole recovery after an oracle first-token intervention."""

    device = torch.device(args.device)
    with command_monitor(args, "q_oracle_first_token_validation") as (logger, log_dir):
        model = load_model(args.model_config, args.checkpoint, device, logger=logger)

        def progress(attribute: str, examples: int) -> None:
            logger.log_event(
                {
                    "event": "probe_diagnostic",
                    "diagnostic": "oracle_first_token",
                    "attribute": attribute,
                    "examples": examples,
                }
            )

        result = oracle_first_token_validation(
            backbone=model,
            data_root=args.data,
            cache_root=args.probe_cache,
            probe_dir=args.probe_dir,
            output_dir=args.output,
            device=device,
            attributes=_diagnostic_attributes(args),
            batch_size=args.batch_size,
            max_examples=args.max_examples,
            backbone_checkpoint=args.checkpoint,
            progress=progress,
        )
        result["log_dir"] = str(log_dir) if log_dir is not None else None
        print(json.dumps(result, indent=2))


def command_validate_probe_bad_case_routes(args: argparse.Namespace) -> None:
    """Measure t1/t2 route branching on Q-first-correct/Q-whole-wrong cases."""

    device = torch.device(args.device)
    with command_monitor(args, "q_bad_case_route_validation") as (logger, log_dir):
        model = load_model(args.model_config, args.checkpoint, device, logger=logger)

        def progress(attribute: str, examples: int) -> None:
            logger.log_event(
                {
                    "event": "probe_diagnostic",
                    "diagnostic": "bad_case_routes",
                    "attribute": attribute,
                    "examples": examples,
                }
            )

        result = bad_case_route_validation(
            backbone=model,
            data_root=args.data,
            cache_root=args.probe_cache,
            probe_dir=args.probe_dir,
            output_dir=args.output,
            device=device,
            attributes=_diagnostic_attributes(args),
            batch_size=args.batch_size,
            max_examples=args.max_examples,
            pair_limit=args.pair_limit,
            backbone_checkpoint=args.checkpoint,
            progress=progress,
        )
        result["log_dir"] = str(log_dir) if log_dir is not None else None
        print(json.dumps(result, indent=2))


def command_probe_pipeline(args: argparse.Namespace) -> None:
    if args.log_dir is None:
        args.log_dir = str(Path(args.output) / args.stage / "operation_logs")
    with command_monitor(args, "probe_pipeline") as (logger, log_dir):
        _command_probe_pipeline(args, logger=logger, log_dir=log_dir)


def _command_probe_pipeline(args: argparse.Namespace, *, logger, log_dir: Path | None) -> None:
    """Run a gated smoke/pilot/formal stage over any number of local GPUs."""

    config = load_pipeline_config(args.pipeline_config)
    steps, jobs, required_stage = jobs_for_stage(config, args.stage)
    runtime_config = ProbeRuntimeConfig.from_config(config).with_overrides(
        p_batch_size=args.p_batch_size,
        q_batch_size=args.q_batch_size,
        p_validation_batch_size=args.p_validation_batch_size,
        q_validation_batch_size=args.q_validation_batch_size,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        heartbeat_seconds=args.heartbeat_seconds,
        log_interval_steps=args.log_interval,
        evaluate_train=args.evaluate_train,
    )
    devices = resolve_devices(args.devices, args.num_gpus)
    cache_status = validate_probe_cache(
        args.probe_cache,
        args.data,
        include_missing_classes=False,
    )
    if args.require_coverage and not cache_status["coverage_complete"]:
        raise SystemExit("probe cache does not cover every validation class in the train split")

    output_root = Path(args.output)
    stage_root = output_root / args.stage
    pipeline_path = stage_root / "pipeline.json"
    probe_runtime = runtime_config.as_dict()
    identity = build_pipeline_identity(
        stage=args.stage,
        steps=steps,
        jobs=jobs,
        seed=args.seed,
        data=args.data,
        cache=args.probe_cache,
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        runtime=probe_runtime,
    )
    requested_checkpoint = str(identity["checkpoint"])
    requested_data = str(identity["data"])
    reuse_existing = pipeline_path.is_file()
    previous = None
    if reuse_existing:
        existing_stage = json.loads(pipeline_path.read_text(encoding="utf-8"))
        try:
            require_matching_identity(existing_stage, identity, label=str(pipeline_path))
        except ValueError as exc:
            raise SystemExit(f"{exc}; use a new output directory") from exc
    if required_stage and not args.ignore_prerequisite:
        prerequisite = output_root / required_stage / "pipeline.json"
        if not prerequisite.is_file():
            raise SystemExit(
                f"stage {args.stage} requires completed stage {required_stage}: {prerequisite}"
            )
        previous = json.loads(prerequisite.read_text(encoding="utf-8"))
        if previous.get("status") != "completed":
            raise SystemExit(f"required stage {required_stage} is not completed")
        required_steps, required_jobs, _ = jobs_for_stage(config, required_stage)
        prerequisite_identity = {
            **common_pipeline_identity(identity),
            "stage": required_stage,
            "steps": required_steps,
            "jobs": [job.key for job in required_jobs],
        }
        try:
            require_matching_identity(
                previous,
                prerequisite_identity,
                label=f"required stage {required_stage}",
            )
        except ValueError as exc:
            raise SystemExit(f"{exc}; rerun the prerequisite in a new output directory") from exc

    gate_result = None
    if not args.skip_gate:
        gate_path = output_root / "pretrain_gate.json"
        if gate_path.is_file() and not args.force_gate:
            candidate = json.loads(gate_path.read_text(encoding="utf-8"))
            if reusable_cloze_gate(candidate, identity):
                gate_result = candidate
        if gate_result is None:
            gate_cfg = config.get("gate", {})
            device = torch.device(devices[0])
            model = load_model(args.model_config, args.checkpoint, device)
            gate_result = evaluate_progressive_biography_cloze(
                model,
                args.data,
                device=device,
                max_biographies=int(gate_cfg.get("examples", 10_000)),
                batch_size=int(gate_cfg.get("batch_size", 8)),
                max_new_tokens=int(gate_cfg.get("max_new_tokens", 16)),
                sample_biographies=int(gate_cfg.get("sample_biographies", 12)),
                logger=logger,
                log_interval=runtime_config.log_interval_steps,
            )
            gate_result["checkpoint"] = requested_checkpoint
            gate_result["identity"] = common_pipeline_identity(identity)
            write_json_atomic(gate_path, gate_result)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        threshold = float(
            args.gate_threshold
            if args.gate_threshold is not None
            else config.get("gate", {}).get("threshold", 0.9)
        )
        gate_accuracy = float(gate_result["micro_field_accuracy"])
        if gate_accuracy < threshold:
            raise SystemExit(
                f"pretrain cloze gate failed: micro_field_accuracy={gate_accuracy:.4f} "
                f"< threshold={threshold:.4f}"
            )

    script = Path(__file__).resolve()
    common = {
        "script": script,
        "data": Path(args.data).resolve(),
        "cache": Path(args.probe_cache).resolve(),
        "model_config": Path(args.model_config).resolve(),
        "checkpoint": Path(args.checkpoint).resolve(),
        "output_dir": stage_root.resolve(),
        "quiet": args.quiet_workers,
        "log_interval": runtime_config.log_interval_steps,
        "tensorboard": args.tensorboard,
    }
    base_state = {
        "stage": args.stage,
        "steps": steps,
        "devices": devices,
        "checkpoint": requested_checkpoint,
        "data": requested_data,
        "jobs": [job.key for job in jobs],
        "identity": identity,
        "monitoring_log_dir": str(log_dir) if log_dir is not None else None,
        "runtime": probe_runtime,
    }
    state = ProbePipelineState(
        pipeline_path=pipeline_path,
        events_path=stage_root / "pipeline_events.jsonl",
        base_state=base_state,
        jobs=jobs,
        logger=logger,
        phase_duration_estimates=estimate_phase_durations(
            previous,
            jobs,
            steps,
            device_count=len(devices),
        ),
    )
    state.write("running")

    training = schedule_jobs(
        jobs,
        devices,
        probe_train_command_builder(
            **common,
            steps=steps,
            seed=args.seed,
            batch_sizes=probe_runtime["training_batch_sizes"],
            validation_batch_sizes=probe_runtime["validation_batch_sizes"],
            checkpoint_interval_steps=runtime_config.checkpoint_interval_steps,
            evaluate_train=runtime_config.evaluate_train,
            checkpoint_model_sha256=str(identity["checkpoint_model_sha256"]),
        ),
        on_event=state.monitor_phase("training"),
        heartbeat_seconds=runtime_config.heartbeat_seconds,
        reuse_existing=reuse_existing,
    )
    if any(item["status"] == "failed" for item in training):
        state.write("failed", training=training)
        raise SystemExit("one or more probe training jobs failed; inspect stage logs")

    validation = schedule_jobs(
        jobs,
        devices,
        probe_validation_command_builder(
            **common,
            validation_batch_sizes=probe_runtime["validation_batch_sizes"],
            checkpoint_model_sha256=str(identity["checkpoint_model_sha256"]),
        ),
        on_event=state.monitor_phase("validation", extra_state={"training": training}),
        heartbeat_seconds=runtime_config.heartbeat_seconds,
        reuse_existing=reuse_existing,
    )
    if any(item["status"] == "failed" for item in validation):
        state.write("failed", training=training, validation=validation)
        raise SystemExit("one or more probe validation jobs failed; inspect stage logs")
    summary = summarize_probe_results(
        {args.stage: stage_root / "validation"},
        stage_root / "summary",
        expected_jobs=jobs,
    )
    final_fields = {
        "cache": cache_status,
        "gate": gate_result,
        "training": training,
        "validation": validation,
        "summary_rows": len(summary["rows"]),
    }
    state.write("completed", **final_fields)
    print(json.dumps({"status": "completed", **base_state, **final_fields}))


def command_summarize_probes(args: argparse.Namespace) -> None:
    named = {}
    for value in args.run:
        if "=" not in value:
            raise SystemExit("--run must use NAME=VALIDATION_DIR")
        name, path = value.split("=", 1)
        if not name or not path:
            raise SystemExit("--run must use NAME=VALIDATION_DIR")
        named[name] = Path(path)
    result = summarize_probe_results(named, args.output)
    print(json.dumps({"runs": list(named), "rows": len(result["rows"])}))


def command_report_formal_study(args: argparse.Namespace) -> None:
    """Audit matched formal runs and render the canonical comparison artifacts."""

    result = build_formal_report_artifacts(
        single_root=args.single,
        multi_root=args.multi5_permute,
        single_cloze=args.single_cloze,
        multi_cloze=args.multi5_permute_cloze,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(Path(args.output).resolve()),
                "headline_metrics": result["headline_metrics"],
            },
            indent=2,
        )
    )


def command_report_probe_diagnostics(args: argparse.Namespace) -> None:
    """Audit both completed diagnostics and render their canonical report artifacts."""

    result = build_diagnostic_report_artifacts(
        single_formal_root=args.single_formal,
        multi_formal_root=args.multi5_permute_formal,
        diagnostics_root=args.diagnostics,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(Path(args.output).resolve()),
                "oracle_headline": result["oracle_headline"],
                "route_headline": result["route_headline"],
            },
            indent=2,
        )
    )


def command_audit_synbios_repository(args: argparse.Namespace) -> None:
    """Validate the canonical data-to-report graph and emit audit catalogs."""

    result = build_repository_audit(
        repo_root=args.repo_root,
        output_dir=args.output,
    )
    print(json.dumps(result, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    """Evaluate teacher-forced accuracy only on biography attribute tokens."""

    device = torch.device(args.device)
    with command_monitor(args, "evaluate") as (logger, log_dir):
        model = load_model(args.model_config, args.checkpoint, device, logger=logger)
        result = evaluate_attribute_tokens(
            model,
            args.data,
            device=device,
            max_biographies=args.examples,
            batch_size=args.batch_size,
            logger=logger,
            log_interval=args.log_interval,
        )
        result.update(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "provenance": collect_provenance(ROOT),
                "log_dir": str(log_dir) if log_dir is not None else None,
            }
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result))


def command_cloze_evaluate(args: argparse.Namespace) -> None:
    """Progressively fill the six removed facts in each original biography."""

    device = torch.device(args.device)
    with command_monitor(args, "cloze_evaluate") as (logger, log_dir):
        model = load_model(args.model_config, args.checkpoint, device, logger=logger)
        result = evaluate_progressive_biography_cloze(
            model,
            args.data,
            device=device,
            start_index=args.start_index,
            max_biographies=args.examples,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            sample_biographies=args.sample_biographies,
            logger=logger,
            log_interval=args.log_interval,
        )
        result.update(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "provenance": collect_provenance(ROOT),
                "log_dir": str(log_dir) if log_dir is not None else None,
            }
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result))


def command_summarize_cloze(args: argparse.Namespace) -> None:
    """Merge disjoint progressive-cloze result shards."""

    result = summarize_progressive_cloze_results(args.run)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"biographies": result["biographies"], "fields": result["fields"]}))


def build_parser() -> argparse.ArgumentParser:
    def add_monitoring_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--log-dir")
        command.add_argument("--log-interval", type=int, default=10)
        command.add_argument(
            "--tensorboard",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        command.add_argument("--quiet", action="store_true")

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    prepare.add_argument(
        "--variant",
        default="single",
        choices=(
            "single",
            "single+fullname",
            "single+permute1",
            "single+permute5",
            "multi2",
            "multi5",
            "multi2+permute",
            "multi5+permute",
            "multi5+permute+fullname",
        ),
    )
    prepare.add_argument("--num-people", type=int, default=100_000)
    prepare.add_argument("--seed", type=int, default=1337)
    prepare.add_argument("--max-shard-tokens", type=int, default=10_000_000)
    add_monitoring_arguments(prepare)
    prepare.set_defaults(func=command_prepare)

    cache = commands.add_parser("cache-probes")
    cache.add_argument("--data", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--force", action="store_true")
    cache.add_argument("--require-coverage", action="store_true")
    cache.set_defaults(func=command_cache_probes)

    validate_cache = commands.add_parser("validate-probe-cache")
    validate_cache.add_argument("--probe-cache", required=True)
    validate_cache.add_argument("--data")
    validate_cache.set_defaults(func=command_validate_cache)

    for name, function in (
        ("probe", command_probe),
        ("analyze", command_analyze),
        ("evaluate", command_evaluate),
    ):
        command = commands.add_parser(name)
        command.add_argument("--data", required=True)
        command.add_argument("--model-config", required=True)
        command.add_argument("--checkpoint", required=True)
        if name != "evaluate":
            command.add_argument("--attribute", choices=ATTRIBUTES, required=True)
            command.add_argument("--target", choices=("first", "whole"), default="first")
        command.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        command.add_argument("--output", required=True)
        add_monitoring_arguments(command)
        if name == "probe":
            command.add_argument("--probe-cache")
            command.add_argument("--kind", choices=("p", "q"), required=True)
            command.add_argument("--rank", type=int)
            command.add_argument("--checkpoint-model-sha256", help=argparse.SUPPRESS)
            command.add_argument("--batch-size", type=int)
            command.add_argument("--evaluation-batch-size", type=int)
            command.add_argument("--steps", type=int, default=30_000)
            command.add_argument("--seed", type=int, default=1337)
            command.add_argument("--recovery-checkpoint")
            command.add_argument("--checkpoint-interval-steps", type=int)
            command.add_argument(
                "--resume-probe",
                action=argparse.BooleanOptionalAction,
                default=True,
            )
            command.add_argument("--evaluate-train", action="store_true")
            command.add_argument("--skip-final-validation", action="store_true")
        elif name == "analyze":
            command.add_argument("--probe-cache")
            command.add_argument("--examples", type=int, default=1024)
        else:
            command.add_argument("--examples", type=int, default=10_000)
            command.add_argument("--batch-size", type=int, default=8)
        command.set_defaults(func=function)

    validate_probe = commands.add_parser("validate-probe")
    validate_probe.add_argument("--data", required=True)
    validate_probe.add_argument("--probe-cache")
    validate_probe.add_argument("--model-config", required=True)
    validate_probe.add_argument("--checkpoint", required=True)
    validate_probe.add_argument("--probe-checkpoint", required=True)
    validate_probe.add_argument("--batch-size", type=int)
    validate_probe.add_argument("--allow-checkpoint-mismatch", action="store_true")
    validate_probe.add_argument("--checkpoint-model-sha256", help=argparse.SUPPRESS)
    validate_probe.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    validate_probe.add_argument("--output", required=True)
    add_monitoring_arguments(validate_probe)
    validate_probe.set_defaults(func=command_validate_probe)

    for name, function in (
        ("validate-probe-oracle-first-token", command_validate_probe_oracle_first_token),
        ("validate-probe-bad-case-routes", command_validate_probe_bad_case_routes),
    ):
        diagnostic = commands.add_parser(name)
        diagnostic.add_argument("--data", required=True)
        diagnostic.add_argument("--probe-cache", required=True)
        diagnostic.add_argument("--probe-dir", required=True)
        diagnostic.add_argument("--model-config", required=True)
        diagnostic.add_argument("--checkpoint", required=True)
        diagnostic.add_argument(
            "--attribute",
            action="append",
            choices=WHOLE_ATTRIBUTES,
            help="repeat to select attributes; defaults to all five whole-value tasks",
        )
        diagnostic.add_argument("--batch-size", type=int, default=512)
        diagnostic.add_argument("--max-examples", type=int)
        diagnostic.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
        diagnostic.add_argument("--output", required=True)
        if name == "validate-probe-bad-case-routes":
            diagnostic.add_argument("--pair-limit", type=int, default=2000)
        add_monitoring_arguments(diagnostic)
        diagnostic.set_defaults(func=function)

    benchmark_probe = commands.add_parser("benchmark-probe-batches")
    benchmark_probe.add_argument("--data", required=True)
    benchmark_probe.add_argument("--probe-cache", required=True)
    benchmark_probe.add_argument("--model-config", required=True)
    benchmark_probe.add_argument("--checkpoint", required=True)
    benchmark_probe.add_argument("--kind", choices=("p", "q"), required=True)
    benchmark_probe.add_argument(
        "--mode", choices=("training", "validation"), default="training"
    )
    benchmark_probe.add_argument("--attribute", choices=ATTRIBUTES, default="university")
    benchmark_probe.add_argument("--target", choices=("first", "whole"), default="whole")
    benchmark_probe.add_argument("--rank", type=int)
    benchmark_probe.add_argument("--batch-sizes", required=True)
    benchmark_probe.add_argument("--warmup-steps", type=int, default=3)
    benchmark_probe.add_argument("--measure-steps", type=int, default=10)
    benchmark_probe.add_argument("--memory-limit-percent", type=float, default=92.0)
    benchmark_probe.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    benchmark_probe.add_argument("--output", required=True)
    add_monitoring_arguments(benchmark_probe)
    benchmark_probe.set_defaults(func=command_probe_benchmark)

    summarize_benchmarks = commands.add_parser("summarize-probe-benchmarks")
    summarize_benchmarks.add_argument("--run", action="append", required=True)
    summarize_benchmarks.add_argument("--output", required=True)
    summarize_benchmarks.add_argument("--env-output")
    summarize_benchmarks.add_argument("--require-complete-search", action="store_true")
    summarize_benchmarks.set_defaults(func=command_summarize_probe_benchmarks)

    cloze = commands.add_parser("cloze-evaluate")
    cloze.add_argument("--data", required=True)
    cloze.add_argument("--model-config", required=True)
    cloze.add_argument("--checkpoint", required=True)
    cloze.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cloze.add_argument("--output", required=True)
    cloze.add_argument("--examples", type=int, default=1_000)
    cloze.add_argument("--start-index", type=int, default=0)
    cloze.add_argument("--batch-size", type=int, default=16)
    cloze.add_argument("--max-new-tokens", type=int, default=16)
    cloze.add_argument("--sample-biographies", type=int, default=12)
    add_monitoring_arguments(cloze)
    cloze.set_defaults(func=command_cloze_evaluate)

    summarize_cloze = commands.add_parser("summarize-cloze")
    summarize_cloze.add_argument("--run", action="append", required=True)
    summarize_cloze.add_argument("--output", required=True)
    summarize_cloze.set_defaults(func=command_summarize_cloze)

    pipeline = commands.add_parser("probe-pipeline")
    pipeline.add_argument("--data", required=True)
    pipeline.add_argument("--probe-cache", required=True)
    pipeline.add_argument("--model-config", required=True)
    pipeline.add_argument("--checkpoint", required=True)
    pipeline.add_argument("--output", required=True)
    pipeline.add_argument(
        "--pipeline-config",
        default=str(ROOT / "configs" / "synbios_moe" / "probe_pipeline.yaml"),
    )
    pipeline.add_argument("--stage", choices=("smoke", "pilot", "formal"), required=True)
    pipeline.add_argument("--devices", default="auto")
    pipeline.add_argument("--num-gpus", type=int)
    pipeline.add_argument("--seed", type=int, default=1337)
    pipeline.add_argument("--gate-threshold", type=float)
    pipeline.add_argument("--skip-gate", action="store_true")
    pipeline.add_argument("--force-gate", action="store_true")
    pipeline.add_argument("--ignore-prerequisite", action="store_true")
    pipeline.add_argument("--require-coverage", action="store_true")
    pipeline.add_argument("--quiet-workers", action="store_true")
    pipeline.add_argument("--heartbeat-seconds", type=float)
    pipeline.add_argument("--p-batch-size", type=int)
    pipeline.add_argument("--q-batch-size", type=int)
    pipeline.add_argument("--p-validation-batch-size", type=int)
    pipeline.add_argument("--q-validation-batch-size", type=int)
    pipeline.add_argument("--checkpoint-interval-steps", type=int)
    pipeline.add_argument(
        "--evaluate-train",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    add_monitoring_arguments(pipeline)
    pipeline.set_defaults(func=command_probe_pipeline, log_interval=None)

    summarize = commands.add_parser("summarize-probes")
    summarize.add_argument("--run", action="append", required=True)
    summarize.add_argument("--output", required=True)
    summarize.set_defaults(func=command_summarize_probes)

    formal_report = commands.add_parser("report-formal-study")
    formal_report.add_argument("--single", required=True, help="single formal stage directory")
    formal_report.add_argument(
        "--multi5-permute",
        required=True,
        help="multi5_permute formal stage directory",
    )
    formal_report.add_argument("--single-cloze", required=True)
    formal_report.add_argument("--multi5-permute-cloze", required=True)
    formal_report.add_argument("--output", required=True)
    formal_report.set_defaults(func=command_report_formal_study)

    diagnostic_report = commands.add_parser("report-probe-diagnostics")
    diagnostic_report.add_argument("--single-formal", required=True)
    diagnostic_report.add_argument("--multi5-permute-formal", required=True)
    diagnostic_report.add_argument("--diagnostics", required=True)
    diagnostic_report.add_argument("--output", required=True)
    diagnostic_report.set_defaults(func=command_report_probe_diagnostics)

    repository_audit = commands.add_parser("audit-synbios-repository")
    repository_audit.add_argument("--repo-root", default=str(ROOT))
    repository_audit.add_argument("--output", required=True)
    repository_audit.set_defaults(func=command_audit_synbios_repository)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if (
        getattr(arguments, "attribute", None) == "birth_date"
        and getattr(arguments, "target", None) == "whole"
    ):
        raise SystemExit("whole birth-date classification is not part of the paper protocol")
    arguments.func(arguments)
    (require_matching_identity,)
