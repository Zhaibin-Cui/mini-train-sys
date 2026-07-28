"""Command-line schema for the SynBioS experiment workflow."""

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path

from experiments.synbios_moe.pretraining.dataset import ATTRIBUTES, WHOLE_ATTRIBUTES


CommandHandler = Callable[[argparse.Namespace], None]
PUBLIC_COMMANDS = (
    "prepare",
    "cache-probes",
    "validate-probe-cache",
    "probe",
    "analyze",
    "evaluate",
    "validate-probe",
    "train-ground-truth-first-whole",
    "summarize-ground-truth-first-whole",
    "benchmark-ground-truth-first-whole-batches",
    "validate-probe-oracle-first-token",
    "validate-probe-bad-case-routes",
    "benchmark-probe-batches",
    "summarize-probe-benchmarks",
    "cloze-evaluate",
    "summarize-cloze",
    "probe-pipeline",
    "summarize-probes",
    "report-formal-study",
    "report-probe-diagnostics",
    "audit-synbios-repository",
)


def _monitoring_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--log-dir")
    command.add_argument("--log-interval", type=int, default=10)
    command.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    command.add_argument("--quiet", action="store_true")


def _bind(
    command: argparse.ArgumentParser,
    handlers: Mapping[str, CommandHandler],
    name: str,
    **defaults: object,
) -> None:
    command.set_defaults(func=handlers[name], **defaults)


def _add_data_commands(
    commands: argparse._SubParsersAction,
    handlers: Mapping[str, CommandHandler],
) -> None:
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
    _monitoring_arguments(prepare)
    _bind(prepare, handlers, "prepare")

    cache = commands.add_parser("cache-probes")
    cache.add_argument("--data", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--force", action="store_true")
    cache.add_argument("--require-coverage", action="store_true")
    _bind(cache, handlers, "cache-probes")

    validate_cache = commands.add_parser("validate-probe-cache")
    validate_cache.add_argument("--probe-cache", required=True)
    validate_cache.add_argument("--data")
    _bind(validate_cache, handlers, "validate-probe-cache")


def _add_core_probe_commands(
    commands: argparse._SubParsersAction,
    handlers: Mapping[str, CommandHandler],
    *,
    default_device: str,
) -> None:
    for name in ("probe", "analyze", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--data", required=True)
        command.add_argument("--model-config", required=True)
        command.add_argument("--checkpoint", required=True)
        if name != "evaluate":
            command.add_argument("--attribute", choices=ATTRIBUTES, required=True)
            command.add_argument("--target", choices=("first", "whole"), default="first")
        command.add_argument("--device", default=default_device)
        command.add_argument("--output", required=True)
        _monitoring_arguments(command)
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
        _bind(command, handlers, name)

    validate_probe = commands.add_parser("validate-probe")
    validate_probe.add_argument("--data", required=True)
    validate_probe.add_argument("--probe-cache")
    validate_probe.add_argument("--model-config", required=True)
    validate_probe.add_argument("--checkpoint", required=True)
    validate_probe.add_argument("--probe-checkpoint", required=True)
    validate_probe.add_argument("--batch-size", type=int)
    validate_probe.add_argument("--allow-checkpoint-mismatch", action="store_true")
    validate_probe.add_argument("--checkpoint-model-sha256", help=argparse.SUPPRESS)
    validate_probe.add_argument("--device", default=default_device)
    validate_probe.add_argument("--output", required=True)
    _monitoring_arguments(validate_probe)
    _bind(validate_probe, handlers, "validate-probe")


def _add_first_token_commands(
    commands: argparse._SubParsersAction,
    handlers: Mapping[str, CommandHandler],
    *,
    default_device: str,
) -> None:
    train = commands.add_parser("train-ground-truth-first-whole")
    train.add_argument("--data", required=True)
    train.add_argument("--probe-cache", required=True)
    train.add_argument("--probe-dir", required=True)
    train.add_argument("--model-config", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--attribute", choices=WHOLE_ATTRIBUTES, required=True)
    train.add_argument("--steps", type=int, default=3_000)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--evaluation-batch-size", type=int)
    train.add_argument(
        "--max-validation-examples",
        type=int,
        help="deterministic prefix limit for smoke tests; formal runs omit this option",
    )
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--recovery-checkpoint")
    train.add_argument("--checkpoint-interval-steps", type=int, default=1_000)
    train.add_argument(
        "--resume-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument("--evaluate-train", action="store_true")
    train.add_argument("--device", default=default_device)
    train.add_argument("--output", required=True)
    _monitoring_arguments(train)
    _bind(train, handlers, "train-ground-truth-first-whole")

    summarize = commands.add_parser("summarize-ground-truth-first-whole")
    summarize.add_argument("--run", required=True)
    _bind(summarize, handlers, "summarize-ground-truth-first-whole")

    benchmark = commands.add_parser("benchmark-ground-truth-first-whole-batches")
    benchmark.add_argument("--data", required=True)
    benchmark.add_argument("--probe-cache", required=True)
    benchmark.add_argument("--probe-dir", required=True)
    benchmark.add_argument("--model-config", required=True)
    benchmark.add_argument("--checkpoint", required=True)
    benchmark.add_argument("--attribute", choices=WHOLE_ATTRIBUTES, default="university")
    benchmark.add_argument("--mode", choices=("training", "validation"), default="training")
    benchmark.add_argument("--batch-sizes", required=True)
    benchmark.add_argument("--warmup-steps", type=int, default=3)
    benchmark.add_argument("--measure-steps", type=int, default=10)
    benchmark.add_argument("--memory-limit-percent", type=float, default=92.0)
    benchmark.add_argument("--device", default=default_device)
    benchmark.add_argument("--output", required=True)
    _monitoring_arguments(benchmark)
    _bind(benchmark, handlers, "benchmark-ground-truth-first-whole-batches")

    for name in (
        "validate-probe-oracle-first-token",
        "validate-probe-bad-case-routes",
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
        diagnostic.add_argument("--device", default=default_device)
        diagnostic.add_argument("--output", required=True)
        if name == "validate-probe-bad-case-routes":
            diagnostic.add_argument("--pair-limit", type=int, default=2000)
        _monitoring_arguments(diagnostic)
        _bind(diagnostic, handlers, name)


def _add_benchmark_and_cloze_commands(
    commands: argparse._SubParsersAction,
    handlers: Mapping[str, CommandHandler],
    *,
    default_device: str,
) -> None:
    benchmark = commands.add_parser("benchmark-probe-batches")
    benchmark.add_argument("--data", required=True)
    benchmark.add_argument("--probe-cache", required=True)
    benchmark.add_argument("--model-config", required=True)
    benchmark.add_argument("--checkpoint", required=True)
    benchmark.add_argument("--kind", choices=("p", "q"), required=True)
    benchmark.add_argument("--mode", choices=("training", "validation"), default="training")
    benchmark.add_argument("--attribute", choices=ATTRIBUTES, default="university")
    benchmark.add_argument("--target", choices=("first", "whole"), default="whole")
    benchmark.add_argument("--rank", type=int)
    benchmark.add_argument("--batch-sizes", required=True)
    benchmark.add_argument("--warmup-steps", type=int, default=3)
    benchmark.add_argument("--measure-steps", type=int, default=10)
    benchmark.add_argument("--memory-limit-percent", type=float, default=92.0)
    benchmark.add_argument("--device", default=default_device)
    benchmark.add_argument("--output", required=True)
    _monitoring_arguments(benchmark)
    _bind(benchmark, handlers, "benchmark-probe-batches")

    summarize_benchmarks = commands.add_parser("summarize-probe-benchmarks")
    summarize_benchmarks.add_argument("--run", action="append", required=True)
    summarize_benchmarks.add_argument("--output", required=True)
    summarize_benchmarks.add_argument("--env-output")
    summarize_benchmarks.add_argument("--require-complete-search", action="store_true")
    _bind(summarize_benchmarks, handlers, "summarize-probe-benchmarks")

    cloze = commands.add_parser("cloze-evaluate")
    cloze.add_argument("--data", required=True)
    cloze.add_argument("--model-config", required=True)
    cloze.add_argument("--checkpoint", required=True)
    cloze.add_argument("--device", default=default_device)
    cloze.add_argument("--output", required=True)
    cloze.add_argument("--examples", type=int, default=1_000)
    cloze.add_argument("--start-index", type=int, default=0)
    cloze.add_argument("--batch-size", type=int, default=16)
    cloze.add_argument("--max-new-tokens", type=int, default=16)
    cloze.add_argument("--sample-biographies", type=int, default=12)
    _monitoring_arguments(cloze)
    _bind(cloze, handlers, "cloze-evaluate")

    summarize_cloze = commands.add_parser("summarize-cloze")
    summarize_cloze.add_argument("--run", action="append", required=True)
    summarize_cloze.add_argument("--output", required=True)
    _bind(summarize_cloze, handlers, "summarize-cloze")


def _add_pipeline_and_report_commands(
    commands: argparse._SubParsersAction,
    handlers: Mapping[str, CommandHandler],
    *,
    project_root: Path,
) -> None:
    pipeline = commands.add_parser("probe-pipeline")
    pipeline.add_argument("--data", required=True)
    pipeline.add_argument("--probe-cache", required=True)
    pipeline.add_argument("--model-config", required=True)
    pipeline.add_argument("--checkpoint", required=True)
    pipeline.add_argument("--output", required=True)
    pipeline.add_argument(
        "--pipeline-config",
        default=str(project_root / "configs" / "synbios_moe" / "probe_pipeline.yaml"),
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
    _monitoring_arguments(pipeline)
    _bind(pipeline, handlers, "probe-pipeline", log_interval=None)

    summarize = commands.add_parser("summarize-probes")
    summarize.add_argument("--run", action="append", required=True)
    summarize.add_argument("--output", required=True)
    _bind(summarize, handlers, "summarize-probes")

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
    _bind(formal_report, handlers, "report-formal-study")

    diagnostic_report = commands.add_parser("report-probe-diagnostics")
    diagnostic_report.add_argument("--single-formal", required=True)
    diagnostic_report.add_argument("--multi5-permute-formal", required=True)
    diagnostic_report.add_argument("--diagnostics", required=True)
    diagnostic_report.add_argument("--output", required=True)
    _bind(diagnostic_report, handlers, "report-probe-diagnostics")

    repository_audit = commands.add_parser("audit-synbios-repository")
    repository_audit.add_argument("--repo-root", default=str(project_root))
    repository_audit.add_argument("--output", required=True)
    _bind(repository_audit, handlers, "audit-synbios-repository")


def build_parser(
    *,
    project_root: Path,
    handlers: Mapping[str, CommandHandler],
    default_device: str,
) -> argparse.ArgumentParser:
    """Build the public CLI without importing command implementations."""

    provided = set(handlers)
    expected = set(PUBLIC_COMMANDS)
    if provided != expected:
        raise ValueError(
            "command handler mismatch: "
            f"missing={sorted(expected - provided)}, extra={sorted(provided - expected)}"
        )
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    _add_data_commands(commands, handlers)
    _add_core_probe_commands(commands, handlers, default_device=default_device)
    _add_first_token_commands(commands, handlers, default_device=default_device)
    _add_benchmark_and_cloze_commands(commands, handlers, default_device=default_device)
    _add_pipeline_and_report_commands(commands, handlers, project_root=project_root)
    if tuple(commands.choices) != PUBLIC_COMMANDS:
        raise AssertionError("registered SynBioS command order does not match PUBLIC_COMMANDS")
    return parser
