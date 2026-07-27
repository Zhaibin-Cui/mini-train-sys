"""Audit the true-t1 rank-matched fresh whole-probe protocol on real artifacts."""

# ruff: noqa: E402 -- direct script execution needs the repository root on sys.path.


import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.synbios_moe.probe_diagnostics import (
    WHOLE_ATTRIBUTES,
    build_ground_truth_first_input,
    prepare_ground_truth_first_whole_data,
)
from experiments.synbios_moe.probes import AttributeProbe
from scripts.synbios_moe import load_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    backbone = load_model(args.model_config, args.checkpoint, device)
    tasks = []
    for attribute in WHOLE_ATTRIBUTES:
        prepared = prepare_ground_truth_first_whole_data(
            backbone=backbone,
            cache_root=args.probe_cache,
            probe_dir=args.probe_dir,
            attribute=attribute,
            backbone_checkpoint=args.checkpoint,
        )
        token_ids = [token_id for token_id, _ in prepared.token_entries]
        checked = 0
        for split_name, first_data, whole_data, rebuilt_data in (
            (
                "train",
                prepared.first_train,
                prepared.whole_train,
                prepared.train_data,
            ),
            (
                "validation",
                prepared.validation_data.first_data,
                prepared.validation_data.whole_data,
                prepared.validation_data,
            ),
        ):
            indices = sorted({0, len(first_data) // 2, len(first_data) - 1})
            positions = range(6)
            for source_index in indices:
                first_item = first_data[source_index]
                whole_item = whole_data[source_index]
                if (
                    first_item.input_ids != whole_item.input_ids
                    or first_item.positions != whole_item.positions
                ):
                    raise ValueError(
                        f"p/{attribute}/{split_name} source inputs are not aligned"
                    )
                token_id = token_ids[first_item.label]
                for position in positions:
                    expanded_index = source_index * len(tuple(positions)) + position
                    actual = rebuilt_data[expanded_index]
                    expected = build_ground_truth_first_input(
                        item=first_item,
                        source_position=position,
                        token_id=token_id,
                    )
                    if (
                        actual.input_ids != expected.input_ids
                        or actual.positions != expected.positions
                        or actual.label != whole_item.label
                    ):
                        raise ValueError(
                            f"p/{attribute}/{split_name} rebuilt item mismatch"
                        )
                    if actual.input_ids[actual.positions[0]] != token_id:
                        raise ValueError(
                            f"p/{attribute}/{split_name} readout token mismatch"
                        )
                    checked += 1

        probe = AttributeProbe(
            backbone,
            len(prepared.whole_train.class_names),
            rank=prepared.rank,
            kind="p",
        )
        trainable = {
            name: tuple(parameter.shape)
            for name, parameter in probe.named_parameters()
            if parameter.requires_grad
        }
        expected_prefixes = (
            "delta.a.weight",
            "delta.b.weight",
            "normalizer.weight",
            "normalizer.bias",
            "classifier.weight",
            "classifier.bias",
        )
        if tuple(trainable) != expected_prefixes:
            raise ValueError(
                f"p/{attribute} trainable parameters mismatch: {trainable}"
            )
        if any(parameter.requires_grad for parameter in backbone.parameters()):
            raise ValueError("fresh probe did not freeze every backbone parameter")
        tasks.append(
            {
                "kind": "p",
                "attribute": attribute,
                "rank": prepared.rank,
                "classes": len(prepared.whole_train.class_names),
                "train_sources": len(prepared.first_train),
                "validation_sources": len(prepared.validation_data.first_data),
                "expanded_train_examples": len(prepared.train_data),
                "expanded_validation_examples": len(prepared.validation_data),
                "sampled_items_checked": checked,
                "reference_whole_probe": str(
                    prepared.reference_whole_checkpoint.resolve()
                ),
                "reference_whole_probe_sha256": _sha256(
                    prepared.reference_whole_checkpoint
                ),
                "trainable_parameters": trainable,
                "input_protocol": "P prefix + true_t1, read true_t1",
                "passed": True,
            }
        )
    result = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "ground_truth_first_whole_rank_matched_v1",
        "quality_gate_passed": len(tasks) == 5 and all(task["passed"] for task in tasks),
        "data": str(Path(args.data).resolve()),
        "probe_cache": str(Path(args.probe_cache).resolve()),
        "probe_cache_manifest_sha256": _sha256(Path(args.probe_cache) / "manifest.json"),
        "probe_dir": str(Path(args.probe_dir).resolve()),
        "backbone_checkpoint": str(Path(args.checkpoint).resolve()),
        "tasks": tasks,
    }
    _atomic_json(Path(args.output), result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--data", required=True)
    result.add_argument("--probe-cache", required=True)
    result.add_argument("--probe-dir", required=True)
    result.add_argument("--model-config", required=True)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--output", required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(audit(parser().parse_args()), indent=2))
