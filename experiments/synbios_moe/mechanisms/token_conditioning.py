"""Test first-token intervention and token-conditioned route branching.

The legacy diagnostics in this module never update model or probe parameters:

* ``oracle_first_token_validation`` measures whether inserting the ground-truth
  first attribute token before the Q readout EOS unlocks a trained Q-whole head.
* ``bad_case_route_validation`` finds Q-first-correct/Q-whole-wrong examples and
  tests whether examples sharing token 1 use similar expert paths before
  branching at token 2.

``train_ground_truth_first_whole_probe`` inserts the cached ground-truth first
token, reruns the frozen backbone, and trains a fresh whole-value probe with the
same low-rank input-delta architecture and rank as the original formal probe.
P reads the inserted token itself; Q reads the EOS immediately after it.

Raw per-example evidence, tidy aggregates, and plots are deliberately kept
separate so conclusions remain auditable.
"""


import csv
import hashlib
import heapq
import json
import random
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.torch_version import TorchVersion

from experiments.synbios_moe.mechanisms.routing import normalized_mutual_information
from experiments.synbios_moe.pretraining.dataset import ATTRIBUTES
from experiments.synbios_moe.probes.dataset import CachedProbeDataset
from experiments.synbios_moe.probes.model import (
    AttributeProbe,
    GPT2Codec,
    ProbeBatchItem,
    collate_probe,
    train_probe,
)
from minitrain.model.transformer import MiniTransformer
from minitrain.runtime.monitoring import ProgressReporter


WHOLE_ATTRIBUTES = tuple(attribute for attribute in ATTRIBUTES if attribute != "birth_date")


@dataclass(frozen=True)
class QPrediction:
    """Aligned first/whole prediction for one held-out person and attribute."""

    case_id: int
    profile_index: int
    person_id: str
    attribute: str
    input_ids: tuple[int, ...]
    true_first_id: int
    pred_first_id: int
    true_first_token: str
    pred_first_token: str
    true_whole_id: int
    pred_whole_id: int
    true_whole_value: str
    pred_whole_value: str
    whole_true_probability: float

    @property
    def first_correct(self) -> bool:
        return self.true_first_id == self.pred_first_id

    @property
    def whole_correct(self) -> bool:
        return self.true_whole_id == self.pred_whole_id


def _read_profiles(data_root: Path) -> list[dict[str, object]]:
    with (data_root / "profiles.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, object]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames or sorted({key for row in rows for key in row}))
    if not columns:
        raise ValueError(f"cannot infer CSV columns for empty output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_class_names(cache_root: str | Path, attribute: str, target: str) -> list[str]:
    """Resolve a cached task class mapping without reading raw dataset payloads."""

    root = Path(cache_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for task in manifest["tasks"]:
        if task["attribute"] == attribute and task["target"] == target:
            return [str(value) for value in task["class_names"]]
    raise ValueError(f"missing cache task {attribute}/{target}")


def _load_probe(
    path: Path,
    *,
    backbone: MiniTransformer,
    dataset: CachedProbeDataset,
    kind: str,
    attribute: str,
    target: str,
    backbone_checkpoint: str | Path | None,
    cache_manifest_sha256: str,
) -> AttributeProbe:
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("result")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} is missing probe result metadata")
    if kind not in {"p", "q"}:
        raise ValueError(f"invalid probe kind: {kind}")
    expected = {"kind": kind, "attribute": attribute, "target": target}
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"{path} metadata is {actual}, expected {expected}")
    if list(metadata.get("class_names", ())) != dataset.class_names:
        raise ValueError(f"{path} class mapping does not match the probe cache")
    if metadata.get("probe_cache_manifest_sha256") != cache_manifest_sha256:
        raise ValueError(f"{path} was trained with a different probe cache")
    if (
        backbone_checkpoint is not None
        and Path(str(metadata.get("checkpoint"))).resolve() != Path(backbone_checkpoint).resolve()
    ):
        raise ValueError(f"{path} was trained with a different backbone checkpoint")
    probe = AttributeProbe(
        backbone,
        len(dataset.class_names),
        rank=int(metadata["rank"]),
        kind=kind,
    )
    incompatible = probe.load_state_dict(payload["probe"], strict=False)
    if incompatible.unexpected_keys or any(
        not key.startswith("backbone.") for key in incompatible.missing_keys
    ):
        raise ValueError(f"incompatible probe state in {path}: {incompatible}")
    return probe


def _dataset_profile_index(dataset: CachedProbeDataset, index: int) -> int:
    sample = int(dataset.sample_indices[index])
    return int(dataset.profile_indices[sample])


def first_token_text(class_name: str, codec: GPT2Codec) -> tuple[int, str]:
    """Recover one cached first-token class as its exact leading-space token text."""

    try:
        token_id = int(class_name)
    except ValueError as exc:
        raise ValueError(f"first-token class is not a token ID: {class_name!r}") from exc
    if not 0 <= token_id < codec.vocab_size or token_id == codec.eos:
        raise ValueError(f"invalid first-token ID: {token_id}")
    text = codec.encoding.decode([token_id])
    if not text.startswith(" "):
        raise ValueError(
            f"first-token {token_id} decodes to {text!r}, expected an ASCII leading space"
        )
    if codec.encode(text) != [token_id]:
        raise ValueError(f"first-token {token_id} does not round-trip through its decoded text")
    return token_id, text


def build_ground_truth_first_input(
    *,
    item: ProbeBatchItem,
    source_position: int,
    token_id: int,
) -> ProbeBatchItem:
    """Append the ground-truth t1 to one P biography prefix and read that token."""

    if not 0 <= source_position < len(item.positions):
        raise IndexError("P source position is outside the cached item")
    prefix = item.input_ids[: item.positions[source_position] + 1]
    input_ids = [*prefix, int(token_id)]
    return ProbeBatchItem(input_ids, [len(input_ids) - 1], item.label)


class GroundTruthFirstWholeDataset(Dataset):
    """Whole labels paired with inputs rebuilt from cached ground-truth t1 labels."""

    def __init__(
        self,
        *,
        first_data: CachedProbeDataset,
        whole_data: CachedProbeDataset,
        token_ids_by_class: Sequence[int],
    ) -> None:
        if len(first_data) != len(whole_data):
            raise ValueError("first/whole datasets are not aligned")
        self.kind = "p"
        self.first_data = first_data
        self.whole_data = whole_data
        self.token_ids_by_class = tuple(int(value) for value in token_ids_by_class)
        self.positions_per_source = 6
        self.class_names = list(whole_data.class_names)

    def __len__(self) -> int:
        return len(self.first_data) * self.positions_per_source

    def source_position(self, index: int) -> int:
        return index % self.positions_per_source

    def longest_items(self, limit: int) -> list[ProbeBatchItem]:
        """Return exact rebuilt inputs with the largest sequence lengths."""

        if limit <= 0:
            raise ValueError("limit must be positive")

        def input_length(index: int) -> int:
            source_index, source_position = divmod(index, self.positions_per_source)
            sample = int(self.first_data.sample_indices[source_index])
            return int(self.first_data.positions[sample][source_position]) + 2

        selected = heapq.nlargest(min(limit, len(self)), range(len(self)), key=input_length)
        return [self[index] for index in selected]

    def __getitem__(self, index: int) -> ProbeBatchItem:
        source_index, source_position = divmod(index, self.positions_per_source)
        first_item = self.first_data[source_index]
        whole_item = self.whole_data[source_index]
        if (
            first_item.input_ids != whole_item.input_ids
            or first_item.positions != whole_item.positions
        ):
            raise ValueError("first and whole cached items are not input-aligned")
        ground_truth_class = int(first_item.label)
        if not 0 <= ground_truth_class < len(self.token_ids_by_class):
            raise ValueError("ground-truth first-token class is out of range")
        rebuilt = build_ground_truth_first_input(
            item=first_item,
            source_position=source_position,
            token_id=self.token_ids_by_class[ground_truth_class],
        )
        return ProbeBatchItem(rebuilt.input_ids, rebuilt.positions, whole_item.label)


@torch.no_grad()
def evaluate_ground_truth_first_whole_by_source_position(
    probe: AttributeProbe,
    dataset: GroundTruthFirstWholeDataset,
    *,
    device: torch.device,
    batch_size: int,
    max_examples: int | None = None,
    logger=None,
    log_interval: int | None = None,
) -> dict[str, object]:
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive or None")
    evaluation_data: Dataset = dataset
    if max_examples is not None:
        evaluation_data = Subset(dataset, range(min(max_examples, len(dataset))))
    loader = DataLoader(
        evaluation_data,
        batch_size=batch_size,
        collate_fn=collate_probe,
        pin_memory=device.type == "cuda",
    )
    probe.to(device).eval()
    progress = (
        ProgressReporter(
            "ground_truth_probe_validation",
            len(loader),
            logger,
            device,
            log_interval=max(1, min(log_interval or 10, len(loader))),
            unit="batch",
        )
        if logger is not None and len(loader) > 0
        else None
    )
    correct = [0] * dataset.positions_per_source
    total = [0] * dataset.positions_per_source
    offset = 0
    for batch_index, (input_ids, positions, labels) in enumerate(loader, start=1):
        logits = probe(
            input_ids.to(device, non_blocking=True),
            positions.to(device, non_blocking=True),
        )[:, 0]
        predictions = logits.argmax(-1).cpu()
        for local_index, (prediction, label) in enumerate(zip(predictions, labels)):
            source_position = dataset.source_position(offset + local_index)
            total[source_position] += 1
            correct[source_position] += int(prediction == label)
        offset += len(labels)
        if progress is not None:
            progress.update(
                batch_index,
                items=len(labels),
                tokens=input_ids.numel(),
                metrics={
                    "accuracy_running": sum(correct) / max(sum(total), 1),
                    "accuracy_by_position_running": [
                        hits / max(count, 1) for hits, count in zip(correct, total)
                    ],
                },
            )
    return {
        "accuracy": sum(correct) / max(sum(total), 1),
        "accuracy_by_position": [hits / max(count, 1) for hits, count in zip(correct, total)],
        "correct_by_position": correct,
        "total_by_position": total,
        "monitoring": progress.summary() if progress is not None else {},
    }


def train_ground_truth_first_whole_probe(
    *,
    backbone: MiniTransformer,
    cache_root: str | Path,
    probe_dir: str | Path,
    attribute: str,
    device: torch.device,
    batch_size: int,
    evaluation_batch_size: int,
    steps: int = 3_000,
    seed: int = 1337,
    backbone_checkpoint: str | Path | None = None,
    logger=None,
    log_interval: int | None = None,
    recovery_path: str | Path | None = None,
    checkpoint_interval_steps: int | None = None,
    resume: bool = True,
    evaluate_train: bool = False,
    max_validation_examples: int | None = None,
) -> tuple[AttributeProbe, dict[str, object]]:
    """Train a rank-matched whole probe after ground-truth-t1 insertion."""

    prepared = prepare_ground_truth_first_whole_data(
        backbone=backbone,
        cache_root=cache_root,
        probe_dir=probe_dir,
        attribute=attribute,
        backbone_checkpoint=backbone_checkpoint,
    )
    cache_root = Path(cache_root)
    first_train = prepared.first_train
    whole_train = prepared.whole_train
    train_data = prepared.train_data
    validation_data = prepared.validation_data
    token_entries = prepared.token_entries
    reference_whole_checkpoint = prepared.reference_whole_checkpoint
    cache_manifest_sha256 = prepared.cache_manifest_sha256
    rank = prepared.rank
    if max_validation_examples is not None and max_validation_examples <= 0:
        raise ValueError("max_validation_examples must be positive or None")
    validation_for_training: Dataset = validation_data
    if max_validation_examples is not None:
        validation_for_training = Subset(
            validation_data,
            range(min(max_validation_examples, len(validation_data))),
        )
    probe = AttributeProbe(
        backbone,
        len(whole_train.class_names),
        rank=rank,
        kind="p",
    )
    recovery_metadata = {
        "protocol": "ground_truth_first_whole_rank_matched_v1",
        "kind": "p",
        "attribute": attribute,
        "rank": rank,
        "steps": steps,
        "batch_size": batch_size,
        "evaluation_batch_size": evaluation_batch_size,
        "max_validation_examples": max_validation_examples,
        "seed": seed,
        "reference_whole_probe_checkpoint": str(reference_whole_checkpoint.resolve()),
        "reference_whole_probe_checkpoint_sha256": _sha256(reference_whole_checkpoint),
        "reference_whole_probe_rank": rank,
        "probe_cache_manifest_sha256": cache_manifest_sha256,
        "backbone_checkpoint": (
            str(Path(backbone_checkpoint).resolve()) if backbone_checkpoint is not None else None
        ),
    }
    result = train_probe(
        probe,
        train_data,
        validation_for_training,
        device=device,
        batch_size=batch_size,
        steps=steps,
        seed=seed,
        logger=logger,
        log_interval=log_interval,
        recovery_path=recovery_path,
        checkpoint_interval_steps=checkpoint_interval_steps,
        recovery_metadata=recovery_metadata,
        resume=resume,
        evaluate_train=evaluate_train,
        evaluate_validation=False,
        evaluation_batch_size=evaluation_batch_size,
    )
    validation = evaluate_ground_truth_first_whole_by_source_position(
        probe,
        validation_data,
        device=device,
        batch_size=evaluation_batch_size,
        max_examples=max_validation_examples,
        logger=logger,
        log_interval=log_interval,
    )
    result["validation_accuracy"] = [float(validation["accuracy"])]
    result["monitoring"]["validation"] = validation["monitoring"]
    result.update(
        {
            **recovery_metadata,
            "target": "whole",
            "class_names": list(whole_train.class_names),
            "classes": len(whole_train.class_names),
            "first_token_class_names": list(first_train.class_names),
            "first_token_text": {str(token_id): text for token_id, text in token_entries},
            "ground_truth_first_token": True,
            "ground_truth_first_accuracy_by_position": [1.0] * 6,
            "whole_accuracy_validation_by_source_position": validation["accuracy_by_position"],
            "whole_correct_validation_by_source_position": validation["correct_by_position"],
            "whole_total_validation_by_source_position": validation["total_by_position"],
            "validation_examples_total": len(validation_data),
            "validation_examples_evaluated": len(validation_for_training),
            "input_protocol": (
                "P: biography prefix through original pre-attribute readout, append cached "
                "ground-truth leading-space t1, read t1 final-layer hidden"
            ),
            "backbone_parameters_updated": False,
            "architecture_match": {
                "probe_class": "AttributeProbe",
                "rank": rank,
                "reference_rank": rank,
                "low_rank_embedding_delta": True,
                "normalizer": "LayerNorm",
                "classifier_classes": len(whole_train.class_names),
            },
            "alignment_checks": {
                "first_whole_sample_indices_equal": True,
                "first_whole_profile_indices_equal": True,
                "first_whole_token_offsets_equal": True,
                "first_whole_readout_positions_equal": True,
                "person_disjoint_cache_split": True,
            },
            "stage_two_trainable_components": [
                "low_rank_embedding_delta",
                "normalizer",
                "whole_classifier",
            ],
        }
    )
    return probe, result


@dataclass(frozen=True)
class GroundTruthFirstWholeData:
    first_train: CachedProbeDataset
    whole_train: CachedProbeDataset
    train_data: GroundTruthFirstWholeDataset
    validation_data: GroundTruthFirstWholeDataset
    token_entries: list[tuple[int, str]]
    reference_whole_checkpoint: Path
    cache_manifest_sha256: str
    rank: int


def prepare_ground_truth_first_whole_data(
    *,
    backbone: MiniTransformer,
    cache_root: str | Path,
    probe_dir: str | Path,
    attribute: str,
    backbone_checkpoint: str | Path | None = None,
) -> GroundTruthFirstWholeData:
    """Build true-t1 inputs and bind rank to the original formal whole probe."""

    if attribute not in WHOLE_ATTRIBUTES:
        raise ValueError(f"ground-truth-t1 whole probe does not support {attribute!r}")
    cache_root, probe_dir = Path(cache_root), Path(probe_dir)
    codec = GPT2Codec()
    cache_manifest_sha256 = _sha256(cache_root / "manifest.json")
    first_train = CachedProbeDataset(
        cache_root, kind="p", attribute=attribute, target="first", split="train"
    )
    first_validation = CachedProbeDataset(
        cache_root, kind="p", attribute=attribute, target="first", split="validation"
    )
    whole_train = CachedProbeDataset(
        cache_root, kind="p", attribute=attribute, target="whole", split="train"
    )
    whole_validation = CachedProbeDataset(
        cache_root, kind="p", attribute=attribute, target="whole", split="validation"
    )
    token_entries = [first_token_text(name, codec) for name in first_train.class_names]
    if first_train.class_names != first_validation.class_names:
        raise ValueError("first-token train/validation class mappings differ")
    if whole_train.class_names != whole_validation.class_names:
        raise ValueError("whole train/validation class mappings differ")
    for split, first_data, whole_data in (
        ("train", first_train, whole_train),
        ("validation", first_validation, whole_validation),
    ):
        if not np.array_equal(first_data.sample_indices, whole_data.sample_indices):
            raise ValueError(f"{split} first/whole sample identities differ")
        if not np.array_equal(first_data.profile_indices, whole_data.profile_indices):
            raise ValueError(f"{split} first/whole person identities differ")
        if not np.array_equal(first_data.offsets, whole_data.offsets):
            raise ValueError(f"{split} first/whole token offsets differ")
        if not np.array_equal(first_data.positions, whole_data.positions):
            raise ValueError(f"{split} first/whole P readout positions differ")
    reference_whole_checkpoint = probe_dir / f"p_{attribute}_whole.pt"
    with torch.serialization.safe_globals([TorchVersion]):
        reference_payload = torch.load(
            reference_whole_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    metadata = reference_payload.get("result")
    if not isinstance(metadata, dict):
        raise ValueError(f"{reference_whole_checkpoint} is missing result metadata")
    expected = {"kind": "p", "attribute": attribute, "target": "whole"}
    if {key: metadata.get(key) for key in expected} != expected:
        raise ValueError("reference whole probe task identity mismatch")
    if list(metadata.get("class_names", ())) != whole_train.class_names:
        raise ValueError("reference whole probe class mapping mismatch")
    if metadata.get("probe_cache_manifest_sha256") != cache_manifest_sha256:
        raise ValueError("reference whole probe cache manifest mismatch")
    if (
        backbone_checkpoint is not None
        and Path(str(metadata.get("checkpoint"))).resolve() != Path(backbone_checkpoint).resolve()
    ):
        raise ValueError("reference whole probe backbone checkpoint mismatch")
    rank = int(metadata["rank"])
    if rank != 2:
        raise ValueError(f"reference P whole probe rank is {rank}, expected 2")
    state = reference_payload.get("probe")
    if not isinstance(state, dict):
        raise ValueError("reference whole probe is missing its state dict")
    expected_shapes = {
        "delta.a.weight": (backbone.cfg.vocab_size, rank),
        "delta.b.weight": (backbone.cfg.hidden_size, rank),
        "normalizer.weight": (backbone.cfg.hidden_size,),
        "normalizer.bias": (backbone.cfg.hidden_size,),
        "classifier.weight": (len(whole_train.class_names), backbone.cfg.hidden_size),
        "classifier.bias": (len(whole_train.class_names),),
    }
    actual_shapes = {
        key: tuple(state[key].shape) if key in state else None for key in expected_shapes
    }
    if actual_shapes != expected_shapes:
        raise ValueError(f"reference whole probe trainable architecture mismatch: {actual_shapes}")
    token_ids = [token_id for token_id, _ in token_entries]
    train_data = GroundTruthFirstWholeDataset(
        first_data=first_train,
        whole_data=whole_train,
        token_ids_by_class=token_ids,
    )
    validation_data = GroundTruthFirstWholeDataset(
        first_data=first_validation,
        whole_data=whole_validation,
        token_ids_by_class=token_ids,
    )
    return GroundTruthFirstWholeData(
        first_train=first_train,
        whole_train=whole_train,
        train_data=train_data,
        validation_data=validation_data,
        token_entries=token_entries,
        reference_whole_checkpoint=reference_whole_checkpoint,
        cache_manifest_sha256=cache_manifest_sha256,
        rank=rank,
    )


@torch.no_grad()
def collect_q_predictions(
    *,
    backbone: MiniTransformer,
    data_root: str | Path,
    cache_root: str | Path,
    probe_dir: str | Path,
    attribute: str,
    device: torch.device,
    batch_size: int,
    max_examples: int | None = None,
    backbone_checkpoint: str | Path | None = None,
    progress: Callable[[int], None] | None = None,
) -> tuple[list[QPrediction], AttributeProbe]:
    """Run aligned Q-first and Q-whole heads on the held-out person split."""

    data_root, cache_root, probe_dir = Path(data_root), Path(cache_root), Path(probe_dir)
    first_data = CachedProbeDataset(
        cache_root, kind="q", attribute=attribute, target="first", split="validation"
    )
    whole_data = CachedProbeDataset(
        cache_root, kind="q", attribute=attribute, target="whole", split="validation"
    )
    if len(first_data) != len(whole_data):
        raise ValueError("first and whole Q datasets are not aligned")
    profiles = _read_profiles(data_root)
    cache_manifest_sha256 = _sha256(cache_root / "manifest.json")
    first_probe = _load_probe(
        probe_dir / f"q_{attribute}_first.pt",
        backbone=backbone,
        dataset=first_data,
        kind="q",
        attribute=attribute,
        target="first",
        backbone_checkpoint=backbone_checkpoint,
        cache_manifest_sha256=cache_manifest_sha256,
    ).to(device)
    whole_probe = _load_probe(
        probe_dir / f"q_{attribute}_whole.pt",
        backbone=backbone,
        dataset=whole_data,
        kind="q",
        attribute=attribute,
        target="whole",
        backbone_checkpoint=backbone_checkpoint,
        cache_manifest_sha256=cache_manifest_sha256,
    ).to(device)
    first_probe.eval()
    whole_probe.eval()
    count = len(first_data) if max_examples is None else min(max_examples, len(first_data))
    records: list[QPrediction] = []
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        first_items = [first_data[index] for index in range(start, end)]
        whole_items = [whole_data[index] for index in range(start, end)]
        first_ids, first_positions, first_labels = collate_probe(first_items)
        whole_ids, whole_positions, whole_labels = collate_probe(whole_items)
        if not torch.equal(first_ids, whole_ids):
            raise ValueError("first and whole Q inputs are not aligned")
        first_logits = first_probe(first_ids.to(device), first_positions.to(device))[:, 0]
        whole_logits = whole_probe(whole_ids.to(device), whole_positions.to(device))[:, 0]
        first_predictions = first_logits.argmax(-1).cpu()
        whole_predictions = whole_logits.argmax(-1).cpu()
        whole_probabilities = whole_logits.softmax(-1).gather(1, whole_labels.to(device)[:, None])
        for offset, index in enumerate(range(start, end)):
            profile_index = _dataset_profile_index(first_data, index)
            profile = profiles[profile_index]
            true_first_id = int(first_labels[offset])
            pred_first_id = int(first_predictions[offset])
            true_whole_id = int(whole_labels[offset])
            pred_whole_id = int(whole_predictions[offset])
            records.append(
                QPrediction(
                    case_id=index,
                    profile_index=profile_index,
                    person_id=str(profile["person_id"]),
                    attribute=attribute,
                    input_ids=tuple(first_items[offset].input_ids),
                    true_first_id=true_first_id,
                    pred_first_id=pred_first_id,
                    true_first_token=first_data.class_names[true_first_id],
                    pred_first_token=first_data.class_names[pred_first_id],
                    true_whole_id=true_whole_id,
                    pred_whole_id=pred_whole_id,
                    true_whole_value=whole_data.class_names[true_whole_id],
                    pred_whole_value=whole_data.class_names[pred_whole_id],
                    whole_true_probability=float(whole_probabilities[offset]),
                )
            )
        if progress is not None:
            progress(end)
    return records, whole_probe


def _prediction_row(record: QPrediction) -> dict[str, object]:
    return {
        "case_id": record.case_id,
        "profile_index": record.profile_index,
        "person_id": record.person_id,
        "attribute": record.attribute,
        "true_first_token": record.true_first_token,
        "pred_first_token": record.pred_first_token,
        "true_whole_value": record.true_whole_value,
        "pred_whole_value": record.pred_whole_value,
        "first_correct": record.first_correct,
        "whole_correct": record.whole_correct,
        "whole_true_probability": record.whole_true_probability,
    }


def insert_oracle_first_token(input_ids: Sequence[int], token_id: int, eos_id: int) -> list[int]:
    """Insert one token before the final Q readout EOS without moving the readout."""

    if not input_ids or input_ids[-1] != eos_id:
        raise ValueError("Q input must end in the readout EOS token")
    return [*input_ids[:-1], int(token_id), eos_id]


def summarize_oracle_rows(rows: Sequence[dict[str, object]]) -> dict[str, float | int]:
    count = len(rows)
    before_correct = sum(bool(row["whole_before_correct"]) for row in rows)
    after_correct = sum(bool(row["whole_after_correct"]) for row in rows)
    recoverable = [row for row in rows if not bool(row["whole_before_correct"])]
    recovered = sum(bool(row["whole_after_correct"]) for row in recoverable)
    initially_correct = [row for row in rows if bool(row["whole_before_correct"])]
    harmed = sum(not bool(row["whole_after_correct"]) for row in initially_correct)
    return {
        "examples": count,
        "accuracy_before": before_correct / count if count else 0.0,
        "accuracy_after": after_correct / count if count else 0.0,
        "accuracy_delta": (after_correct - before_correct) / count if count else 0.0,
        "baseline_errors": len(recoverable),
        "recovered_errors": recovered,
        "recovery_rate": recovered / len(recoverable) if recoverable else 0.0,
        "baseline_correct": len(initially_correct),
        "harmed_correct": harmed,
        "harm_rate": harmed / len(initially_correct) if initially_correct else 0.0,
    }


def _plot_oracle(summary_rows: Sequence[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    attributes = [str(row["attribute"]) for row in summary_rows]
    before = [100 * float(row["accuracy_before"]) for row in summary_rows]
    after = [100 * float(row["accuracy_after"]) for row in summary_rows]
    recovery = [100 * float(row["recovery_rate"]) for row in summary_rows]
    x = list(range(len(attributes)))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    width = 0.38
    axes[0].bar([value - width / 2 for value in x], before, width, label="name only")
    axes[0].bar([value + width / 2 for value in x], after, width, label="+ true first token")
    axes[0].set_ylabel("Q-whole held-out accuracy (%)")
    axes[0].set_xticks(x, attributes, rotation=25, ha="right")
    axes[0].set_ylim(0, 100)
    axes[0].legend()
    axes[0].set_title("Oracle first-token intervention")
    axes[1].bar(x, recovery, color="#4c9f70")
    axes[1].set_ylabel("Recovered baseline errors (%)")
    axes[1].set_xticks(x, attributes, rotation=25, ha="right")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Recovery among name-only errors")
    figure.savefig(output, dpi=180)
    plt.close(figure)


@torch.no_grad()
def oracle_first_token_validation(
    *,
    backbone: MiniTransformer,
    data_root: str | Path,
    cache_root: str | Path,
    probe_dir: str | Path,
    output_dir: str | Path,
    device: torch.device,
    attributes: Sequence[str] = WHOLE_ATTRIBUTES,
    batch_size: int = 1024,
    max_examples: int | None = None,
    backbone_checkpoint: str | Path | None = None,
    progress: Callable[[str, int], None] | None = None,
) -> dict[str, object]:
    """Compare Q-whole accuracy before/after inserting the true first token."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive or None")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    codec = GPT2Codec()
    all_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for attribute in attributes:
        records, whole_probe = collect_q_predictions(
            backbone=backbone,
            data_root=data_root,
            cache_root=cache_root,
            probe_dir=probe_dir,
            attribute=attribute,
            device=device,
            batch_size=batch_size,
            max_examples=max_examples,
            backbone_checkpoint=backbone_checkpoint,
            progress=(lambda done, name=attribute: progress(name, done)) if progress else None,
        )
        whole_class_names = task_class_names(cache_root, attribute, "whole")
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            items = [
                ProbeBatchItem(
                    insert_oracle_first_token(
                        record.input_ids, int(record.true_first_token), codec.eos
                    ),
                    [len(record.input_ids)],
                    record.true_whole_id,
                )
                for record in batch
            ]
            input_ids, positions, labels = collate_probe(items)
            logits = whole_probe(input_ids.to(device), positions.to(device))[:, 0]
            predictions = logits.argmax(-1).cpu()
            probabilities = logits.softmax(-1).gather(1, labels.to(device)[:, None])
            for offset, record in enumerate(batch):
                row = _prediction_row(record)
                row.update(
                    {
                        "whole_before_correct": record.whole_correct,
                        "whole_after_correct": int(predictions[offset]) == record.true_whole_id,
                        "pred_whole_after": whole_class_names[int(predictions[offset])],
                        "whole_true_probability_after": float(probabilities[offset]),
                        "true_probability_delta": float(probabilities[offset])
                        - record.whole_true_probability,
                    }
                )
                all_rows.append(row)
        attribute_rows = [row for row in all_rows if row["attribute"] == attribute]
        summary_rows.append({"attribute": attribute, **summarize_oracle_rows(attribute_rows)})
        del whole_probe
    overall = summarize_oracle_rows(all_rows)
    summary = {
        "protocol": "q_whole_oracle_first_token_v1",
        "intervention": "[EOS, name, true_t1, EOS], read final EOS with unchanged Q-whole head",
        "split": "person-held-out validation",
        "parameters_updated": False,
        "data": str(Path(data_root).resolve()),
        "probe_cache": str(Path(cache_root).resolve()),
        "probe_dir": str(Path(probe_dir).resolve()),
        "backbone_checkpoint": (
            str(Path(backbone_checkpoint).resolve()) if backbone_checkpoint is not None else None
        ),
        "overall": overall,
        "attributes": summary_rows,
    }
    _write_csv(output / "records.csv", all_rows)
    _write_csv(output / "summary.csv", summary_rows)
    _write_json(output / "summary.json", summary)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    _plot_oracle(summary_rows, figures / "accuracy_before_after.png")
    return summary


@contextmanager
def _capture_router_indices(model: MiniTransformer):
    captured: list[torch.Tensor] = []
    handles = []

    def hook(_module, _inputs, output):
        captured.append(output.expert_indices.detach())

    for block in model.blocks:
        if hasattr(block.ffn, "router"):
            handles.append(block.ffn.router.register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def _routes_at_positions(
    model: MiniTransformer,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Return routes as ``[batch, selected_position, layer, top_k]``."""

    if not model.cfg.is_moe:
        raise ValueError("bad-case route validation requires an MoE backbone")
    model.eval()
    with _capture_router_indices(model) as captured:
        model.hidden_states(input_ids)
    batch = torch.arange(input_ids.shape[0], device=input_ids.device)[:, None]
    selected = [
        routes.view(input_ids.shape[0], input_ids.shape[1], -1)[batch, positions]
        for routes in captured
    ]
    return torch.stack(selected, dim=2).cpu()


def _route_jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def _sample_group_pairs(
    groups: dict[int, list[int]],
    *,
    same_second_token: bool,
    limit: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    eligible = [key for key, values in groups.items() if values]
    if same_second_token:
        eligible = [key for key in eligible if len(groups[key]) >= 2]
        if not eligible:
            return []
    elif len(eligible) < 2:
        return []
    pairs: list[tuple[int, int]] = []
    attempts = 0
    while len(pairs) < limit and attempts < max(100, limit * 10):
        attempts += 1
        if same_second_token:
            key = rng.choice(eligible)
            left, right = rng.sample(groups[key], 2)
        else:
            left_key, right_key = rng.sample(eligible, 2)
            left, right = rng.choice(groups[left_key]), rng.choice(groups[right_key])
        pairs.append((left, right))
    return pairs


def pairwise_route_summary(
    cases: Sequence[dict[str, object]],
    *,
    layers: int,
    pair_limit: int = 2000,
    seed: int = 1337,
) -> list[dict[str, object]]:
    """Compare route overlap for same-t1 pairs with same versus different t2."""

    rng = random.Random(seed)
    by_attribute_t1: dict[tuple[str, int], dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, case in enumerate(cases):
        by_attribute_t1[(str(case["attribute"]), int(case["t1_id"]))][int(case["t2_id"])].append(
            index
        )
    accumulators: dict[tuple[str, int, str], list[tuple[float, float]]] = defaultdict(list)
    for (attribute, _t1), t2_groups in by_attribute_t1.items():
        for label, same in (("same_t2", True), ("different_t2", False)):
            pairs = _sample_group_pairs(
                t2_groups,
                same_second_token=same,
                limit=pair_limit,
                rng=rng,
            )
            for left_index, right_index in pairs:
                left, right = cases[left_index], cases[right_index]
                left_routes = left["routes"]
                right_routes = right["routes"]
                for layer in range(layers):
                    t1_overlap = _route_jaccard(left_routes[0][layer], right_routes[0][layer])
                    t2_overlap = _route_jaccard(left_routes[1][layer], right_routes[1][layer])
                    accumulators[(attribute, layer, label)].append((t1_overlap, t2_overlap))
    rows = []
    for (attribute, layer, label), values in sorted(accumulators.items()):
        t1_mean = sum(value[0] for value in values) / len(values)
        t2_mean = sum(value[1] for value in values) / len(values)
        rows.append(
            {
                "attribute": attribute,
                "layer": layer,
                "pair_group": label,
                "pair_count": len(values),
                "t1_route_overlap": t1_mean,
                "t2_route_overlap": t2_mean,
                "branching_score": t1_mean - t2_mean,
            }
        )
    return rows


@torch.no_grad()
def bad_case_route_validation(
    *,
    backbone: MiniTransformer,
    data_root: str | Path,
    cache_root: str | Path,
    probe_dir: str | Path,
    output_dir: str | Path,
    device: torch.device,
    attributes: Sequence[str] = WHOLE_ATTRIBUTES,
    batch_size: int = 512,
    max_examples: int | None = None,
    backbone_checkpoint: str | Path | None = None,
    pair_limit: int = 2000,
    progress: Callable[[str, int], None] | None = None,
) -> dict[str, object]:
    """Analyze t1/t2 expert branching on first-correct/whole-wrong Q cases."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive or None")
    if pair_limit <= 0:
        raise ValueError("pair_limit must be positive")
    if not backbone.cfg.is_moe:
        raise ValueError("bad-case route validation requires an MoE backbone")
    if backbone.cfg.experts_per_token != 2:
        raise ValueError(
            "bad-case route validation currently requires exactly two routed experts per token"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    codec = GPT2Codec()
    all_cases: list[dict[str, object]] = []
    for attribute in attributes:
        predictions, whole_probe = collect_q_predictions(
            backbone=backbone,
            data_root=data_root,
            cache_root=cache_root,
            probe_dir=probe_dir,
            attribute=attribute,
            device=device,
            batch_size=batch_size,
            max_examples=max_examples,
            backbone_checkpoint=backbone_checkpoint,
            progress=(lambda done, name=attribute: progress(name, done)) if progress else None,
        )
        del whole_probe
        bad = [
            record
            for record in predictions
            if record.first_correct
            and not record.whole_correct
            and len(codec.encode(" " + record.true_whole_value)) >= 2
        ]
        for start in range(0, len(bad), batch_size):
            batch = bad[start : start + batch_size]
            tokenized = [codec.encode(" " + record.true_whole_value) for record in batch]
            items = []
            for record, attribute_ids in zip(batch, tokenized):
                prefix = list(record.input_ids[:-1])
                t1_position = len(prefix)
                ids = [*prefix, attribute_ids[0], attribute_ids[1], codec.eos]
                items.append(
                    ProbeBatchItem(ids, [t1_position, t1_position + 1], record.true_whole_id)
                )
            input_ids, positions, _ = collate_probe(items)
            routes = _routes_at_positions(
                backbone,
                input_ids.to(device),
                positions.to(device),
            )
            for offset, (record, attribute_ids) in enumerate(zip(batch, tokenized)):
                all_cases.append(
                    {
                        **_prediction_row(record),
                        "t1_id": attribute_ids[0],
                        "t2_id": attribute_ids[1],
                        "t1_text": codec.encoding.decode([attribute_ids[0]]),
                        "t2_text": codec.encoding.decode([attribute_ids[1]]),
                        "routes": routes[offset].tolist(),
                    }
                )
    layers = backbone.cfg.n_layers
    route_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    for case in all_cases:
        case_row = {key: value for key, value in case.items() if key != "routes"}
        case_row["t1_route_path"] = "|".join(
            "+".join(map(str, layer)) for layer in case["routes"][0]
        )
        case_row["t2_route_path"] = "|".join(
            "+".join(map(str, layer)) for layer in case["routes"][1]
        )
        case_rows.append(case_row)
        for layer in range(layers):
            t1_route, t2_route = case["routes"][0][layer], case["routes"][1][layer]
            route_rows.append(
                {
                    "case_id": case["case_id"],
                    "person_id": case["person_id"],
                    "attribute": case["attribute"],
                    "t1_id": case["t1_id"],
                    "t2_id": case["t2_id"],
                    "layer": layer,
                    "t1_expert_0": t1_route[0],
                    "t1_expert_1": t1_route[1],
                    "t2_expert_0": t2_route[0],
                    "t2_expert_1": t2_route[1],
                    "within_case_jaccard": _route_jaccard(t1_route, t2_route),
                    "top1_changed": t1_route[0] != t2_route[0],
                }
            )
    pair_rows = pairwise_route_summary(
        all_cases,
        layers=layers,
        pair_limit=pair_limit,
    )
    nmi_rows = []
    for attribute in attributes:
        selected = [case for case in all_cases if case["attribute"] == attribute]
        for layer in range(layers):
            nmi_rows.append(
                {
                    "attribute": attribute,
                    "layer": layer,
                    "t1_top1_token_nmi": normalized_mutual_information(
                        [case["routes"][0][layer][0] for case in selected],
                        [case["t1_id"] for case in selected],
                    ),
                    "t2_top1_token_nmi": normalized_mutual_information(
                        [case["routes"][1][layer][0] for case in selected],
                        [case["t2_id"] for case in selected],
                    ),
                    "examples": len(selected),
                }
            )
    summary = {
        "protocol": "q_bad_case_t1_t2_route_branching_v1",
        "case_definition": "Q-first correct, Q-whole wrong, whole value has >=2 tokens",
        "route_model": "frozen pretrained backbone without probe embedding delta",
        "split": "person-held-out validation",
        "parameters_updated": False,
        "data": str(Path(data_root).resolve()),
        "probe_cache": str(Path(cache_root).resolve()),
        "probe_dir": str(Path(probe_dir).resolve()),
        "backbone_checkpoint": (
            str(Path(backbone_checkpoint).resolve()) if backbone_checkpoint is not None else None
        ),
        "examples": len(all_cases),
        "attributes": {
            attribute: sum(case["attribute"] == attribute for case in all_cases)
            for attribute in attributes
        },
        "layers": layers,
        "experts": backbone.cfg.num_experts,
        "top_k": backbone.cfg.experts_per_token,
    }
    prediction_fields = [
        "case_id",
        "profile_index",
        "person_id",
        "attribute",
        "true_first_token",
        "pred_first_token",
        "true_whole_value",
        "pred_whole_value",
        "first_correct",
        "whole_correct",
        "whole_true_probability",
    ]
    _write_csv(
        output / "bad_cases.csv",
        case_rows,
        fieldnames=[
            *prediction_fields,
            "t1_id",
            "t2_id",
            "t1_text",
            "t2_text",
            "t1_route_path",
            "t2_route_path",
        ],
    )
    _write_csv(
        output / "route_records.csv",
        route_rows,
        fieldnames=[
            "case_id",
            "person_id",
            "attribute",
            "t1_id",
            "t2_id",
            "layer",
            "t1_expert_0",
            "t1_expert_1",
            "t2_expert_0",
            "t2_expert_1",
            "within_case_jaccard",
            "top1_changed",
        ],
    )
    _write_csv(
        output / "pairwise_branching.csv",
        pair_rows,
        fieldnames=[
            "attribute",
            "layer",
            "pair_group",
            "pair_count",
            "t1_route_overlap",
            "t2_route_overlap",
            "branching_score",
        ],
    )
    _write_csv(
        output / "token_route_nmi.csv",
        nmi_rows,
        fieldnames=[
            "attribute",
            "layer",
            "t1_top1_token_nmi",
            "t2_top1_token_nmi",
            "examples",
        ],
    )
    _write_json(output / "summary.json", summary)
    return summary
