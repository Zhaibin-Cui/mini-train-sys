"""Matched sequential probes and inference diagnostics for probe failures.

The legacy diagnostics in this module never update model or probe parameters:

* ``oracle_first_token_validation`` measures whether inserting the ground-truth
  first attribute token before the Q readout EOS unlocks a trained Q-whole head.
* ``bad_case_route_validation`` finds Q-first-correct/Q-whole-wrong examples and
  tests whether examples sharing token 1 use similar expert paths before
  branching at token 2.

``train_predicted_first_whole_probe`` is intentionally different: it freezes a
completed first-token probe, inserts its hard prediction, reruns the frozen
backbone, and trains a fresh whole-value readout on the matched hidden state.
P reads the inserted token itself; Q reads the EOS immediately after it.

Raw per-example evidence, tidy aggregates, and plots are deliberately kept
separate so conclusions remain auditable.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import random
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.torch_version import TorchVersion

from experiments.synbios_moe.data import ATTRIBUTES
from experiments.synbios_moe.probe_data import CachedProbeDataset
from experiments.synbios_moe.probes import (
    AttributeProbe,
    GPT2Codec,
    ProbeBatchItem,
    collate_probe,
    train_probe,
)
from experiments.synbios_moe.router_analysis import normalized_mutual_information
from minitrain.model.transformer import MiniTransformer


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


def predicted_first_token_text(class_name: str, codec: GPT2Codec) -> tuple[int, str]:
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


def build_predicted_first_input(
    *,
    kind: str,
    item: ProbeBatchItem,
    source_position: int,
    token_id: int,
    eos_id: int,
) -> ProbeBatchItem:
    """Insert a hard t1 prediction and select the requested matched readout."""

    if kind == "p":
        if not 0 <= source_position < len(item.positions):
            raise IndexError("P source position is outside the cached item")
        prefix = item.input_ids[: item.positions[source_position] + 1]
        input_ids = [*prefix, int(token_id)]
        return ProbeBatchItem(input_ids, [len(input_ids) - 1], item.label)
    if kind == "q":
        if source_position != 0 or len(item.positions) != 1:
            raise ValueError("Q has exactly one source position")
        if not item.input_ids or item.input_ids[-1] != eos_id:
            raise ValueError("Q input must end in its readout EOS")
        input_ids = [*item.input_ids[:-1], int(token_id), eos_id]
        return ProbeBatchItem(input_ids, [len(input_ids) - 1], item.label)
    raise ValueError(f"invalid probe kind: {kind}")


class PredictedFirstWholeDataset(Dataset):
    """Whole labels paired with inputs rebuilt from frozen first-probe predictions."""

    def __init__(
        self,
        *,
        kind: str,
        first_data: CachedProbeDataset,
        whole_data: CachedProbeDataset,
        predicted_class_ids: Sequence[Sequence[int]],
        token_ids_by_class: Sequence[int],
        eos_id: int,
    ) -> None:
        if kind not in {"p", "q"}:
            raise ValueError("kind must be p or q")
        if len(first_data) != len(whole_data) or len(first_data) != len(predicted_class_ids):
            raise ValueError("first/whole datasets and predictions are not aligned")
        expected_positions = 6 if kind == "p" else 1
        if any(len(row) != expected_positions for row in predicted_class_ids):
            raise ValueError(f"{kind.upper()} predictions must have {expected_positions} positions")
        self.kind = kind
        self.first_data = first_data
        self.whole_data = whole_data
        self.predicted_class_ids = tuple(
            tuple(int(value) for value in row) for row in predicted_class_ids
        )
        self.token_ids_by_class = tuple(int(value) for value in token_ids_by_class)
        self.eos_id = int(eos_id)
        self.positions_per_source = expected_positions
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
            if self.kind == "p":
                return int(self.first_data.positions[sample][source_position]) + 2
            return int(self.first_data.offsets[sample + 1] - self.first_data.offsets[sample]) + 1

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
        predicted_class = self.predicted_class_ids[source_index][source_position]
        if not 0 <= predicted_class < len(self.token_ids_by_class):
            raise ValueError("predicted first-token class is out of range")
        rebuilt = build_predicted_first_input(
            kind=self.kind,
            item=first_item,
            source_position=source_position,
            token_id=self.token_ids_by_class[predicted_class],
            eos_id=self.eos_id,
        )
        return ProbeBatchItem(rebuilt.input_ids, rebuilt.positions, whole_item.label)


class PredictedFirstWholeProbe(nn.Module):
    """Frozen backbone plus a matched whole head after a predicted t1 rerun."""

    def __init__(self, backbone: MiniTransformer, num_classes: int, *, kind: str) -> None:
        super().__init__()
        if kind not in {"p", "q"}:
            raise ValueError("kind must be p or q")
        self.backbone = backbone
        self.kind = kind
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        self.normalizer: nn.Module = (
            nn.LayerNorm(backbone.cfg.hidden_size)
            if kind == "p"
            else nn.BatchNorm1d(backbone.cfg.hidden_size)
        )
        self.classifier = nn.Linear(backbone.cfg.hidden_size, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        # The second stage trains only the matched readout. Keep the frozen
        # backbone deterministic so route changes come from the inserted token.
        self.backbone.eval()
        return self

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.backbone.hidden_states(input_ids)
        batch = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        selected = hidden[batch, positions]
        shape = selected.shape
        normalized = self.normalizer(selected.reshape(-1, shape[-1]))
        return self.classifier(normalized).view(shape[0], shape[1], -1)


@torch.no_grad()
def _predict_first_classes(
    probe: AttributeProbe,
    dataset: CachedProbeDataset,
    *,
    device: torch.device,
    batch_size: int,
    progress: Callable[[int], None] | None = None,
) -> tuple[list[list[int]], list[float]]:
    if batch_size <= 0:
        raise ValueError("first-token prediction batch size must be positive")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_probe,
        pin_memory=device.type == "cuda",
    )
    probe.to(device).eval()
    predictions: list[list[int]] = []
    correct: list[int] | None = None
    total = 0
    for input_ids, positions, labels in loader:
        logits = probe(
            input_ids.to(device, non_blocking=True),
            positions.to(device, non_blocking=True),
        )
        predicted = logits.argmax(-1).cpu()
        predictions.extend(predicted.tolist())
        expanded = labels[:, None].expand_as(predicted)
        batch_correct = (predicted == expanded).sum(0).tolist()
        correct = (
            batch_correct
            if correct is None
            else [left + right for left, right in zip(correct, batch_correct)]
        )
        total += len(labels)
        if progress is not None:
            progress(total)
    return predictions, [value / max(total, 1) for value in (correct or [])]


@torch.no_grad()
def evaluate_predicted_first_whole_by_source_position(
    probe: PredictedFirstWholeProbe,
    dataset: PredictedFirstWholeDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_probe,
        pin_memory=device.type == "cuda",
    )
    probe.to(device).eval()
    correct = [0] * dataset.positions_per_source
    total = [0] * dataset.positions_per_source
    offset = 0
    for input_ids, positions, labels in loader:
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
    return [hits / max(count, 1) for hits, count in zip(correct, total)]


def train_predicted_first_whole_probe(
    *,
    backbone: MiniTransformer,
    cache_root: str | Path,
    probe_dir: str | Path,
    kind: str,
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
    prediction_progress: Callable[[str, int], None] | None = None,
) -> tuple[PredictedFirstWholeProbe, dict[str, object]]:
    """Train one pilot-scale whole head after hard predicted-t1 insertion."""

    prepared = prepare_predicted_first_whole_data(
        backbone=backbone,
        cache_root=cache_root,
        probe_dir=probe_dir,
        kind=kind,
        attribute=attribute,
        device=device,
        prediction_batch_size=evaluation_batch_size,
        backbone_checkpoint=backbone_checkpoint,
        prediction_progress=prediction_progress,
    )
    cache_root = Path(cache_root)
    first_train = prepared.first_train
    whole_train = prepared.whole_train
    train_data = prepared.train_data
    validation_data = prepared.validation_data
    token_entries = prepared.token_entries
    first_checkpoint = prepared.first_checkpoint
    cache_manifest_sha256 = prepared.cache_manifest_sha256
    train_first_accuracy = prepared.train_first_accuracy
    validation_first_accuracy = prepared.validation_first_accuracy
    probe = PredictedFirstWholeProbe(backbone, len(whole_train.class_names), kind=kind)
    recovery_metadata = {
        "protocol": "predicted_first_whole_pilot_v1",
        "kind": kind,
        "attribute": attribute,
        "steps": steps,
        "batch_size": batch_size,
        "seed": seed,
        "first_probe_checkpoint": str(first_checkpoint.resolve()),
        "first_probe_checkpoint_sha256": _sha256(first_checkpoint),
        "probe_cache_manifest_sha256": cache_manifest_sha256,
        "backbone_checkpoint": (
            str(Path(backbone_checkpoint).resolve()) if backbone_checkpoint is not None else None
        ),
    }
    result = train_probe(
        probe,
        train_data,
        validation_data,
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
        evaluation_batch_size=evaluation_batch_size,
    )
    result.update(
        {
            **recovery_metadata,
            "target": "whole",
            "class_names": list(whole_train.class_names),
            "classes": len(whole_train.class_names),
            "first_token_class_names": list(first_train.class_names),
            "first_token_text": {str(token_id): text for token_id, text in token_entries},
            "first_accuracy_train_by_position": train_first_accuracy,
            "first_accuracy_validation_by_position": validation_first_accuracy,
            "whole_accuracy_validation_by_source_position": (
                evaluate_predicted_first_whole_by_source_position(
                    probe,
                    validation_data,
                    device=device,
                    batch_size=evaluation_batch_size,
                )
            ),
            "input_protocol": (
                "P: biography prefix through original pre-attribute readout, append hard "
                "predicted leading-space t1, read t1 final-layer hidden"
                if kind == "p"
                else "Q: [EOS, name, hard predicted leading-space t1, EOS], read final EOS "
                "final-layer hidden"
            ),
            "backbone_parameters_updated": False,
            "first_probe_parameters_updated": False,
            "stage_two_trainable_components": ["normalizer", "whole_classifier"],
        }
    )
    return probe, result


@dataclass(frozen=True)
class PredictedFirstWholeData:
    first_train: CachedProbeDataset
    whole_train: CachedProbeDataset
    train_data: PredictedFirstWholeDataset
    validation_data: PredictedFirstWholeDataset
    token_entries: list[tuple[int, str]]
    first_checkpoint: Path
    cache_manifest_sha256: str
    train_first_accuracy: list[float]
    validation_first_accuracy: list[float]


def prepare_predicted_first_whole_data(
    *,
    backbone: MiniTransformer,
    cache_root: str | Path,
    probe_dir: str | Path,
    kind: str,
    attribute: str,
    device: torch.device,
    prediction_batch_size: int,
    backbone_checkpoint: str | Path | None = None,
    prediction_progress: Callable[[str, int], None] | None = None,
) -> PredictedFirstWholeData:
    """Freeze a formal t1 probe and materialize exact stage-two train/val inputs."""

    if attribute not in WHOLE_ATTRIBUTES:
        raise ValueError(f"predicted-t1 whole probe does not support {attribute!r}")
    cache_root, probe_dir = Path(cache_root), Path(probe_dir)
    codec = GPT2Codec()
    cache_manifest_sha256 = _sha256(cache_root / "manifest.json")
    first_train = CachedProbeDataset(
        cache_root, kind=kind, attribute=attribute, target="first", split="train"
    )
    first_validation = CachedProbeDataset(
        cache_root, kind=kind, attribute=attribute, target="first", split="validation"
    )
    whole_train = CachedProbeDataset(
        cache_root, kind=kind, attribute=attribute, target="whole", split="train"
    )
    whole_validation = CachedProbeDataset(
        cache_root, kind=kind, attribute=attribute, target="whole", split="validation"
    )
    token_entries = [predicted_first_token_text(name, codec) for name in first_train.class_names]
    if first_train.class_names != first_validation.class_names:
        raise ValueError("first-token train/validation class mappings differ")
    if whole_train.class_names != whole_validation.class_names:
        raise ValueError("whole train/validation class mappings differ")
    first_checkpoint = probe_dir / f"{kind}_{attribute}_first.pt"
    first_probe = _load_probe(
        first_checkpoint,
        backbone=backbone,
        dataset=first_train,
        kind=kind,
        attribute=attribute,
        target="first",
        backbone_checkpoint=backbone_checkpoint,
        cache_manifest_sha256=cache_manifest_sha256,
    )
    train_predictions, train_first_accuracy = _predict_first_classes(
        first_probe,
        first_train,
        device=device,
        batch_size=prediction_batch_size,
        progress=(
            (lambda count: prediction_progress("train", count))
            if prediction_progress is not None
            else None
        ),
    )
    validation_predictions, validation_first_accuracy = _predict_first_classes(
        first_probe,
        first_validation,
        device=device,
        batch_size=prediction_batch_size,
        progress=(
            (lambda count: prediction_progress("validation", count))
            if prediction_progress is not None
            else None
        ),
    )
    del first_probe
    token_ids = [token_id for token_id, _ in token_entries]
    train_data = PredictedFirstWholeDataset(
        kind=kind,
        first_data=first_train,
        whole_data=whole_train,
        predicted_class_ids=train_predictions,
        token_ids_by_class=token_ids,
        eos_id=codec.eos,
    )
    validation_data = PredictedFirstWholeDataset(
        kind=kind,
        first_data=first_validation,
        whole_data=whole_validation,
        predicted_class_ids=validation_predictions,
        token_ids_by_class=token_ids,
        eos_id=codec.eos,
    )
    return PredictedFirstWholeData(
        first_train=first_train,
        whole_train=whole_train,
        train_data=train_data,
        validation_data=validation_data,
        token_entries=token_entries,
        first_checkpoint=first_checkpoint,
        cache_manifest_sha256=cache_manifest_sha256,
        train_first_accuracy=train_first_accuracy,
        validation_first_accuracy=validation_first_accuracy,
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


def _plot_route_overlap(rows: Sequence[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
    for axis, group in zip(axes, ("same_t2", "different_t2")):
        selected = [row for row in rows if row["pair_group"] == group]
        layers = sorted({int(row["layer"]) for row in selected})
        if not layers:
            axis.text(
                0.5,
                0.5,
                "No eligible pairs",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        for token, field, style in (
            ("t1", "t1_route_overlap", "-"),
            ("t2", "t2_route_overlap", "--"),
        ):
            means = [
                sum(float(row[field]) for row in selected if int(row["layer"]) == layer)
                / max(
                    sum(int(row["layer"]) == layer for row in selected),
                    1,
                )
                for layer in layers
            ]
            axis.plot(layers, means, style, marker="o", label=token)
        axis.set_title(group.replace("_", " "))
        axis.set_xlabel("MoE layer")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Mean top-2 expert-set Jaccard")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_branching_heatmap(rows: Sequence[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["pair_group"] == "different_t2"]
    attributes = sorted({str(row["attribute"]) for row in selected})
    layers = sorted({int(row["layer"]) for row in selected})
    matrix = [
        [
            next(
                (
                    float(row["branching_score"])
                    for row in selected
                    if row["attribute"] == attribute and int(row["layer"]) == layer
                ),
                math.nan,
            )
            for layer in layers
        ]
        for attribute in attributes
    ]
    figure, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    if not attributes or not layers:
        axis.text(0.5, 0.5, "No same-t1/different-t2 pairs", ha="center", va="center")
        axis.set_axis_off()
        figure.savefig(output, dpi=180)
        plt.close(figure)
        return
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(layers)), layers)
    axis.set_yticks(range(len(attributes)), attributes)
    axis.set_xlabel("MoE layer")
    axis.set_title("Branching score: t1 overlap − t2 overlap (same t1, different t2)")
    figure.colorbar(image, ax=axis, label="Branching score")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_expert_heatmaps(
    route_rows: Sequence[dict[str, object]], output: Path, num_experts: int
) -> None:
    import matplotlib.pyplot as plt

    layers = sorted({int(row["layer"]) for row in route_rows})
    if not layers:
        figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
        axis.text(0.5, 0.5, "No eligible bad cases", ha="center", va="center")
        axis.set_axis_off()
        figure.savefig(output, dpi=180)
        plt.close(figure)
        return
    matrices = []
    for token in ("t1", "t2"):
        matrix = []
        for layer in layers:
            counts = [0] * num_experts
            for row in route_rows:
                if int(row["layer"]) != layer:
                    continue
                counts[int(row[f"{token}_expert_0"])] += 1
                counts[int(row[f"{token}_expert_1"])] += 1
            total = sum(counts)
            matrix.append([count / total if total else 0.0 for count in counts])
        matrices.append(matrix)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, token, matrix in zip(axes, ("t1", "t2"), matrices):
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0)
        axis.set_title(f"Bad-case {token} route load")
        axis.set_xlabel("Expert")
        axis.set_ylabel("Layer")
        axis.set_xticks(range(num_experts))
        axis.set_yticks(range(len(layers)), layers)
        figure.colorbar(image, ax=axis, label="Route fraction")
    figure.savefig(output, dpi=180)
    plt.close(figure)


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
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    _plot_route_overlap(pair_rows, figures / "route_overlap_by_layer.png")
    _plot_branching_heatmap(pair_rows, figures / "branching_heatmap.png")
    _plot_expert_heatmaps(
        route_rows,
        figures / "expert_load_t1_vs_t2.png",
        backbone.cfg.num_experts,
    )
    return summary
