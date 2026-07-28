"""Load aligned Q-first and Q-whole predictions from retained probe heads."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.torch_version import TorchVersion

from experiments.synbios_moe.artifact_io import sha256_file
from experiments.synbios_moe.probes.dataset import CachedProbeDataset
from experiments.synbios_moe.probes.model import (
    AttributeProbe,
    collate_probe,
)
from minitrain.model.transformer import MiniTransformer


@dataclass(frozen=True)
class QPrediction:
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


def task_class_names(cache_root: str | Path, attribute: str, target: str) -> list[str]:
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
    cache_manifest_sha256 = sha256_file(cache_root / "manifest.json")
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


def prediction_row(record: QPrediction) -> dict[str, object]:
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
