"""Train fresh P-whole heads on prefixes extended with the cached first-token label."""


import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.torch_version import TorchVersion

from experiments.synbios_moe.artifact_io import (
    sha256_file,
)
from experiments.synbios_moe.pretraining.dataset import WHOLE_ATTRIBUTES
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
        "reference_whole_probe_checkpoint_sha256": sha256_file(reference_whole_checkpoint),
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
    cache_manifest_sha256 = sha256_file(cache_root / "manifest.json")
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
