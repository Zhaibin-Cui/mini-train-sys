"""Pre-tokenized, memory-mapped datasets shared by every SynBioS probe task."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from torch.utils.data import Dataset

from experiments.synbios_moe.data import ATTRIBUTES
from experiments.synbios_moe.probes import (
    GPT2Codec,
    ProbeBatchItem,
    ordered_p_probe_starts,
    task_label,
)


CACHE_FORMAT_VERSION = 2
SPLIT_CODES = {"train": 0, "validation": 1}


@dataclass(frozen=True)
class ProbeTask:
    attribute: str
    target: str

    @property
    def key(self) -> str:
        return f"{self.attribute}_{self.target}"


def paper_probe_tasks() -> tuple[ProbeTask, ...]:
    """Return the paper's six first-token and five whole-attribute tasks."""

    tasks: list[ProbeTask] = []
    for attribute in ATTRIBUTES:
        tasks.append(ProbeTask(attribute, "first"))
        if attribute != "birth_date":
            tasks.append(ProbeTask(attribute, "whole"))
    return tuple(tasks)


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_ragged_tokens(
    rows,
    token_path: Path,
    offsets_path: Path,
) -> tuple[int, int]:
    offsets = [0]
    count = 0
    with token_path.open("wb") as handle:
        for ids in rows:
            values = np.asarray(ids, dtype="<i4")
            handle.write(values.tobytes())
            offsets.append(offsets[-1] + len(values))
            count += 1
    np.save(offsets_path, np.asarray(offsets, dtype="<i8"), allow_pickle=False)
    return count, offsets[-1]


def _coverage_report(
    labels: np.ndarray,
    splits: np.ndarray,
    tasks: tuple[ProbeTask, ...],
    class_names: dict[str, list[str]],
) -> dict[str, dict[str, object]]:
    report = {}
    train = splits == SPLIT_CODES["train"]
    validation = splits == SPLIT_CODES["validation"]
    for column, task in enumerate(tasks):
        train_ids = set(np.unique(labels[train, column]).tolist())
        validation_ids = set(np.unique(labels[validation, column]).tolist())
        missing = sorted(validation_ids - train_ids)
        report[task.key] = {
            "classes": len(class_names[task.key]),
            "train_classes": len(train_ids),
            "validation_classes": len(validation_ids),
            "validation_classes_missing_from_train": [
                class_names[task.key][index] for index in missing
            ],
        }
    return report


def build_probe_cache(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
    require_coverage: bool = False,
    progress: Callable[[int], None] | None = None,
) -> Path:
    """Tokenize profiles/biographies once and atomically publish a reusable cache."""

    data_root, output = Path(data_root), Path(output_dir)
    source_manifest_path = data_root / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"missing dataset manifest: {source_manifest_path}")
    if output.exists() and not force:
        raise FileExistsError(f"probe cache already exists: {output}; pass --force to rebuild")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        codec = GPT2Codec()
        profiles = list(_read_jsonl(data_root / "profiles.jsonl"))
        if not profiles:
            raise ValueError("profiles.jsonl is empty")
        profile_index = {profile["person_id"]: index for index, profile in enumerate(profiles)}
        if len(profile_index) != len(profiles):
            raise ValueError("profiles.jsonl contains duplicate person_id values")

        tasks = paper_probe_tasks()
        classes: dict[str, list[str]] = {}
        label_maps: dict[str, dict[str, int]] = {}
        for task in tasks:
            names = sorted(
                {task_label(profile, task.attribute, task.target, codec) for profile in profiles}
            )
            classes[task.key] = names
            label_maps[task.key] = {name: index for index, name in enumerate(names)}

        labels = np.empty((len(profiles), len(tasks)), dtype="<i4")
        splits = np.empty(len(profiles), dtype="u1")
        for row, profile in enumerate(profiles):
            try:
                splits[row] = SPLIT_CODES[profile["split"]]
            except KeyError as exc:
                raise ValueError(f"invalid profile split: {profile.get('split')!r}") from exc
            for column, task in enumerate(tasks):
                label = task_label(profile, task.attribute, task.target, codec)
                labels[row, column] = label_maps[task.key][label]
        np.save(staging / "profile_labels.npy", labels, allow_pickle=False)
        np.save(staging / "profile_splits.npy", splits, allow_pickle=False)

        q_rows = (
            [codec.eos, *codec.encode(profile["full_name"]), codec.eos] for profile in profiles
        )
        q_examples, q_tokens = _write_ragged_tokens(
            q_rows, staging / "q_tokens.bin", staging / "q_offsets.npy"
        )

        p_offsets = [0]
        p_positions: list[list[int]] = []
        p_profiles: list[int] = []
        p_tokens = 0
        p_examples = 0
        with (staging / "p_tokens.bin").open("wb") as token_handle:
            for row in _read_jsonl(data_root / "biographies.jsonl"):
                try:
                    mapped_profile = profile_index[row["person_id"]]
                except KeyError as exc:
                    raise ValueError(
                        f"biography references unknown person_id={row.get('person_id')!r}"
                    ) from exc
                starts = ordered_p_probe_starts(row["attribute_spans"])
                ids, positions = codec.positions_before_chars(row["text"], starts)
                values = np.asarray(ids, dtype="<i4")
                token_handle.write(values.tobytes())
                p_tokens += len(values)
                p_offsets.append(p_tokens)
                p_positions.append(positions)
                p_profiles.append(mapped_profile)
                p_examples += 1
                if progress is not None and p_examples % 10_000 == 0:
                    progress(p_examples)
        np.save(staging / "p_offsets.npy", np.asarray(p_offsets, dtype="<i8"), allow_pickle=False)
        np.save(
            staging / "p_positions.npy",
            np.asarray(p_positions, dtype="<i4").reshape(-1, len(ATTRIBUTES)),
            allow_pickle=False,
        )
        np.save(
            staging / "p_profile_indices.npy",
            np.asarray(p_profiles, dtype="<i4"),
            allow_pickle=False,
        )

        coverage = _coverage_report(labels, splits, tasks, classes)
        missing_tasks = [
            key for key, value in coverage.items() if value["validation_classes_missing_from_train"]
        ]
        if require_coverage and missing_tasks:
            raise ValueError(
                "validation labels missing from probe train split for tasks: "
                + ", ".join(missing_tasks)
            )

        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        manifest = {
            "format_version": CACHE_FORMAT_VERSION,
            "generator": "minitrain.synbios_moe.probe_cache.v2",
            "source_data": str(data_root.resolve()),
            "source_manifest": source_manifest,
            "tokenizer": "tiktoken:gpt2",
            "eos_token": codec.eos,
            "profiles": len(profiles),
            "p_examples": p_examples,
            "p_tokens": p_tokens,
            "q_examples": q_examples,
            "q_tokens": q_tokens,
            "tasks": [
                {
                    "attribute": task.attribute,
                    "target": task.target,
                    "key": task.key,
                    "class_names": classes[task.key],
                }
                for task in tasks
            ],
            "coverage": coverage,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "manifest.json"


def validate_probe_cache(
    cache_root: str | Path,
    data_root: str | Path | None = None,
    *,
    include_missing_classes: bool = True,
) -> dict[str, object]:
    """Validate cache schema, array shapes, token sizes, and label coverage."""

    root = Path(cache_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(f"unsupported probe cache format: {manifest.get('format_version')}")
    if data_root is not None:
        data_root = Path(data_root)
        current_manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("source_manifest") != current_manifest:
            raise ValueError("probe cache source manifest does not match --data")
    profiles = int(manifest["profiles"])
    p_examples = int(manifest["p_examples"])
    q_examples = int(manifest["q_examples"])
    expected_tasks = [task.key for task in paper_probe_tasks()]
    task_entries = manifest.get("tasks")
    if not isinstance(task_entries, list):
        raise ValueError("probe cache manifest tasks must be a list")
    actual_tasks = [str(task.get("key")) for task in task_entries]
    if actual_tasks != expected_tasks:
        raise ValueError(f"probe cache tasks are {actual_tasks}, expected {expected_tasks}")
    for task in task_entries:
        expected_key = f"{task.get('attribute')}_{task.get('target')}"
        class_names = task.get("class_names")
        if task.get("key") != expected_key:
            raise ValueError(f"probe cache task key is inconsistent: {task!r}")
        if (
            not isinstance(class_names, list)
            or not class_names
            or any(not isinstance(name, str) for name in class_names)
            or len(set(class_names)) != len(class_names)
        ):
            raise ValueError(f"invalid class_names for probe task {expected_key}")
    expected_shapes = {
        "profile_labels.npy": (profiles, len(manifest["tasks"])),
        "profile_splits.npy": (profiles,),
        "p_offsets.npy": (p_examples + 1,),
        "p_positions.npy": (p_examples, len(ATTRIBUTES)),
        "p_profile_indices.npy": (p_examples,),
        "q_offsets.npy": (q_examples + 1,),
    }
    for name, shape in expected_shapes.items():
        array = np.load(root / name, mmap_mode="r", allow_pickle=False)
        if array.shape != shape:
            raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    splits = np.load(root / "profile_splits.npy", mmap_mode="r", allow_pickle=False)
    if not set(np.unique(splits).tolist()).issubset(set(SPLIT_CODES.values())):
        raise ValueError("profile_splits.npy contains invalid split codes")
    labels = np.load(root / "profile_labels.npy", mmap_mode="r", allow_pickle=False)
    for column, task in enumerate(task_entries):
        column_labels = labels[:, column]
        if len(column_labels) and (
            int(column_labels.min()) < 0
            or int(column_labels.max()) >= len(task["class_names"])
        ):
            raise ValueError(f"profile_labels.npy contains invalid IDs for {task['key']}")
    profile_indices = np.load(root / "p_profile_indices.npy", mmap_mode="r", allow_pickle=False)
    if len(profile_indices) and (
        int(profile_indices.min()) < 0 or int(profile_indices.max()) >= profiles
    ):
        raise ValueError("p_profile_indices.npy contains an out-of-range profile index")
    p_positions = np.load(root / "p_positions.npy", mmap_mode="r", allow_pickle=False)
    if p_positions.shape[1] > 1 and np.any(p_positions[:, 1:] < p_positions[:, :-1]):
        raise ValueError("p_positions.npy is not ordered left-to-right")
    offsets_by_kind = {}
    for kind in ("p", "q"):
        path = root / f"{kind}_tokens.bin"
        expected_bytes = int(manifest[f"{kind}_tokens"]) * np.dtype("<i4").itemsize
        if path.stat().st_size != expected_bytes:
            raise ValueError(
                f"{path.name} has {path.stat().st_size} bytes, expected {expected_bytes}"
            )
        offsets = np.load(root / f"{kind}_offsets.npy", mmap_mode="r", allow_pickle=False)
        offsets_by_kind[kind] = offsets
        if int(offsets[0]) != 0 or int(offsets[-1]) != int(manifest[f"{kind}_tokens"]):
            raise ValueError(f"{kind}_offsets.npy does not span the token file")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError(f"{kind}_offsets.npy is not monotonic")
    p_lengths = offsets_by_kind["p"][1:] - offsets_by_kind["p"][:-1]
    if len(p_positions) and (
        np.any(p_positions < 0) or np.any(p_positions >= p_lengths[:, None])
    ):
        raise ValueError("p_positions.npy contains a position outside its biography")
    missing = {
        key: value["validation_classes_missing_from_train"]
        for key, value in manifest["coverage"].items()
        if value["validation_classes_missing_from_train"]
    }
    result = {
        "valid": True,
        "cache": str(root.resolve()),
        "source_data": manifest["source_data"],
        "profiles": profiles,
        "p_examples": p_examples,
        "q_examples": q_examples,
        "coverage_complete": not missing,
        "missing_validation_class_counts": {key: len(values) for key, values in missing.items()},
    }
    if include_missing_classes:
        result["missing_validation_classes"] = missing
    return result


class CachedProbeDataset(Dataset):
    """Read one P/Q task from the shared memory-mapped probe cache."""

    def __init__(
        self,
        root: str | Path,
        *,
        kind: str,
        attribute: str,
        target: str,
        split: str,
    ) -> None:
        if kind not in {"p", "q"} or split not in SPLIT_CODES:
            raise ValueError("kind must be p/q and split must be train/validation")
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != CACHE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported probe cache format: {self.manifest.get('format_version')}"
            )
        tasks = self.manifest["tasks"]
        matches = [
            (index, task)
            for index, task in enumerate(tasks)
            if task["attribute"] == attribute and task["target"] == target
        ]
        if not matches:
            raise ValueError(f"task is absent from cache: {attribute}/{target}")
        self.task_index, task = matches[0]
        self.class_names = list(task["class_names"])
        self.labels = np.load(self.root / "profile_labels.npy", mmap_mode="r", allow_pickle=False)
        self.splits = np.load(self.root / "profile_splits.npy", mmap_mode="r", allow_pickle=False)
        self.offsets = np.load(self.root / f"{kind}_offsets.npy", mmap_mode="r", allow_pickle=False)
        self.tokens = np.memmap(self.root / f"{kind}_tokens.bin", mode="r", dtype="<i4")
        self.kind = kind
        if kind == "p":
            self.positions = np.load(
                self.root / "p_positions.npy", mmap_mode="r", allow_pickle=False
            )
            self.profile_indices = np.load(
                self.root / "p_profile_indices.npy", mmap_mode="r", allow_pickle=False
            )
            sample_splits = self.splits[self.profile_indices]
        else:
            self.positions = None
            self.profile_indices = np.arange(len(self.splits), dtype="<i4")
            sample_splits = self.splits
        self.sample_indices = np.flatnonzero(sample_splits == SPLIT_CODES[split])

    def __len__(self) -> int:
        return len(self.sample_indices)

    def longest_items(self, limit: int) -> list[ProbeBatchItem]:
        """Return the longest examples for conservative batch-capacity benchmarks."""

        if limit <= 0:
            raise ValueError("limit must be positive")
        samples = self.sample_indices
        lengths = self.offsets[samples + 1] - self.offsets[samples]
        count = min(limit, len(samples))
        if count == 0:
            return []
        selected = np.argpartition(lengths, -count)[-count:]
        selected = selected[np.argsort(lengths[selected])[::-1]]
        return [self[int(index)] for index in selected]

    def __getitem__(self, index: int) -> ProbeBatchItem:
        sample = int(self.sample_indices[index])
        start, end = int(self.offsets[sample]), int(self.offsets[sample + 1])
        ids = self.tokens[start:end].astype(np.int64).tolist()
        profile = int(self.profile_indices[sample])
        positions = (
            self.positions[sample].astype(np.int64).tolist() if self.kind == "p" else [len(ids) - 1]
        )
        return ProbeBatchItem(ids, positions, int(self.labels[profile, self.task_index]))
