"""Paper-style P/Q probes for facts stored by the synthetic biographies."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments.synbios_moe.data import ATTRIBUTES
from experiments.synbios_moe.probe_checkpoint import (
    load_probe_recovery,
    save_probe_recovery,
)
from minitrain.model.transformer import MiniTransformer
from minitrain.runtime.logger import EventLogger
from minitrain.runtime.monitoring import GpuUtilizationMonitor, ProgressReporter


GPT2_EOS_TOKEN = 50256


def linear_decay_fraction(step: int, steps: int) -> float:
    """Return paper-style no-warmup decay: 1 at step 0 and 0 at the final step."""

    if steps <= 0 or not 0 <= step < steps:
        raise ValueError("step must be in [0, steps)")
    return 1.0 if steps == 1 else 1.0 - step / (steps - 1)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class GPT2Codec:
    """Centralize GPT-2 text/token and character/byte position conversion."""

    def __init__(self) -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError("Install the 'data' extra for the bioS experiment") from exc
        self.encoding = tiktoken.get_encoding("gpt2")
        self.eos = self.encoding.eot_token
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, allowed_special=set())

    def position_before_char(self, text: str, char_index: int) -> tuple[list[int], int]:
        ids, positions = self.positions_before_chars(text, [char_index])
        return ids, positions[0]

    def positions_before_chars(
        self, text: str, char_indices: list[int]
    ) -> tuple[list[int], list[int]]:
        # Python spans count Unicode code points, whereas byte-level BPE token
        # boundaries count UTF-8 bytes.  Convert both into the same byte space.
        ids = self.encode(text)
        byte_starts = [len(text[:index].encode("utf-8")) for index in char_indices]
        token_ends = []
        consumed = 0
        for token in ids:
            consumed += len(self.encoding.decode_single_token_bytes(token))
            token_ends.append(consumed)
        # An EOS is prepended, so `complete` is exactly the index of the token
        # ending immediately before the attribute byte span.
        positions = [sum(end <= start for end in token_ends) for start in byte_starts]
        return [self.eos, *ids], positions


def ordered_p_probe_starts(attribute_spans: Mapping[str, Sequence[int]]) -> list[int]:
    """Return paper-style P0..P5 character positions in left-to-right text order.

    The positional axis follows successive facts in the rendered biography,
    rather than the fixed semantic attribute order. This distinction matters
    when sentence permutation changes which attribute appears first.
    """

    spans: list[tuple[int, int, str]] = []
    for attribute in ATTRIBUTES:
        try:
            raw_span = attribute_spans[attribute]
            start, end = raw_span
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid span for attribute {attribute!r}") from exc
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            raise ValueError(f"invalid span for attribute {attribute!r}: {raw_span!r}")
        spans.append((start, end, attribute))

    spans.sort(key=lambda item: item[0])
    for previous, current in zip(spans, spans[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"overlapping attribute spans: {previous[2]!r} and {current[2]!r}")
    return [start for start, _, _ in spans]


def task_label(profile: dict, attribute: str, target: str, codec: GPT2Codec) -> str:
    value = str(profile[attribute])
    if target == "whole":
        return value
    if attribute == "birth_date":
        return value.split(" ", 1)[0]
    ids = codec.encode(" " + value)
    return str(ids[0])


@dataclass(frozen=True)
class ProbeBatchItem:
    input_ids: list[int]
    positions: list[int]
    label: int


class PProbeDataset(Dataset):
    """Probe factual information at all six pre-attribute hidden positions."""

    def __init__(self, root: str | Path, *, attribute: str, target: str, split: str) -> None:
        if attribute not in ATTRIBUTES or target not in {"first", "whole"}:
            raise ValueError("invalid P-probe task")
        if attribute == "birth_date" and target == "whole":
            raise ValueError("the paper does not use whole-date classification")
        root, codec = Path(root), GPT2Codec()
        profiles = {row["person_id"]: row for row in _read_jsonl(root / "profiles.jsonl")}
        labels = sorted({task_label(p, attribute, target, codec) for p in profiles.values()})
        label_to_id = {value: index for index, value in enumerate(labels)}
        self.class_names = labels
        self.items: list[ProbeBatchItem] = []
        # One biography yields six observation positions.  The supervised label
        # is the selected fact of that biography's underlying person.
        for row in _read_jsonl(root / "biographies.jsonl"):
            profile = profiles[row["person_id"]]
            if profile["split"] != split:
                continue
            starts = ordered_p_probe_starts(row["attribute_spans"])
            ids, positions = codec.positions_before_chars(row["text"], starts)
            self.items.append(
                ProbeBatchItem(
                    ids, positions, label_to_id[task_label(profile, attribute, target, codec)]
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> ProbeBatchItem:
        return self.items[index]


class QProbeDataset(Dataset):
    """Probe a fact from the representation immediately after a person's name."""

    def __init__(self, root: str | Path, *, attribute: str, target: str, split: str) -> None:
        root, codec = Path(root), GPT2Codec()
        profiles = _read_jsonl(root / "profiles.jsonl")
        labels = sorted({task_label(p, attribute, target, codec) for p in profiles})
        label_to_id = {value: index for index, value in enumerate(labels)}
        self.class_names = labels
        self.items = []
        for profile in profiles:
            if profile["split"] != split:
                continue
            ids = [codec.eos, *codec.encode(profile["full_name"]), codec.eos]
            self.items.append(
                ProbeBatchItem(
                    ids, [len(ids) - 1], label_to_id[task_label(profile, attribute, target, codec)]
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> ProbeBatchItem:
        return self.items[index]


def collate_probe(items: list[ProbeBatchItem]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad variable-length inputs while preserving requested positions."""

    if not items:
        raise ValueError("cannot collate an empty probe batch")
    width = max(len(item.input_ids) for item in items)
    input_ids = torch.full((len(items), width), GPT2_EOS_TOKEN, dtype=torch.long)
    for row, item in enumerate(items):
        input_ids[row, : len(item.input_ids)] = torch.tensor(item.input_ids)
    return (
        input_ids,
        torch.tensor([item.positions for item in items]),
        torch.tensor([item.label for item in items]),
    )


class LowRankEmbeddingDelta(nn.Module):
    """Trainable rank-r input perturbation used while the backbone stays frozen."""

    def __init__(self, vocab_size: int, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.a = nn.Embedding(vocab_size, rank)
        self.b = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.a.weight, std=0.02)
        nn.init.zeros_(self.b.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(input_ids))


class AttributeProbe(nn.Module):
    """Frozen backbone plus low-rank input delta, normalization, and classifier."""

    def __init__(
        self, backbone: MiniTransformer, num_classes: int, *, rank: int, kind: str
    ) -> None:
        super().__init__()
        self.backbone = backbone
        # Only the delta/normalizer/classifier learn; backbone knowledge remains
        # fixed so accuracy measures extractability rather than fine-tuning.
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        self.delta = LowRankEmbeddingDelta(backbone.cfg.vocab_size, backbone.cfg.hidden_size, rank)
        self.normalizer: nn.Module = (
            nn.LayerNorm(backbone.cfg.hidden_size)
            if kind == "p"
            else nn.BatchNorm1d(backbone.cfg.hidden_size)
        )
        self.classifier = nn.Linear(backbone.cfg.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone.hidden_states(input_ids, embedding_delta=self.delta(input_ids))
        batch = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        selected = hidden[batch, positions]
        shape = selected.shape
        normalized = self.normalizer(selected.reshape(-1, shape[-1]))
        return self.classifier(normalized).view(shape[0], shape[1], -1)


@torch.no_grad()
def evaluate(
    probe: AttributeProbe,
    loader: DataLoader,
    device: torch.device,
    *,
    progress: ProgressReporter | None = None,
) -> list[float]:
    probe.eval()
    correct: torch.Tensor | None = None
    total = 0
    for batch_index, (input_ids, positions, labels) in enumerate(loader, start=1):
        logits = probe(
            input_ids.to(device, non_blocking=True),
            positions.to(device, non_blocking=True),
        )
        predictions = logits.argmax(-1).cpu()
        batch_correct = (predictions == labels[:, None]).sum(0)
        correct = batch_correct if correct is None else correct + batch_correct
        total += labels.numel()
        if progress is not None:
            accuracy_by_position = [float(value / max(total, 1)) for value in correct]
            progress.update(
                batch_index,
                items=labels.numel(),
                tokens=input_ids.numel(),
                metrics={
                    "accuracy_running": float(correct.sum() / max(total * correct.numel(), 1)),
                    "accuracy_by_position_running": accuracy_by_position,
                    **{
                        f"accuracy_position_{index}_running": value
                        for index, value in enumerate(accuracy_by_position)
                    },
                },
            )
    if correct is None:
        correct = torch.zeros(1)
    return [float(value / max(total, 1)) for value in correct]


def train_probe(
    probe: AttributeProbe,
    train_data: Dataset,
    validation_data: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    steps: int,
    lr: float = 1e-3,
    weight_decay: float = 0.3,
    eps: float = 1e-6,
    seed: int = 1337,
    logger: EventLogger | None = None,
    log_interval: int | None = None,
    recovery_path: str | Path | None = None,
    checkpoint_interval_steps: int | None = None,
    recovery_metadata: dict[str, object] | None = None,
    resume: bool = True,
    evaluate_train: bool = False,
    evaluate_validation: bool = True,
    evaluation_batch_size: int | None = None,
) -> dict[str, object]:
    """Optimize one probe with monitored, resumable, deterministic training."""

    if steps <= 0:
        raise ValueError("probe steps must be positive")
    if batch_size <= 0:
        raise ValueError("probe batch_size must be positive")
    if checkpoint_interval_steps is not None and checkpoint_interval_steps <= 0:
        raise ValueError("checkpoint_interval_steps must be positive or None")
    if len(train_data) == 0 or len(validation_data) == 0:
        raise ValueError("probe train and validation datasets must both be non-empty")

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_probe,
        generator=generator,
        drop_last=len(train_data) >= batch_size and batch_size > 1,
        pin_memory=device.type == "cuda",
    )
    eval_batch_size = evaluation_batch_size or batch_size
    validation_loader = DataLoader(
        validation_data,
        batch_size=eval_batch_size,
        collate_fn=collate_probe,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        (p for p in probe.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
    )
    probe.to(device).train()
    interval = log_interval or max(steps // 100, 1)
    train_progress = (
        ProgressReporter(
            "probe_train",
            steps,
            logger,
            device,
            log_interval=interval,
            unit="step",
        )
        if logger is not None
        else None
    )
    recovery = Path(recovery_path) if recovery_path is not None else None
    metadata = dict(recovery_metadata or {})
    start_step = 0
    losses: list[dict[str, object]] = []
    epoch_generator_state = generator.get_state()
    batches_consumed_in_epoch = 0
    if recovery is not None and resume and recovery.is_file():
        restored = load_probe_recovery(
            recovery,
            probe=probe,
            optimizer=optimizer,
            data_generator=generator,
            expected_metadata=metadata,
        )
        start_step = int(restored["step"])
        if start_step > steps:
            raise ValueError(f"probe recovery step {start_step} exceeds requested steps {steps}")
        losses = list(restored["loss_curve"])
        epoch_generator_state = restored["epoch_generator_state"]
        batches_consumed_in_epoch = int(restored["batches_consumed_in_epoch"])

    # Recreate the current epoch's permutation and skip only already-consumed
    # batches. This preserves shuffled data order exactly across interruption.
    generator.set_state(epoch_generator_state)
    iterator = iter(train_loader)
    for _ in range(batches_consumed_in_epoch):
        try:
            next(iterator)
        except StopIteration as exc:
            raise ValueError("probe recovery batch offset exceeds its saved epoch") from exc

    trainable_parameters = [p for p in probe.parameters() if p.requires_grad]
    interval_started = time.perf_counter()
    interval_data_wait_seconds = 0.0
    interval_steps = 0
    interval_correct: torch.Tensor | None = None
    interval_examples = 0
    checkpoints_saved = 0
    utilization = GpuUtilizationMonitor(device)
    utilization.start()
    for step in range(start_step, steps):
        fetch_started = time.perf_counter()
        try:
            input_ids, positions, labels = next(iterator)
        except StopIteration:
            epoch_generator_state = generator.get_state()
            iterator = iter(train_loader)
            batches_consumed_in_epoch = 0
            input_ids, positions, labels = next(iterator)
        batches_consumed_in_epoch += 1
        interval_data_wait_seconds += time.perf_counter() - fetch_started
        interval_steps += 1
        fraction = linear_decay_fraction(step, steps)
        for group in optimizer.param_groups:
            group["lr"] = lr * fraction
        logits = probe(
            input_ids.to(device, non_blocking=True),
            positions.to(device, non_blocking=True),
        )
        expanded = labels.to(device, non_blocking=True)[:, None].expand(-1, logits.shape[1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), expanded.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        predictions = logits.detach().argmax(-1)
        batch_correct = (predictions == expanded).sum(0)
        interval_correct = (
            batch_correct if interval_correct is None else interval_correct + batch_correct
        )
        interval_examples += labels.numel()
        should_report = step == 0 or (step + 1) % interval == 0 or step + 1 == steps
        grad_norm = None
        if should_report:
            gradient_norms = [
                torch.linalg.vector_norm(parameter.grad.detach().float())
                for parameter in trainable_parameters
                if parameter.grad is not None
            ]
            grad_norm = float(
                torch.linalg.vector_norm(torch.stack(gradient_norms))
                if gradient_norms
                else torch.zeros((), device=device)
            )
        optimizer.step()
        if should_report:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            elapsed = max(now - interval_started, 1e-12)
            accuracy_by_position = [
                float(value / max(interval_examples, 1)) for value in interval_correct
            ]
            metrics = {
                "loss": float(loss.detach()),
                "lr": lr * fraction,
                "accuracy": sum(accuracy_by_position) / max(len(accuracy_by_position), 1),
                "accuracy_by_position": accuracy_by_position,
                "grad_norm": grad_norm,
                "data_wait_ms": 1000.0 * interval_data_wait_seconds / interval_steps,
                "data_wait_percent": 100.0 * interval_data_wait_seconds / elapsed,
                "step_time_ms": 1000.0 * elapsed / interval_steps,
                "sequence_length": input_ids.shape[1],
                **{
                    f"accuracy_position_{index}": value
                    for index, value in enumerate(accuracy_by_position)
                },
            }
            metrics.update(utilization.read_interval())
            losses.append({"step": step + 1, **metrics})
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"non-finite probe loss at step {step + 1}")
            interval_started = now
            interval_data_wait_seconds = 0.0
            interval_steps = 0
            interval_correct = None
            interval_examples = 0
        if train_progress is not None:
            train_progress.update(
                step + 1,
                metrics=(
                    metrics
                    if should_report
                    else {"loss": float(loss.detach()), "lr": lr * fraction}
                ),
                items=labels.numel(),
                tokens=input_ids.numel(),
            )
        completed_step = step + 1
        if (
            recovery is not None
            and checkpoint_interval_steps is not None
            and completed_step < steps
            and completed_step % checkpoint_interval_steps == 0
        ):
            save_probe_recovery(
                recovery,
                probe=probe,
                optimizer=optimizer,
                step=completed_step,
                loss_curve=losses,
                data_generator=generator,
                epoch_generator_state=epoch_generator_state,
                batches_consumed_in_epoch=batches_consumed_in_epoch,
                metadata=metadata,
            )
            checkpoints_saved += 1
            if logger is not None:
                logger.log_event(
                    {
                        "event": "probe_checkpoint",
                        "step": completed_step,
                        "steps_total": steps,
                        "path": str(recovery.resolve()),
                    }
                )
    utilization.close()

    validation_progress = (
        ProgressReporter(
            "probe_validation",
            len(validation_loader),
            logger,
            device,
            log_interval=max(1, min(interval, len(validation_loader))),
            unit="batch",
        )
        if logger is not None and evaluate_validation and len(validation_loader) > 0
        else None
    )
    validation_accuracy = (
        evaluate(probe, validation_loader, device, progress=validation_progress)
        if evaluate_validation
        else None
    )
    train_accuracy = None
    train_evaluation_progress = None
    if evaluate_train:
        train_evaluation_loader = DataLoader(
            train_data,
            batch_size=eval_batch_size,
            collate_fn=collate_probe,
            pin_memory=device.type == "cuda",
        )
        train_evaluation_progress = (
            ProgressReporter(
                "probe_train_evaluation",
                len(train_evaluation_loader),
                logger,
                device,
                log_interval=max(1, min(interval, len(train_evaluation_loader))),
                unit="batch",
            )
            if logger is not None
            else None
        )
        train_accuracy = evaluate(
            probe, train_evaluation_loader, device, progress=train_evaluation_progress
        )
    return {
        "validation_accuracy": validation_accuracy,
        "train_accuracy": train_accuracy,
        "loss_curve": losses,
        "batch_size": batch_size,
        "evaluation_batch_size": eval_batch_size,
        "steps": steps,
        "resumed_from_step": start_step,
        "recovery_checkpoints_saved": checkpoints_saved,
        "trainable_parameters": sum(p.numel() for p in trainable_parameters),
        "monitoring": {
            "train": train_progress.summary() if train_progress is not None else {},
            "train_evaluation": (
                train_evaluation_progress.summary() if train_evaluation_progress is not None else {}
            ),
            "validation": (
                validation_progress.summary() if validation_progress is not None else {}
            ),
        },
    }


def active_parameter_estimate(model: MiniTransformer) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    if not model.cfg.is_moe:
        return {"total": total, "active_estimate": total}
    expert_total = sum(
        block.ffn.gate_up_proj.numel() + block.ffn.down_proj.numel() for block in model.blocks
    )
    active_experts = expert_total * model.cfg.experts_per_token // model.cfg.num_experts
    return {"total": total, "active_estimate": total - expert_total + active_experts}
