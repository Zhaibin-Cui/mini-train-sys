"""Teacher-forced token accuracy restricted to the six synthetic facts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from experiments.synbios_moe.data import ATTRIBUTES
from experiments.synbios_moe.probes import GPT2Codec
from minitrain.model.transformer import MiniTransformer
from minitrain.runtime.logger import EventLogger
from minitrain.runtime.monitoring import ProgressReporter


def _attribute_target_positions(
    codec: GPT2Codec, text: str, spans: dict[str, list[int]]
) -> tuple[list[int], dict[str, list[int]]]:
    # Convert stored character spans and tokenizer output to UTF-8 byte ranges;
    # overlap then identifies every BPE target token belonging to a fact.
    token_ids = codec.encode(text)
    byte_ranges = []
    cursor = 0
    for token in token_ids:
        end = cursor + len(codec.encoding.decode_single_token_bytes(token))
        byte_ranges.append((cursor, end))
        cursor = end
    byte_spans = {
        attribute: (
            len(text[: span[0]].encode("utf-8")),
            len(text[: span[1]].encode("utf-8")),
        )
        for attribute, span in spans.items()
    }
    positions = {}
    for attribute in ATTRIBUTES:
        start, end = byte_spans[attribute]
        # `ids` prepends EOS. Token j in token_ids is therefore target j+1,
        # predicted at causal-logit position j.
        positions[attribute] = [
            index
            for index, (token_start, token_end) in enumerate(byte_ranges)
            if token_end > start and token_start < end
        ]
    return [codec.eos, *token_ids], positions


@torch.no_grad()
def evaluate_attribute_tokens(
    model: MiniTransformer,
    data_root: str | Path,
    *,
    device: torch.device,
    max_biographies: int = 10_000,
    batch_size: int = 8,
    logger: EventLogger | None = None,
    log_interval: int = 10,
) -> dict[str, object]:
    """Measure teacher-forced next-token accuracy only on the six attributes."""

    codec = GPT2Codec()
    examples = []
    with (Path(data_root) / "biographies.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids, positions = _attribute_target_positions(codec, row["text"], row["attribute_spans"])
            if len(ids) > model.cfg.seq_len:
                raise ValueError(
                    f"biography has {len(ids)} tokens, exceeding seq_len={model.cfg.seq_len}"
                )
            examples.append((ids, positions))
            if len(examples) >= max_biographies:
                break

    # Batch full biographies for the model, but score only positions mapped to
    # attribute spans.  Padding tokens never enter any attribute position list.
    totals = {attribute: 0 for attribute in ATTRIBUTES}
    correct = {attribute: 0 for attribute in ATTRIBUTES}
    model.eval()
    total_batches = math.ceil(len(examples) / batch_size) if examples else 0
    progress = (
        ProgressReporter(
            "evaluate",
            total_batches,
            logger,
            device,
            log_interval=max(1, min(log_interval, total_batches)),
            unit="batch",
        )
        if logger is not None and total_batches > 0
        else None
    )
    for offset in range(0, len(examples), batch_size):
        batch = examples[offset : offset + batch_size]
        width = max(len(ids) for ids, _ in batch) - 1
        input_ids = torch.full((len(batch), width), codec.eos, dtype=torch.long)
        target_ids = torch.full((len(batch), width), codec.eos, dtype=torch.long)
        for row_index, (ids, _) in enumerate(batch):
            input_ids[row_index, : len(ids) - 1] = torch.tensor(ids[:-1])
            target_ids[row_index, : len(ids) - 1] = torch.tensor(ids[1:])
        _, logits = model(input_ids.to(device))
        predicted = logits.argmax(dim=-1).cpu()
        for row_index, (_, positions) in enumerate(batch):
            for attribute, indices in positions.items():
                if indices:
                    index = torch.tensor(indices)
                    correct[attribute] += int(
                        (predicted[row_index, index] == target_ids[row_index, index]).sum()
                    )
                    totals[attribute] += len(indices)
        if progress is not None:
            progress.update(
                offset // batch_size + 1,
                items=len(batch),
                tokens=input_ids.numel(),
                metrics={
                    "micro_accuracy_running": sum(correct.values())
                    / max(sum(totals.values()), 1)
                },
            )
    accuracies = {
        attribute: correct[attribute] / totals[attribute] if totals[attribute] else 0.0
        for attribute in ATTRIBUTES
    }
    return {
        "biographies": len(examples),
        "attribute_token_accuracy": accuracies,
        "micro_accuracy": sum(correct.values()) / max(sum(totals.values()), 1),
        "correct_tokens": correct,
        "total_tokens": totals,
        "monitoring": progress.summary() if progress is not None else {},
    }
