"""Progressively fill facts removed from each original SynBioS biography."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from experiments.synbios_moe.data import ATTRIBUTES
from experiments.synbios_moe.evaluation import _attribute_target_positions
from experiments.synbios_moe.probes import GPT2Codec
from minitrain.model.transformer import MiniTransformer
from minitrain.runtime.logger import EventLogger
from minitrain.runtime.monitoring import ProgressReporter


@dataclass(frozen=True)
class ClozeField:
    attribute: str
    start: int
    end: int
    expected: str


@dataclass(frozen=True)
class TokenClozeField:
    attribute: str
    token_start: int
    token_end: int
    expected: str


def biography_cloze_fields(row: dict) -> list[ClozeField]:
    """Return the six real fact spans in their original textual order."""

    text = str(row["text"])
    fields = [
        ClozeField(attribute, int(span[0]), int(span[1]), text[int(span[0]) : int(span[1])])
        for attribute, span in row["attribute_spans"].items()
    ]
    fields.sort(key=lambda field: field.start)
    if {field.attribute for field in fields} != set(ATTRIBUTES):
        raise ValueError("biography does not contain exactly the six SynBioS attributes")
    if any(left.end > right.start for left, right in zip(fields, fields[1:])):
        raise ValueError("biography attribute spans overlap")
    return fields


def character_similarity(predicted: str, expected: str) -> float:
    """Return case-insensitive normalized Levenshtein character similarity."""

    left = " ".join(predicted.casefold().split())
    right = " ".join(expected.casefold().split())
    if not left and not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + int(left_character != right_character),
                )
            )
        previous = current
    distance = previous[-1]
    return 1.0 - distance / max(len(left), len(right), 1)


def _tokenized_cloze(
    row: dict, codec: GPT2Codec
) -> tuple[list[int], list[TokenClozeField]]:
    """Map character holes onto the unchanged BPE sequence used in training."""

    ids, positions = _attribute_target_positions(
        codec, str(row["text"]), row["attribute_spans"]
    )
    character_fields = {field.attribute: field for field in biography_cloze_fields(row)}
    fields = []
    for attribute, indices in positions.items():
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"non-contiguous token span for {attribute}")
        fields.append(
            TokenClozeField(
                attribute,
                indices[0],
                indices[-1] + 1,
                character_fields[attribute].expected,
            )
        )
    fields.sort(key=lambda field: field.token_start)
    return ids[1:], fields


@torch.no_grad()
def _generate_values(
    model: MiniTransformer,
    prefixes: list[list[int]],
    stop_texts: list[str],
    codec: GPT2Codec,
    device: torch.device,
    *,
    max_new_tokens: int,
) -> tuple[list[str], list[list[int]], list[bool], int]:
    """Greedily continue each causal prompt until its sentence-ending period."""

    sequences = [prefix.copy() for prefix in prefixes]
    generated: list[list[int]] = [[] for _ in prefixes]
    active = list(range(len(prefixes)))
    processed_tokens = 0
    terminated = [False for _ in prefixes]
    for _ in range(max_new_tokens):
        if not active:
            break
        width = max(len(sequences[index]) for index in active)
        if width > model.cfg.seq_len:
            raise ValueError(f"cloze prompt exceeds seq_len={model.cfg.seq_len}")
        input_ids = torch.full((len(active), width), codec.eos, dtype=torch.long, device=device)
        lengths = torch.empty(len(active), dtype=torch.long, device=device)
        for row_index, example_index in enumerate(active):
            sequence = sequences[example_index]
            input_ids[row_index, : len(sequence)] = torch.tensor(sequence, device=device)
            lengths[row_index] = len(sequence)
        hidden = model.hidden_states(input_ids)
        rows = torch.arange(len(active), device=device)
        logits = model.lm_head(hidden[rows, lengths - 1])
        next_tokens = logits.argmax(dim=-1).cpu().tolist()
        processed_tokens += input_ids.numel()
        still_active = []
        for example_index, token in zip(active, next_tokens):
            sequences[example_index].append(token)
            generated[example_index].append(token)
            if stop_texts[example_index] in codec.encoding.decode(
                generated[example_index]
            ):
                terminated[example_index] = True
            else:
                still_active.append(example_index)
        active = still_active
    predictions = [
        codec.encoding.decode(tokens).split(stop_text, 1)[0].strip()
        for tokens, stop_text in zip(generated, stop_texts)
    ]
    return predictions, generated, terminated, processed_tokens


@torch.no_grad()
def evaluate_progressive_biography_cloze(
    model: MiniTransformer,
    data_root: str | Path,
    *,
    device: torch.device,
    max_biographies: int,
    start_index: int = 0,
    batch_size: int = 16,
    max_new_tokens: int = 16,
    sample_biographies: int = 12,
    logger: EventLogger | None = None,
    log_interval: int = 10,
) -> dict[str, object]:
    """Fill all six holes in each original biography, left to right.

    Generated earlier facts are inserted into the real biography text before
    later facts are generated, so mistakes can propagate exactly as they do in
    progressive causal completion. Only non-fact text comes from the source.
    """

    if start_index < 0:
        raise ValueError("start index must be non-negative")
    if max_biographies <= 0 or batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("examples, batch size, and max new tokens must be positive")
    codec = GPT2Codec()
    model.eval()
    total_batches = math.ceil(max_biographies / batch_size)
    progress = (
        ProgressReporter(
            "cloze_evaluate",
            total_batches,
            logger,
            device,
            log_interval=max(1, min(log_interval, total_batches)),
            unit="batch",
        )
        if logger is not None
        else None
    )
    correct = {attribute: 0 for attribute in ATTRIBUTES}
    totals = {attribute: 0 for attribute in ATTRIBUTES}
    similarity_sums = {attribute: 0.0 for attribute in ATTRIBUTES}
    fuzzy_thresholds = (0.5, 0.8, 0.9)
    fuzzy_correct = {
        threshold: {attribute: 0 for attribute in ATTRIBUTES}
        for threshold in fuzzy_thresholds
    }
    exact_count_histogram = {count: 0 for count in range(len(ATTRIBUTES) + 1)}
    all_exact = 0
    unterminated = 0
    biographies = 0
    processed_tokens = 0
    samples: list[dict[str, object]] = []
    started = time.perf_counter()

    def process_batch(rows: list[dict]) -> None:
        nonlocal all_exact, biographies, processed_tokens, unterminated
        states = [
            {
                "row": row,
                "original_tokens": tokenized[0],
                "fields": tokenized[1],
                "cursor": 0,
                "sequence": [codec.eos],
                "predictions": {},
                "correct": {},
            }
            for row in rows
            for tokenized in [_tokenized_cloze(row, codec)]
        ]
        for field_index in range(len(ATTRIBUTES)):
            prefixes = []
            stop_texts = []
            for state in states:
                field = state["fields"][field_index]
                state["sequence"].extend(
                    state["original_tokens"][state["cursor"] : field.token_start]
                )
                prefixes.append(state["sequence"])
                source_remainder = codec.encoding.decode(
                    state["original_tokens"][field.token_end :]
                )
                stop_texts.append(source_remainder.split(".", 1)[0] + ".")
            predictions, generated, terminated, tokens = _generate_values(
                model,
                prefixes,
                stop_texts,
                codec,
                device,
                max_new_tokens=max_new_tokens,
            )
            processed_tokens += tokens
            for state, prediction, generated_tokens, ended in zip(
                states, predictions, generated, terminated
            ):
                field = state["fields"][field_index]
                is_correct = prediction == field.expected
                similarity = character_similarity(prediction, field.expected)
                totals[field.attribute] += 1
                correct[field.attribute] += int(is_correct)
                similarity_sums[field.attribute] += similarity
                for threshold in fuzzy_thresholds:
                    fuzzy_correct[threshold][field.attribute] += int(
                        similarity >= threshold
                    )
                unterminated += int(not ended)
                state["predictions"][field.attribute] = prediction
                state["correct"][field.attribute] = is_correct
                state.setdefault("similarity", {})[field.attribute] = similarity
                state["sequence"].extend(generated_tokens)
                state["cursor"] = field.token_end
                if ended:
                    # Generation already supplied the sentence period. Skip the
                    # corresponding source delimiter before appending real
                    # non-fact tokens leading to the next hole.
                    delimiter = ""
                    while state["cursor"] < len(state["original_tokens"]):
                        token = state["original_tokens"][state["cursor"]]
                        state["cursor"] += 1
                        delimiter += codec.encoding.decode([token])
                        if "." in delimiter:
                            break
        for state in states:
            state["sequence"].extend(state["original_tokens"][state["cursor"] :])
            filled_text = codec.encoding.decode(state["sequence"][1:])
            exact_fields = sum(bool(value) for value in state["correct"].values())
            exact_count_histogram[exact_fields] += 1
            all_exact += int(exact_fields == len(ATTRIBUTES))
            biographies += 1
            if len(samples) < sample_biographies:
                samples.append(
                    {
                        "person_id": state["row"]["person_id"],
                        "original_text": state["row"]["text"],
                        "filled_text": filled_text,
                        "expected": {
                            field.attribute: field.expected for field in state["fields"]
                        },
                        "predicted": state["predictions"],
                        "correct": state["correct"],
                        "character_similarity": state["similarity"],
                        "exact_fields": exact_fields,
                    }
                )

    pending: list[dict] = []
    with (Path(data_root) / "biographies.jsonl").open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index < start_index:
                continue
            if biographies + len(pending) >= max_biographies:
                break
            pending.append(json.loads(line))
            if len(pending) < batch_size:
                continue
            batch_items = len(pending)
            tokens_before = processed_tokens
            process_batch(pending)
            pending = []
            if progress is not None:
                progress.update(
                    math.ceil(biographies / batch_size),
                    items=batch_items,
                    tokens=processed_tokens - tokens_before,
                    metrics={
                        "field_accuracy_running": sum(correct.values())
                        / max(sum(totals.values()), 1),
                        "all_fields_accuracy_running": all_exact / max(biographies, 1),
                    },
                )
        if pending:
            batch_items = len(pending)
            tokens_before = processed_tokens
            process_batch(pending)
            if progress is not None:
                progress.update(
                    math.ceil(biographies / batch_size),
                    items=batch_items,
                    tokens=processed_tokens - tokens_before,
                    metrics={
                        "field_accuracy_running": sum(correct.values())
                        / max(sum(totals.values()), 1),
                        "all_fields_accuracy_running": all_exact / max(biographies, 1),
                    },
                )
    elapsed = time.perf_counter() - started
    return {
        "protocol": "progressive_original_biography_cloze_greedy",
        "protocol_note": (
            "the six source fact spans are removed and generated in their original text order; "
            "earlier predictions replace earlier holes before later fields are generated"
        ),
        "start_index": start_index,
        "biographies": biographies,
        "fields": biographies * len(ATTRIBUTES),
        "attribute_accuracy": {
            attribute: correct[attribute] / max(totals[attribute], 1)
            for attribute in ATTRIBUTES
        },
        "attribute_mean_character_similarity": {
            attribute: similarity_sums[attribute] / max(totals[attribute], 1)
            for attribute in ATTRIBUTES
        },
        "mean_character_similarity": sum(similarity_sums.values())
        / max(sum(totals.values()), 1),
        "fuzzy_accuracy_by_threshold": {
            str(threshold): {
                attribute: fuzzy_correct[threshold][attribute]
                / max(totals[attribute], 1)
                for attribute in ATTRIBUTES
            }
            for threshold in fuzzy_thresholds
        },
        "micro_fuzzy_accuracy_by_threshold": {
            str(threshold): sum(fuzzy_correct[threshold].values())
            / max(sum(totals.values()), 1)
            for threshold in fuzzy_thresholds
        },
        "micro_field_accuracy": sum(correct.values()) / max(sum(totals.values()), 1),
        "biography_all_fields_accuracy": all_exact / max(biographies, 1),
        "mean_correct_fields_per_biography": sum(
            count * examples for count, examples in exact_count_histogram.items()
        )
        / max(biographies, 1),
        "biography_correct_field_count_histogram": {
            str(count): examples for count, examples in exact_count_histogram.items()
        },
        "correct_fields": correct,
        "total_fields": totals,
        "unterminated_fields": unterminated,
        "elapsed_seconds": elapsed,
        "biographies_per_second": biographies / max(elapsed, 1e-9),
        "model_tokens_processed": processed_tokens,
        "samples": samples,
        "monitoring": progress.summary() if progress is not None else {},
    }


def summarize_progressive_cloze_results(paths: list[str | Path]) -> dict[str, object]:
    """Merge disjoint cloze shards and validate their aggregate coverage."""

    results = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if not results:
        raise ValueError("at least one cloze result is required")
    ranges = sorted(
        (int(result["start_index"]), int(result["biographies"])) for result in results
    )
    for (start, count), (next_start, _) in zip(ranges, ranges[1:]):
        if start + count != next_start:
            raise ValueError("cloze shard ranges overlap or contain a gap")
    correct = {
        attribute: sum(int(result["correct_fields"][attribute]) for result in results)
        for attribute in ATTRIBUTES
    }
    totals = {
        attribute: sum(int(result["total_fields"][attribute]) for result in results)
        for attribute in ATTRIBUTES
    }
    similarity_sums = {
        attribute: sum(
            float(result["attribute_mean_character_similarity"][attribute])
            * int(result["total_fields"][attribute])
            for result in results
        )
        for attribute in ATTRIBUTES
    }
    thresholds = tuple(results[0]["micro_fuzzy_accuracy_by_threshold"])
    fuzzy_counts = {
        threshold: {
            attribute: sum(
                round(
                    float(result["fuzzy_accuracy_by_threshold"][threshold][attribute])
                    * int(result["total_fields"][attribute])
                )
                for result in results
            )
            for attribute in ATTRIBUTES
        }
        for threshold in thresholds
    }
    histogram = {
        str(count): sum(
            int(result["biography_correct_field_count_histogram"][str(count)])
            for result in results
        )
        for count in range(len(ATTRIBUTES) + 1)
    }
    biographies = sum(int(result["biographies"]) for result in results)
    fields = sum(totals.values())
    elapsed = max(float(result["elapsed_seconds"]) for result in results)
    return {
        "protocol": results[0]["protocol"],
        "protocol_note": results[0]["protocol_note"],
        "shards": len(results),
        "ranges": [
            {"start_index": start, "biographies": count} for start, count in ranges
        ],
        "biographies": biographies,
        "fields": fields,
        "attribute_accuracy": {
            attribute: correct[attribute] / max(totals[attribute], 1)
            for attribute in ATTRIBUTES
        },
        "attribute_mean_character_similarity": {
            attribute: similarity_sums[attribute] / max(totals[attribute], 1)
            for attribute in ATTRIBUTES
        },
        "micro_field_accuracy": sum(correct.values()) / max(fields, 1),
        "mean_character_similarity": sum(similarity_sums.values()) / max(fields, 1),
        "fuzzy_accuracy_by_threshold": {
            threshold: {
                attribute: fuzzy_counts[threshold][attribute]
                / max(totals[attribute], 1)
                for attribute in ATTRIBUTES
            }
            for threshold in thresholds
        },
        "micro_fuzzy_accuracy_by_threshold": {
            threshold: sum(fuzzy_counts[threshold].values()) / max(fields, 1)
            for threshold in thresholds
        },
        "biography_all_fields_accuracy": histogram[str(len(ATTRIBUTES))]
        / max(biographies, 1),
        "mean_correct_fields_per_biography": sum(
            count * histogram[str(count)] for count in range(len(ATTRIBUTES) + 1)
        )
        / max(biographies, 1),
        "biography_correct_field_count_histogram": histogram,
        "correct_fields": correct,
        "total_fields": totals,
        "unterminated_fields": sum(int(result["unterminated_fields"]) for result in results),
        "parallel_wall_seconds": elapsed,
        "aggregate_biographies_per_second": biographies / max(elapsed, 1e-9),
        "model_tokens_processed": sum(
            int(result["model_tokens_processed"]) for result in results
        ),
        "samples": [sample for result in results for sample in result["samples"]],
        "checkpoint": results[0]["checkpoint"],
        "shard_results": [str(Path(path).resolve()) for path in paths],
    }
