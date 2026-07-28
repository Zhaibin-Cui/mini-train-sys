"""Evaluate an unchanged Q-whole head after inserting its label first token."""

from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from experiments.synbios_moe.artifact_io import write_csv_atomic, write_json_atomic
from experiments.synbios_moe.mechanisms.q_readout import (
    collect_q_predictions,
    prediction_row,
    task_class_names,
)
from experiments.synbios_moe.pretraining.dataset import WHOLE_ATTRIBUTES
from experiments.synbios_moe.probes.model import GPT2Codec, ProbeBatchItem, collate_probe
from minitrain.model.transformer import MiniTransformer


def insert_oracle_first_token(input_ids: Sequence[int], token_id: int, eos_id: int) -> list[int]:
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
    axes[0].bar([value + width / 2 for value in x], after, width, label="+ label first token")
    axes[0].set_ylabel("Q-whole held-out accuracy (%)")
    axes[0].set_xticks(x, attributes, rotation=25, ha="right")
    axes[0].set_ylim(0, 100)
    axes[0].legend()
    axes[0].set_title("First-token label intervention")
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
                row = prediction_row(record)
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
    summary = {
        "protocol": "q_whole_oracle_first_token_v1",
        "intervention": "[EOS, name, label_t1, EOS], read final EOS with unchanged Q-whole head",
        "split": "person-held-out validation",
        "parameters_updated": False,
        "data": str(Path(data_root).resolve()),
        "probe_cache": str(Path(cache_root).resolve()),
        "probe_dir": str(Path(probe_dir).resolve()),
        "backbone_checkpoint": (
            str(Path(backbone_checkpoint).resolve()) if backbone_checkpoint is not None else None
        ),
        "overall": summarize_oracle_rows(all_rows),
        "attributes": summary_rows,
    }
    write_csv_atomic(output / "records.csv", all_rows)
    write_csv_atomic(output / "summary.csv", summary_rows)
    write_json_atomic(output / "summary.json", summary)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    _plot_oracle(summary_rows, figures / "accuracy_before_after.png")
    return summary
