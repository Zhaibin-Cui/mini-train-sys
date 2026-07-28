"""Measure token-conditioned expert-route branching in held-out Q examples."""

import random
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Sequence

import torch

from experiments.synbios_moe.artifact_io import write_csv_atomic, write_json_atomic
from experiments.synbios_moe.mechanisms.first_token_intervention import (
    collect_q_predictions,
    prediction_row,
)
from experiments.synbios_moe.mechanisms.routing import normalized_mutual_information
from experiments.synbios_moe.pretraining.dataset import WHOLE_ATTRIBUTES
from experiments.synbios_moe.probes.model import GPT2Codec, ProbeBatchItem, collate_probe
from minitrain.model.transformer import MiniTransformer


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


def route_jaccard(left: Sequence[int], right: Sequence[int]) -> float:
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
                    t1_overlap = route_jaccard(left_routes[0][layer], right_routes[0][layer])
                    t2_overlap = route_jaccard(left_routes[1][layer], right_routes[1][layer])
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
                        **prediction_row(record),
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
                    "within_case_jaccard": route_jaccard(t1_route, t2_route),
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
    write_csv_atomic(
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
    write_csv_atomic(
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
    write_csv_atomic(
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
    write_csv_atomic(
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
    write_json_atomic(output / "summary.json", summary)
    return summary
