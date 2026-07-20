"""Observe MoE routing load and its correlation with biography labels."""

from __future__ import annotations

import math
from collections import Counter
from contextlib import contextmanager

import torch

from minitrain.model.transformer import MiniTransformer


@contextmanager
def capture_router_outputs(model: MiniTransformer):
    """Capture top-k assignments without changing routing or gradients."""

    captured: list[torch.Tensor] = []
    handles = []

    def hook(_module, _inputs, output):
        captured.append(output.expert_indices.detach().cpu())

    for block in model.blocks:
        if hasattr(block.ffn, "router"):
            handles.append(block.ffn.router.register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def normalized_mutual_information(assignments: list[int], labels: list[int]) -> float:
    """Normalize expert/label mutual information by label entropy."""

    if len(assignments) != len(labels) or not assignments:
        return 0.0
    joint = Counter(zip(assignments, labels))
    experts, classes, count = Counter(assignments), Counter(labels), len(labels)
    mutual_information = 0.0
    for (expert, label), frequency in joint.items():
        probability = frequency / count
        mutual_information += probability * math.log(
            probability / ((experts[expert] / count) * (classes[label] / count))
        )
    label_entropy = -sum((n / count) * math.log(n / count) for n in classes.values())
    return mutual_information / label_entropy if label_entropy else 0.0


@torch.no_grad()
def analyze_batch(
    model: MiniTransformer,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, object]:
    if not model.cfg.is_moe:
        raise ValueError("router analysis requires an MoE model")
    model.eval()
    # Forward hooks capture the router's already-selected expert IDs.  They do
    # not replace logits, reroute tokens, or participate in autograd.
    with capture_router_outputs(model) as captured:
        model.hidden_states(input_ids)
    result = {"layers": []}
    batch = torch.arange(input_ids.shape[0])[:, None]
    # Report global load balance from every routed token, then compute NMI only
    # at probe-selected positions using the top-1 expert assignment.
    for layer, routes in enumerate(captured):
        routes = routes.view(input_ids.shape[0], input_ids.shape[1], -1)
        selected = routes[batch, positions.cpu(), 0]
        flat_experts = selected.reshape(-1).tolist()
        flat_labels = labels[:, None].expand_as(selected).reshape(-1).tolist()
        load = torch.bincount(routes.reshape(-1).long(), minlength=model.cfg.num_experts)
        probabilities = load.float() / load.sum().clamp_min(1)
        entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
        result["layers"].append(
            {
                "layer": layer,
                "load": load.tolist(),
                "normalized_load_entropy": entropy / math.log(model.cfg.num_experts),
                "top1_expert_label_nmi": normalized_mutual_information(flat_experts, flat_labels),
            }
        )
    return result
