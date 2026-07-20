from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from minitrain.model.config import ModelConfig
from minitrain.model.ops import OpsBackend


@dataclass
class RouterOutput:
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    auxiliary_loss: torch.Tensor
    z_loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


class TopKRouter(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.empty(cfg.num_experts, cfg.hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def _capacity_mask(
        self, expert_indices: torch.Tensor, expert_weights: torch.Tensor
    ) -> torch.Tensor:
        # Kept for reference and direct experiments only. The production forward
        # path currently uses dropless routing and does not call this helper,
        # because zeroing weights after dispatch does not save MoE GEMM work.
        factor = self.cfg.expert_capacity_factor
        if factor is None:
            return torch.ones_like(expert_weights, dtype=torch.bool)

        tokens, top_k = expert_indices.shape
        capacity = max(
            self.cfg.expert_min_capacity,
            math.ceil(factor * tokens * top_k / self.cfg.num_experts),
        )
        flat_experts = expert_indices.flatten().long()
        flat_weights = expert_weights.flatten()

        # Sort by descending weight first, then group by expert with a stable
        # sort. Routes within each expert therefore remain confidence-ordered.
        by_weight = torch.argsort(flat_weights, descending=True, stable=True)
        by_expert = torch.argsort(flat_experts[by_weight], stable=True)
        route_order = by_weight[by_expert]
        sorted_experts = flat_experts[route_order]
        counts = torch.bincount(sorted_experts, minlength=self.cfg.num_experts)
        expert_offsets = counts.cumsum(0) - counts
        rank_within_expert = torch.arange(
            route_order.numel(), device=route_order.device
        ) - torch.repeat_interleave(expert_offsets, counts)

        keep = torch.zeros_like(flat_weights, dtype=torch.bool)
        keep[route_order] = rank_within_expert < capacity
        return keep.view_as(expert_weights)

    def forward(self, hidden_states: torch.Tensor, ops: OpsBackend) -> RouterOutput:
        router_input = hidden_states.float()
        if self.training and self.cfg.router_jitter_noise:
            noise = torch.empty_like(router_input).uniform_(
                1.0 - self.cfg.router_jitter_noise,
                1.0 + self.cfg.router_jitter_noise,
            )
            router_input = router_input * noise

        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            logits = F.linear(router_input, self.weight.float())
            route = ops.router_postprocess(
                logits,
                self.cfg.experts_per_token,
                normalize=self.cfg.router_normalize_topk,
            )
            expert_weights = route.expert_weights
            expert_indices = route.expert_indices

            route_counts = torch.bincount(
                expert_indices.flatten().long(), minlength=self.cfg.num_experts
            )
            tokens_per_expert = route_counts.float() / expert_indices.numel()
            # The assignment fraction is non-differentiable; gradients enter
            # through the full-softmax probability mean returned by the op.
            auxiliary_loss = self.cfg.num_experts * torch.sum(
                tokens_per_expert * route.probability_per_expert
            )

            # Capacity masking is intentionally disabled for now. The fused MoE
            # path still dispatches and computes all T*K routes, so applying the
            # mask here would add two argsorts and change model semantics without
            # reducing routing, communication, or GEMM work. Keep the old path
            # commented out until capacity-aware compaction is implemented in the
            # routing metadata and forward/backward kernels.
            # capacity_mask = self._capacity_mask(expert_indices, expert_weights)
            # expert_weights = expert_weights * capacity_mask
            # if self.cfg.router_normalize_topk:
            #     denominator = expert_weights.sum(dim=-1, keepdim=True)
            #     expert_weights = torch.where(
            #         denominator > 0,
            #         expert_weights / denominator.clamp_min(torch.finfo(expert_weights.dtype).eps),
            #         expert_weights,
            #     )
            #
            # accepted_indices = expert_indices[capacity_mask].long()
            # accepted_per_expert = torch.bincount(
            #     accepted_indices, minlength=self.cfg.num_experts
            # ).float()
            accepted_per_expert = route_counts.float()
            load_fraction = accepted_per_expert / max(expert_indices.numel(), 1)
            uniform_fraction = 1.0 / self.cfg.num_experts
            metrics = {
                "moe/router_entropy": route.entropy,
                "moe/router_entropy_normalized": (
                    route.entropy
                    / (math.log(self.cfg.num_experts) if self.cfg.num_experts > 1 else 1.0)
                ).detach(),
                # "moe/dropped_route_fraction": (1.0 - capacity_mask.float().mean()).detach(),
                "moe/dropped_route_fraction": expert_weights.new_zeros(()).detach(),
                "moe/max_expert_load": accepted_per_expert.max().detach(),
                "moe/min_expert_load": accepted_per_expert.min().detach(),
                "moe/expert_load_cv": (
                    load_fraction.std(unbiased=False) / uniform_fraction
                ).detach(),
                "moe/max_to_mean_load": (
                    load_fraction.max() / uniform_fraction
                ).detach(),
                "moe/dead_expert_count": (accepted_per_expert == 0).sum().detach(),
                # Vector metrics are stacked by transformer layer and emitted as
                # TensorBoard heatmaps/histograms by the runtime logger.
                "moe/expert_load_fraction": load_fraction.detach(),
                "moe/expert_probability": route.probability_per_expert.detach(),
            }

        return RouterOutput(
            expert_indices=expert_indices.to(torch.int32),
            expert_weights=expert_weights.to(hidden_states.dtype),
            auxiliary_loss=auxiliary_loss,
            z_loss=route.z_loss,
            metrics=metrics,
        )
