import torch

from minitrain.runtime.config import OptimizerConfig


def build_optimizer(
    model: torch.nn.Module,
    *,
    cfg: OptimizerConfig | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
) -> torch.optim.Optimizer:
    if cfg is None:
        cfg = OptimizerConfig(
            lr=lr if lr is not None else OptimizerConfig.lr,
            weight_decay=(
                weight_decay if weight_decay is not None else OptimizerConfig.weight_decay
            ),
        )

    decay, no_decay = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)

    parameter_groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    has_cuda_params = any(p.is_cuda for p in model.parameters())
    return torch.optim.AdamW(
        parameter_groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        fused=has_cuda_params if cfg.fused is None else cfg.fused,
    )
