import torch


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    has_cuda_params = any(p.is_cuda for p in model.parameters())
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        fused=has_cuda_params,
    )
