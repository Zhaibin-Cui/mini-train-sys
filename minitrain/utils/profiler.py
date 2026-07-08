import torch


def reset_cuda_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2

