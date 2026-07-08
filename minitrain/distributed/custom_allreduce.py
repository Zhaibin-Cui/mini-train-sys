import torch
import torch.distributed as dist


def naive_allreduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Correctness-first allreduce wrapper.

    Replace this with ring reduce-scatter + allgather when implementing the
    teaching version. Keep this function's contract stable for benchmarks.
    """

    out = tensor.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out

