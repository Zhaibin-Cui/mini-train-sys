import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy


class FSDPStrategy:
    name = "fsdp"

    def __init__(self, sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD) -> None:
        self.sharding_strategy = sharding_strategy

    def setup(self) -> None:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return FullyShardedDataParallel(model, sharding_strategy=self.sharding_strategy)

    def barrier(self) -> None:
        dist.barrier()

    def teardown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

