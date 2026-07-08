import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class DDPStrategy:
    name = "ddp"

    def __init__(self, bucket_cap_mb: int = 25, gradient_as_bucket_view: bool = True) -> None:
        self.bucket_cap_mb = bucket_cap_mb
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    def setup(self) -> None:
        torch.cuda.set_device(self.local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return DistributedDataParallel(
            model,
            device_ids=[self.local_rank],
            bucket_cap_mb=self.bucket_cap_mb,
            gradient_as_bucket_view=self.gradient_as_bucket_view,
        )

    def barrier(self) -> None:
        dist.barrier()

    def teardown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

