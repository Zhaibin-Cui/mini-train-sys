import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class DDPStrategy:
    name = "ddp"

    def __init__(
        self,
        *,
        process_group_backend: str = "auto",
        timeout_minutes: int = 30,
        broadcast_buffers: bool = False,
        bucket_cap_mb: int = 25,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
        static_graph: bool = False,
    ) -> None:
        self.process_group_backend = process_group_backend
        self.timeout_minutes = timeout_minutes
        self.broadcast_buffers = broadcast_buffers
        self.bucket_cap_mb = bucket_cap_mb
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.static_graph = static_graph
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    @property
    def rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))

    @property
    def world_size(self) -> int:
        if dist.is_initialized():
            return dist.get_world_size()
        return int(os.environ.get("WORLD_SIZE", "1"))

    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
        if not dist.is_initialized():
            backend = self.process_group_backend
            if backend == "auto":
                backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(
                backend=backend,
                timeout=timedelta(minutes=self.timeout_minutes),
            )

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        kwargs = {
            "broadcast_buffers": self.broadcast_buffers,
            "bucket_cap_mb": self.bucket_cap_mb,
            "find_unused_parameters": self.find_unused_parameters,
            "gradient_as_bucket_view": self.gradient_as_bucket_view,
            "static_graph": self.static_graph,
        }
        if torch.cuda.is_available():
            kwargs["device_ids"] = [self.local_rank]
            kwargs["output_device"] = self.local_rank
        return DistributedDataParallel(model, **kwargs)

    def barrier(self) -> None:
        dist.barrier()

    def teardown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
