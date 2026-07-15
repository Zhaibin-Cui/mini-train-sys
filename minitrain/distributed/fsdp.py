from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, ShardingStrategy


_SHARDING_STRATEGIES = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
    "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
}

_PRECISION_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


class FSDPStrategy:
    name = "fsdp"

    def __init__(
        self,
        sharding_strategy: str | ShardingStrategy = ShardingStrategy.FULL_SHARD,
        precision: str = "fp32",
    ) -> None:
        if isinstance(sharding_strategy, str):
            try:
                sharding_strategy = _SHARDING_STRATEGIES[sharding_strategy]
            except KeyError as exc:
                choices = ", ".join(_SHARDING_STRATEGIES)
                raise ValueError(
                    f"Unknown FSDP sharding strategy {sharding_strategy!r}; expected: {choices}"
                ) from exc
        self.sharding_strategy = sharding_strategy
        self.mixed_precision = self._build_mixed_precision(precision)
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    @staticmethod
    def _build_mixed_precision(precision: str) -> MixedPrecision | None:
        if precision == "fp32":
            return None
        try:
            activation_dtype = _PRECISION_DTYPES[precision]
        except KeyError as exc:
            raise ValueError(f"Unsupported FSDP precision: {precision!r}") from exc
        return MixedPrecision(
            param_dtype=activation_dtype,
            reduce_dtype=torch.float32,
            # RoPE is constructed in fp32 and stored once in activation dtype.
            # Keep FSDP buffers aligned so forward can return slice views only.
            buffer_dtype=activation_dtype,
            keep_low_precision_grads=False,
            cast_forward_inputs=True,
        )

    def setup(self) -> None:
        torch.cuda.set_device(self.local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return FullyShardedDataParallel(
            model,
            sharding_strategy=self.sharding_strategy,
            mixed_precision=self.mixed_precision,
            device_id=self.local_rank,
            use_orig_params=True,
        )

    def barrier(self) -> None:
        dist.barrier()

    def teardown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
