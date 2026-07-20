from __future__ import annotations

import os
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from minitrain.model.blocks import TransformerBlock


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


def build_auto_wrap_policy(name: str) -> ModuleWrapPolicy | None:
    """Return the explicit FSDP unit boundary used by this transformer."""

    if name == "none":
        return None
    if name == "transformer_block":
        return ModuleWrapPolicy({TransformerBlock})
    raise ValueError(f"Unsupported FSDP auto-wrap policy: {name!r}")


class FSDPStrategy:
    name = "fsdp"

    def __init__(
        self,
        *,
        process_group_backend: str = "auto",
        timeout_minutes: int = 30,
        sharding_strategy: str = "full_shard",
        auto_wrap_policy: str = "transformer_block",
        backward_prefetch: str = "backward_pre",
        forward_prefetch: bool = False,
        limit_all_gathers: bool = True,
        use_orig_params: bool = True,
        sync_module_states: bool = True,
        cpu_offload: bool = False,
        activation_checkpointing: bool = False,
        precision: str = "fp32",
    ) -> None:
        self.process_group_backend = process_group_backend
        self.timeout_minutes = timeout_minutes
        self.sharding_strategy = _SHARDING_STRATEGIES[sharding_strategy]
        self.auto_wrap_policy_name = auto_wrap_policy
        self.backward_prefetch = {
            "backward_pre": BackwardPrefetch.BACKWARD_PRE,
            "backward_post": BackwardPrefetch.BACKWARD_POST,
            "none": None,
        }[backward_prefetch]
        self.forward_prefetch = forward_prefetch
        self.limit_all_gathers = limit_all_gathers
        self.use_orig_params = use_orig_params
        self.sync_module_states = sync_module_states
        self.cpu_offload = cpu_offload
        self.activation_checkpointing = activation_checkpointing
        self.mixed_precision = self._build_mixed_precision(precision)
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    @property
    def rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))

    @property
    def world_size(self) -> int:
        if dist.is_initialized():
            return dist.get_world_size()
        return int(os.environ.get("WORLD_SIZE", "1"))

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
        if not torch.cuda.is_available():
            raise RuntimeError("FSDP training requires a CUDA accelerator")
        torch.cuda.set_device(self.local_rank)
        if not dist.is_initialized():
            backend = (
                "nccl"
                if self.process_group_backend == "auto"
                else self.process_group_backend
            )
            dist.init_process_group(
                backend=backend,
                timeout=timedelta(minutes=self.timeout_minutes),
            )

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        auto_wrap = build_auto_wrap_policy(self.auto_wrap_policy_name)
        wrapped = FullyShardedDataParallel(
            model,
            auto_wrap_policy=auto_wrap,
            sharding_strategy=self.sharding_strategy,
            mixed_precision=self.mixed_precision,
            backward_prefetch=self.backward_prefetch,
            forward_prefetch=self.forward_prefetch,
            limit_all_gathers=self.limit_all_gathers,
            device_id=torch.device("cuda", self.local_rank),
            use_orig_params=self.use_orig_params,
            sync_module_states=self.sync_module_states,
            cpu_offload=CPUOffload(offload_params=self.cpu_offload),
        )
        if self.activation_checkpointing:
            wrapper = partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            apply_activation_checkpointing(
                wrapped,
                checkpoint_wrapper_fn=wrapper,
                check_fn=lambda module: isinstance(module, TransformerBlock),
            )
        return wrapped

    def barrier(self) -> None:
        dist.barrier()

    def teardown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
