from typing import Protocol

import torch


class ParallelStrategy(Protocol):
    """Contract for distributed training choices.

    Training code should ask the strategy to set up process groups and wrap the
    model. This keeps DDP/FSDP/custom allreduce experiments outside the model and
    kernel layers.
    """

    name: str

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def setup(self) -> None:
        """Initialize device and process-group state needed by the strategy."""
        ...

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Return the model as the strategy wants the trainer to see it."""
        ...

    def barrier(self) -> None:
        """Synchronize ranks or the local CUDA stream for timing boundaries."""
        ...

    def teardown(self) -> None:
        """Release process groups and strategy-owned resources."""
        ...
