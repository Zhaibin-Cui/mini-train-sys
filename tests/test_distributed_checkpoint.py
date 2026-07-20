from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from minitrain.train.checkpoint import (
    load_model_state_dict_from_checkpoint,
    restore_training_checkpoint,
    save_checkpoint,
)


def _save_ddp_checkpoint(rank: int, world_size: int, store_path: str, output: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=Path(store_path).as_uri(),
        rank=rank,
        world_size=world_size,
    )
    torch.manual_seed(7)
    model = DistributedDataParallel(torch.nn.Linear(3, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, 3)).sum()
    loss.backward()
    optimizer.step()
    save_checkpoint(output, model, optimizer, step=1, epoch=1, export_model=True)
    dist.destroy_process_group()


@pytest.mark.skipif(os.name == "nt", reason="Windows CI PyTorch lacks a usable Gloo store")
def test_ddp_checkpoint_reshards_into_single_process(tmp_path):
    """The same DCP payload restores without DDP or its original world size."""

    checkpoint = tmp_path / "epoch_000001_step_000000001"
    mp.spawn(
        _save_ddp_checkpoint,
        args=(2, str(tmp_path / "store"), str(checkpoint)),
        nprocs=2,
        join=True,
    )

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = restore_training_checkpoint(checkpoint, model, optimizer)

    assert state["step"] == 1
    assert len(optimizer.state) == 2

    # Inference and probe jobs deliberately ignore Adam/DCP and consume the
    # consolidated export on one process, even when training used many ranks.
    inference_model = torch.nn.Linear(3, 2)
    inference_model.load_state_dict(load_model_state_dict_from_checkpoint(checkpoint))
    for name, value in model.state_dict().items():
        assert torch.equal(inference_model.state_dict()[name], value)
