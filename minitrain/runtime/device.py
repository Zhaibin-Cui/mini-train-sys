import os

import torch


def get_default_device() -> torch.device:
    """Return the device for the current process.

    DDP launches set `LOCAL_RANK`; single-process runs use `cuda:0` when
    available and CPU only as a smoke-test fallback.
    """

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")

