import torch


class SingleDeviceStrategy:
    name = "single"

    rank = 0
    world_size = 1

    def setup(self) -> None:
        return None

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def barrier(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def teardown(self) -> None:
        return None
