import torch

from minitrain.data.dataloader import build_training_dataloader
from minitrain.runtime.config import (
    DataConfig,
    LoggingConfig,
    TrainConfig,
    experiment_config_from_dict,
)
from minitrain.runtime.logger import (
    NullLogger,
    TensorBoardLogger,
    build_event_logger,
    get_tensorboard_log_dir,
)
from minitrain.train.trainer import Trainer


class TinyLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        targets: torch.Tensor,
        use_fused_loss: bool = False,
    ) -> tuple[torch.Tensor, None]:
        return self.weight * input_ids.float().mean(), None


def test_experiment_config_loads_data_section() -> None:
    cfg = experiment_config_from_dict(
        {
            "data": {
                "source": "random",
                "num_tokens": 32,
                "shuffle": False,
            },
            "logging": {
                "console": False,
                "tensorboard": False,
            },
        }
    )

    assert cfg.data == DataConfig(source="random", path=None, num_tokens=32, shuffle=False)
    assert cfg.logging == LoggingConfig(console=False, tensorboard=False)


def test_experiment_config_loads_precision_policy() -> None:
    cfg = experiment_config_from_dict(
        {
            "train": {
                "precision": "bf16",
                "grad_clip_norm": 0.5,
            }
        }
    )

    assert cfg.train == TrainConfig(precision="bf16", grad_clip_norm=0.5)


def test_random_training_dataloader_shapes() -> None:
    dataloader = build_training_dataloader(
        DataConfig(source="random", num_tokens=32, shuffle=False),
        seq_len=8,
        batch_size=2,
        vocab_size=128,
        seed=123,
    )

    batch = next(iter(dataloader))

    assert batch["input_ids"].shape == torch.Size([2, 8])
    assert batch["targets"].shape == torch.Size([2, 8])
    assert torch.equal(batch["input_ids"][:, 1:], batch["targets"][:, :-1])


def test_random_training_dataloader_shards_by_torchrun_rank(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    dataloader = build_training_dataloader(
        DataConfig(source="random", num_tokens=40, shuffle=False),
        seq_len=4,
        batch_size=2,
        vocab_size=128,
        seed=123,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    tokens = torch.randint(0, 128, (40,), dtype=torch.long, generator=generator)
    expected_shard = tokens[20:]
    batch = next(iter(dataloader))

    assert torch.equal(batch["input_ids"][0], expected_shard[:4])
    assert torch.equal(batch["targets"][0], expected_shard[1:5])


def test_disabled_logger_is_safe_to_call() -> None:
    logging_cfg = LoggingConfig(console=False, tensorboard=False)
    logger = build_event_logger(logging_cfg, run_name="unit_test")

    assert isinstance(logger, NullLogger)
    assert get_tensorboard_log_dir(logging_cfg, run_name="unit_test") is None
    logger.log_event({"event": "train", "step": 1, "loss": 1.0})
    logger.close()


def test_tensorboard_logger_writes_scalar_events(tmp_path) -> None:
    logger = TensorBoardLogger(log_dir=tmp_path)

    logger.log_event(
        {
            "event": "train",
            "step": 1,
            "loss": 1.0,
            "tokens_per_sec": 128.0,
            "gpu_memory_allocated_mb": 64.0,
        }
    )
    logger.close()

    assert any(path.name.startswith("events.out.tfevents") for path in tmp_path.iterdir())


def test_trainer_steps_optimizer_on_every_batch() -> None:
    model = TinyLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer, device=torch.device("cpu"))
    batch = {
        "input_ids": torch.ones(2, 3, dtype=torch.long),
        "targets": torch.ones(2, 3, dtype=torch.long),
    }

    trainer.train_step(batch)
    first_weight = float(model.weight.detach())
    trainer.train_step(batch)

    assert trainer.state.step == 2
    assert trainer.state.tokens_seen == 12
    assert first_weight < 1.0
    assert float(model.weight.detach()) < first_weight
