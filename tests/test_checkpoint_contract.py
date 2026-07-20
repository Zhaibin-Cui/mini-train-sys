import torch

from minitrain.runtime.config import CheckpointConfig
from minitrain.train.checkpoint import (
    _fill_missing_optimizer_group_options,
    load_model_state_dict_from_checkpoint,
    prune_checkpoints,
    resolve_resume_checkpoint,
    restore_training_checkpoint,
    save_checkpoint,
)


def test_missing_dcp_optimizer_group_options_are_filled_without_overwriting_loaded_values():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    original_groups = [dict(group) for group in optimizer.param_groups]

    optimizer.param_groups[0].pop("betas")
    optimizer.param_groups[0]["lr"] = 2e-3
    _fill_missing_optimizer_group_options(optimizer, original_groups)

    assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)
    assert optimizer.param_groups[0]["lr"] == 2e-3
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()


def test_default_epoch_checkpoint_policy():
    assert CheckpointConfig().every_epochs == 1
    assert CheckpointConfig().keep_last is None
    assert CheckpointConfig().keep_safety == 0


def test_model_export_interval_requires_enabled_export():
    try:
        CheckpointConfig(export_model_every_epochs=10)
    except ValueError as exc:
        assert "requires checkpoint.export_model=true" in str(exc)
    else:
        raise AssertionError("disabled model export accepted an export interval")


def test_full_checkpoint_resumes_adam_but_model_reader_ignores_it(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, 3)).sum()
    loss.backward()
    optimizer.step()

    path = tmp_path / "epoch_000001_step_000000001"
    save_checkpoint(path, model, optimizer, 1, epoch=1, tokens_seen=6, export_model=True)

    model_only = load_model_state_dict_from_checkpoint(path)
    assert set(model_only) == set(model.state_dict())

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    state = restore_training_checkpoint(path, restored_model, restored_optimizer)
    assert state == {
        "step": 1,
        "lr_step": 1,
        "epoch": 1,
        "tokens_seen": 6,
        "rng_restored": True,
        "saved_world_size": 1,
    }
    assert len(restored_optimizer.state) == len(optimizer.state)
    for name, value in model.state_dict().items():
        assert torch.equal(restored_model.state_dict()[name], value)


def test_checkpoint_retention_keeps_newest_files(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paths = [
        run_dir / f"epoch_{epoch:06d}_step_{epoch:09d}.pt" for epoch in range(1, 5)
    ]
    for path in paths:
        path.touch()

    removed = prune_checkpoints(tmp_path, "run", keep_last=2)

    assert removed == paths[:2]
    assert [path for path in paths if path.exists()] == paths[2:]


def test_retention_keeps_newest_committed_directories(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paths = [run_dir / f"epoch_{epoch:06d}_step_{epoch:09d}" for epoch in range(1, 5)]
    for path in paths:
        path.mkdir()
        (path / "COMMITTED").write_text("ok\n", encoding="utf-8")

    removed = prune_checkpoints(tmp_path, "run", keep_last=2)

    assert removed == paths[:2]
    assert [path for path in paths if path.exists()] == paths[2:]


def test_retention_keeps_recent_plus_old_safety_with_one_model_export(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paths = [run_dir / f"epoch_{epoch:06d}_step_{epoch:09d}" for epoch in range(1, 16)]
    for path in paths:
        path.mkdir()
        (path / "COMMITTED").write_text("ok\n", encoding="utf-8")
        (path / "model.pt").write_bytes(b"model")

    removed = prune_checkpoints(
        tmp_path,
        "run",
        keep_last=2,
        keep_safety=1,
        safety_every_epochs=10,
        keep_model_exports=1,
    )

    kept = [path for path in paths if path.exists()]
    assert kept == [paths[9], paths[13], paths[14]]
    assert removed == [path for path in paths if path not in kept]
    assert (paths[9] / "SAFETY").is_file()
    assert not (paths[9] / "model.pt").exists()
    assert not (paths[13] / "model.pt").exists()
    assert (paths[14] / "model.pt").is_file()
    assert resolve_resume_checkpoint(
        "latest", checkpoint_dir=tmp_path, run_name="run"
    ) == paths[14]
    assert resolve_resume_checkpoint(
        "safety", checkpoint_dir=tmp_path, run_name="run"
    ) == paths[9]


def test_safety_retention_has_fallback_before_first_milestone(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paths = [run_dir / f"epoch_{epoch:06d}_step_{epoch:09d}" for epoch in range(1, 5)]
    for path in paths:
        path.mkdir()
        (path / "COMMITTED").write_text("ok\n", encoding="utf-8")

    prune_checkpoints(
        tmp_path,
        "run",
        keep_last=2,
        keep_safety=1,
        safety_every_epochs=10,
    )

    assert [path for path in paths if path.exists()] == [paths[0], paths[2], paths[3]]
    assert (paths[0] / "SAFETY").is_file()


def test_safety_checkpoint_without_model_export_restores_training_state(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    run_dir = tmp_path / "run"
    for epoch in range(1, 5):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        save_checkpoint(
            run_dir / f"epoch_{epoch:06d}_step_{epoch:09d}",
            model,
            optimizer,
            step=epoch,
            epoch=epoch,
            export_model=True,
        )
    prune_checkpoints(
        tmp_path,
        "run",
        keep_last=2,
        keep_safety=1,
        safety_every_epochs=10,
        keep_model_exports=1,
    )
    safety = resolve_resume_checkpoint("safety", checkpoint_dir=tmp_path, run_name="run")
    assert not (safety / "model.pt").exists()

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    state = restore_training_checkpoint(safety, restored_model, restored_optimizer)
    assert state["epoch"] == 1
    assert len(restored_optimizer.state) == len(optimizer.state)
