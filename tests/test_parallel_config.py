from pathlib import Path

from minitrain.distributed.fsdp import build_auto_wrap_policy
from minitrain.model.blocks import TransformerBlock
from minitrain.runtime.config import experiment_config_from_dict, load_yaml_dict
from minitrain.runtime.scaling import resolve_batch_scale


def test_fsdp_policy_targets_transformer_blocks():
    policy = build_auto_wrap_policy("transformer_block")
    assert policy is not None
    assert TransformerBlock in policy._module_classes


def test_synbios_layered_config_resolves_fsdp():
    payload = load_yaml_dict("configs/synbios_moe/runs/single_fsdp.yaml")
    cfg = experiment_config_from_dict(payload)

    assert cfg.parallel.strategy == "fsdp"
    assert cfg.parallel.fsdp.auto_wrap_policy == "transformer_block"
    assert cfg.train.epochs == 540
    assert cfg.train.grad_clip_norm == 5.0
    assert cfg.checkpoint.every_epochs == 1
    assert cfg.checkpoint.export_model


def test_linear_batch_scaling_preserves_epoch_budget():
    payload = load_yaml_dict("configs/synbios_moe/runs/single_ddp.yaml")
    cfg = experiment_config_from_dict(payload)
    resolved = resolve_batch_scale(
        cfg.train,
        cfg.optimizer,
        cfg.lr_scheduler,
        world_size=8,
    )

    assert cfg.train.epochs == 540
    assert resolved.global_batch_size == 64
    assert resolved.optimizer.lr == cfg.optimizer.lr * 2 / 3
    assert resolved.lr_scheduler.warmup_steps == 1500


def test_4090_server_matrix_has_explicit_topology_and_auto_workers():
    for strategy in ("ddp", "fsdp"):
        for world_size in (1, 4, 8):
            payload = load_yaml_dict(
                f"configs/server/rtx4090_24gb/runs/{strategy}_{world_size}gpu.yaml"
            )
            cfg = experiment_config_from_dict(payload)
            assert cfg.parallel.strategy == strategy
            assert cfg.parallel.expected_world_size == world_size
            assert cfg.train.batch_size == 4
            assert cfg.train.grad_clip_norm == 5.0
            assert cfg.data.num_workers is None
            assert cfg.data.worker_budget == 32


def test_synbios_distributed_run_names_match_server_launcher():
    for variant in ("single", "multi5_permute"):
        single = experiment_config_from_dict(
            load_yaml_dict(f"configs/synbios_moe/runs/{variant}_single.yaml")
        )
        assert single.run.name == f"synbios_moe_{variant}_single"
        for strategy in ("ddp", "fsdp"):
            for world_size in (4, 8):
                cfg = experiment_config_from_dict(
                    load_yaml_dict(
                        f"configs/synbios_moe/runs/"
                        f"{variant}_{strategy}_{world_size}gpu.yaml"
                    )
                )
                assert cfg.run.name == (
                    f"synbios_moe_{variant}_{strategy}_{world_size}gpu"
                )
                assert cfg.parallel.expected_world_size == world_size


def test_full_synbios_launcher_preserves_every_gated_stage():
    source = Path("scripts/bash/synbios_full_experiment.sh").read_text(encoding="utf-8")
    assert "CONFIRM_FULL_EXPERIMENT" in source
    assert "variants=(single multi5_permute)" in source
    assert "for stage in smoke pilot formal" in source
    assert "synbios_moe.sh" in source
    assert "synbios_probes.sh" in source
    assert "summarize-probes" in source


def test_synbios_launcher_prepares_every_required_data_artifact():
    source = Path("scripts/bash/synbios_moe.sh").read_text(encoding="utf-8")
    assert '"$DATA_ROOT/profiles.jsonl"' in source
    assert '"$DATA_ROOT/manifest.json"' in source
    assert '"$DATA_ROOT/token_shards/manifest.json"' in source
    assert "scripts/synbios_moe.py prepare" in source
    assert '--variant "$PREPARE_VARIANT"' in source
    assert "FORCE_PREPARE would invalidate existing checkpoints" in source
