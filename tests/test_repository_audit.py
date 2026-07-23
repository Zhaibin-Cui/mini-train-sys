import json
from pathlib import Path

import yaml

from experiments.synbios_moe.repository_audit import EXPECTED_PROBE_RUNTIME, classify_log


ROOT = Path(__file__).resolve().parents[1]


def test_log_catalog_categories_are_stable():
    assert classify_log("probe_formal_single_final_20260723.log") == "probe"
    assert classify_log("multi5_cloze_full_gpu0_20260721.log") == "pretraining_validation"
    assert classify_log("synbios_moe_fsdp4_capacity_refine.log") == "benchmark"
    assert classify_log("synbios_moe_single_fsdp4_formal.log") == "pretraining"
    assert classify_log("synbios_multi5_permute_prepare.log") == "dataset"
    assert classify_log("tensorboard.log") == "infrastructure"


def test_probe_defaults_match_both_completed_formal_identities():
    config = yaml.safe_load(
        (ROOT / "configs/synbios_moe/probe_pipeline.yaml").read_text(encoding="utf-8")
    )
    assert config["runtime"] == EXPECTED_PROBE_RUNTIME
    for condition in ("single", "multi5_permute"):
        pipeline = json.loads(
            (
                ROOT
                / "artifacts/synbios_moe/results"
                / f"{condition}_fsdp_4gpu/probe_pipeline/formal/pipeline.json"
            ).read_text(encoding="utf-8")
        )
        assert pipeline["identity"]["runtime"] == EXPECTED_PROBE_RUNTIME


def test_exporter_persists_dataset_lineage_sidecars():
    exporter = (ROOT / "scripts/bash/export_test_results.sh").read_text(encoding="utf-8")
    assert 'source_root/lineage.json"' in exporter
    assert 'source_root/token_shards/lineage.json"' in exporter
    assert 'source_root/probe_cache/lineage.json"' in exporter
