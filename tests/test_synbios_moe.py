import json
from pathlib import Path

import pytest
import torch

import experiments.synbios_moe.probe_pipeline as probe_pipeline_module

from experiments.synbios_moe.data import (
    ATTRIBUTES,
    candidate_pools,
    generate_profiles,
    render_biography,
    write_dataset,
)
from experiments.synbios_moe.cloze_evaluation import (
    biography_cloze_fields,
    character_similarity,
)
from experiments.synbios_moe.evaluation import _attribute_target_positions
from experiments.synbios_moe.probe_data import (
    CachedProbeDataset,
    build_probe_cache,
    paper_probe_tasks,
    validate_probe_cache,
)
from experiments.synbios_moe.probe_pipeline import (
    JobCommand,
    ProbeJob,
    build_pipeline_identity,
    jobs_for_stage,
    load_pipeline_config,
    require_matching_identity,
    resolve_devices,
    schedule_jobs,
    summarize_probe_results,
)
from experiments.synbios_moe.probes import (
    AttributeProbe,
    GPT2Codec,
    PProbeDataset,
    ProbeBatchItem,
    active_parameter_estimate,
    collate_probe,
    train_probe,
)
from experiments.synbios_moe.router_analysis import analyze_batch
from minitrain.model.config import ModelConfig
from minitrain.model.transformer import MiniTransformer
from minitrain.model.ops import get_ops_backend


def tiny_model() -> MiniTransformer:
    return MiniTransformer(
        ModelConfig(
            vocab_size=50257,
            seq_len=256,
            n_layers=1,
            n_heads=4,
            hidden_size=32,
            intermediate_size=32,
            ffn_type="moe",
            num_experts=2,
            experts_per_token=1,
        ),
        get_ops_backend("torch"),
    )


def test_generation_is_deterministic_and_spans_are_exact():
    left = generate_profiles(3, 7)
    right = generate_profiles(3, 7)
    assert left == right
    biography = render_biography(left[0], variant="multi5+permute", sample=0, seed=9)
    for attribute in ATTRIBUTES:
        start, end = biography.attribute_spans[attribute]
        assert biography.text[start:end] == getattr(left[0], attribute)
    assert left[0].full_name in biography.text.split(". ", 1)[0]
    assert candidate_pools()["company_city"].count("New York, NY") == 36


def test_bios_uses_full_name_only_in_the_final_first_sentence():
    profile = generate_profiles(1, 7)[0]
    saw_permuted_birthday = False
    for seed in range(20):
        biography = render_biography(profile, variant="single+permute1", sample=0, seed=seed)
        first_sentence = biography.text.split(". ", 1)[0]
        assert profile.full_name in first_sentence
        assert biography.text.count(profile.full_name) == 1
        if biography.attribute_spans["birth_date"][0] > len(first_sentence):
            saw_permuted_birthday = True
    assert saw_permuted_birthday

    fullname = render_biography(profile, variant="single+fullname", sample=0, seed=7)
    assert fullname.text.count(profile.full_name) == len(ATTRIBUTES)


def test_progressive_cloze_uses_original_fact_order_and_spans():
    profile = generate_profiles(1, 19)[0]
    biography = render_biography(profile, variant="single+permute1", sample=0, seed=23)
    row = {
        "text": biography.text,
        "attribute_spans": biography.attribute_spans,
    }

    fields = biography_cloze_fields(row)

    assert len(fields) == len(ATTRIBUTES)
    assert [field.start for field in fields] == sorted(field.start for field in fields)
    assert {field.attribute for field in fields} == set(ATTRIBUTES)
    for field in fields:
        assert biography.text[field.start : field.end] == getattr(profile, field.attribute)

    calls_biography = next(
        render_biography(profile, variant="single", sample=0, seed=seed)
        for seed in range(100)
        if " a birthplace." in render_biography(
            profile, variant="single", sample=0, seed=seed
        ).text
    )
    city = next(
        field
        for field in biography_cloze_fields(
            {
                "text": calls_biography.text,
                "attribute_spans": calls_biography.attribute_spans,
            }
        )
        if field.attribute == "birth_city"
    )
    assert calls_biography.text[city.end :].startswith(" a birthplace.")


def test_cloze_character_similarity_is_normalized_and_partial():
    assert character_similarity("New York, NY", "new   york, ny") == 1.0
    assert character_similarity("New York", "New York, NY") == pytest.approx(8 / 12)
    assert character_similarity("", "Boston, MA") == 0.0


def test_p_probe_positions_and_frozen_backbone(tmp_path):
    pytest.importorskip("tiktoken")
    write_dataset(tmp_path, num_people=100, variant="single", seed=11)
    data = PProbeDataset(tmp_path, attribute="company", target="first", split="train")
    input_ids, positions, _ = collate_probe([data[0], data[1]])
    assert positions.shape == (2, 6)
    assert torch.all(positions >= 0)
    assert input_ids.max() < GPT2Codec().vocab_size

    codec = GPT2Codec()
    row = data.items[0]
    assert row.input_ids[0] == codec.eos
    raw = json.loads(
        (tmp_path / "biographies.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    ids, attribute_positions = _attribute_target_positions(
        codec, raw["text"], raw["attribute_spans"]
    )
    assert ids[0] == codec.eos
    assert all(attribute_positions[name] for name in ATTRIBUTES)

    model = tiny_model()
    probe = AttributeProbe(model, len(data.class_names), rank=2, kind="p")
    logits = probe(input_ids, positions)
    logits.sum().backward()
    assert logits.shape == (2, 6, len(data.class_names))
    assert not any(parameter.grad is not None for parameter in model.parameters())
    assert probe.delta.b.weight.grad is not None


def test_router_analysis_and_active_parameter_count():
    model = tiny_model()
    input_ids = torch.randint(0, model.cfg.vocab_size, (2, 8))
    positions = torch.tensor([[1, 3], [2, 4]])
    labels = torch.tensor([0, 1])
    result = analyze_batch(model, input_ids, positions, labels)
    assert len(result["layers"]) == 1
    assert len(result["layers"][0]["load"]) == 2
    counts = active_parameter_estimate(model)
    assert counts["active_estimate"] < counts["total"]


def test_probe_cache_matches_legacy_datasets(tmp_path):
    pytest.importorskip("tiktoken")
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    write_dataset(data_root, num_people=100, variant="multi2+permute", seed=17)
    build_probe_cache(data_root, cache_root)
    status = validate_probe_cache(cache_root)
    assert status["valid"]
    assert status["p_examples"] == 200
    assert len(paper_probe_tasks()) == 11
    compact_status = validate_probe_cache(cache_root, include_missing_classes=False)
    assert "missing_validation_classes" not in compact_status
    assert "missing_validation_class_counts" in compact_status

    legacy = PProbeDataset(data_root, attribute="company", target="whole", split="train")
    cached = CachedProbeDataset(
        cache_root, kind="p", attribute="company", target="whole", split="train"
    )
    assert cached.class_names == legacy.class_names
    assert len(cached) == len(legacy)
    assert cached[0] == legacy[0]

    cached_q = CachedProbeDataset(
        cache_root, kind="q", attribute="birth_city", target="first", split="validation"
    )
    assert len(cached_q) > 0
    assert len(cached_q[0].positions) == 1


def test_probe_stage_config_and_device_resolution():
    config = load_pipeline_config("configs/synbios_moe/probe_pipeline.yaml")
    smoke_steps, smoke_jobs, required = jobs_for_stage(config, "smoke")
    assert smoke_steps == 500
    assert len(smoke_jobs) == 2
    assert required is None
    formal_steps, formal_jobs, required = jobs_for_stage(config, "formal")
    assert formal_steps == 30_000
    assert len(formal_jobs) == 22
    assert required == "pilot"
    assert resolve_devices("0,3") == ("cuda:0", "cuda:3")
    assert resolve_devices("cpu") == ("cpu",)


def test_probe_stage_rejects_duplicate_jobs():
    config = {
        "stages": {
            "broken": {
                "steps": 1,
                "tasks": [
                    {"kind": "p", "attribute": "major", "target": "first"},
                    {"kind": "p", "attribute": "major", "target": "first"},
                ],
            }
        }
    }
    with pytest.raises(ValueError, match="duplicate probe jobs"):
        jobs_for_stage(config, "broken")


def test_pipeline_identity_binds_every_durable_input(tmp_path):
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    checkpoint = tmp_path / "checkpoint"
    data.mkdir()
    cache.mkdir()
    checkpoint.mkdir()
    (data / "manifest.json").write_text('{"dataset": 1}', encoding="utf-8")
    (cache / "manifest.json").write_text('{"cache": 1}', encoding="utf-8")
    model_config = tmp_path / "model.yaml"
    model_config.write_text("model: {}\n", encoding="utf-8")
    (checkpoint / "model.pt").write_bytes(b"model-v1")
    job = ProbeJob("p", "major", "first")

    identity = build_pipeline_identity(
        stage="smoke",
        steps=3,
        jobs=[job],
        seed=7,
        data=data,
        cache=cache,
        model_config=model_config,
        checkpoint=checkpoint,
    )
    require_matching_identity({"identity": identity}, identity, label="same")
    changed = {**identity, "seed": 8}
    with pytest.raises(ValueError, match="seed"):
        require_matching_identity({"identity": identity}, changed, label="changed")

    (checkpoint / "model.pt").write_bytes(b"model-v2")
    rebuilt = build_pipeline_identity(
        stage="smoke",
        steps=3,
        jobs=[job],
        seed=7,
        data=data,
        cache=cache,
        model_config=model_config,
        checkpoint=checkpoint,
    )
    assert rebuilt["checkpoint_model_sha256"] != identity["checkpoint_model_sha256"]


def test_probe_training_emits_health_metrics():
    pytest.importorskip("tiktoken")

    class CaptureLogger:
        def __init__(self):
            self.events = []

        def log_event(self, payload):
            self.events.append(payload)

        def close(self):
            pass

    items = [
        ProbeBatchItem([50256, 1, 2, 3], [1, 2], index % 2)
        for index in range(4)
    ]
    logger = CaptureLogger()
    result = train_probe(
        AttributeProbe(tiny_model(), 2, rank=2, kind="p"),
        items,
        items,
        device=torch.device("cpu"),
        batch_size=2,
        steps=2,
        logger=logger,
        log_interval=1,
    )
    train_events = [event for event in logger.events if event["event"] == "probe_train"]
    assert len(train_events) == 2
    assert {
        "accuracy",
        "accuracy_by_position",
        "grad_norm",
        "data_wait_ms",
        "data_wait_percent",
        "step_time_ms",
    } <= train_events[-1].keys()
    assert len(result["loss_curve"]) == 2


def test_probe_scheduler_reports_started_and_finished_for_cached_job(tmp_path):
    output = tmp_path / "done.pt"
    output.touch()
    events = []

    def builder(job, device):
        return JobCommand(["unused"], output, tmp_path / "unused.log")

    result = schedule_jobs(
        [ProbeJob("p", "major", "first")],
        ("cpu",),
        builder,
        on_event=events.append,
    )
    assert result[0]["status"] == "skipped_existing"
    assert [event["action"] for event in events] == ["started", "finished"]


def test_probe_scheduler_reports_heartbeats(tmp_path, monkeypatch):
    events = []
    output = tmp_path / "new.pt"

    def fake_run(command, log_path, *, heartbeat_seconds, on_heartbeat):
        assert heartbeat_seconds == 0.25
        on_heartbeat(1.5)
        output.touch()
        return 0, 2.0

    monkeypatch.setattr(probe_pipeline_module, "_run_one", fake_run)

    def builder(job, device):
        return JobCommand(["probe"], output, tmp_path / "probe.log")

    result = schedule_jobs(
        [ProbeJob("q", "birth_city", "first")],
        ("cpu",),
        builder,
        on_event=events.append,
        heartbeat_seconds=0.25,
    )
    assert result[0]["status"] == "completed"
    assert [event["action"] for event in events] == [
        "started",
        "heartbeat",
        "finished",
    ]
    assert events[1]["seconds"] == pytest.approx(1.5)


def test_probe_scheduler_fails_when_process_omits_output(tmp_path, monkeypatch):
    monkeypatch.setattr(
        probe_pipeline_module,
        "_run_one",
        lambda *args, **kwargs: (0, 1.0),
    )

    result = schedule_jobs(
        [ProbeJob("q", "major", "whole")],
        ("cpu",),
        lambda job, device: JobCommand(
            ["probe"], tmp_path / "missing.json", tmp_path / "probe.log"
        ),
    )
    assert result[0]["status"] == "failed"
    assert "did not create" in result[0]["error"]


def test_probe_scheduler_does_not_trust_orphaned_output(tmp_path, monkeypatch):
    output = tmp_path / "orphaned.pt"
    output.touch()
    monkeypatch.setattr(
        probe_pipeline_module,
        "_run_one",
        lambda *args, **kwargs: (0, 1.0),
    )

    result = schedule_jobs(
        [ProbeJob("p", "company", "first")],
        ("cpu",),
        lambda job, device: JobCommand(["probe"], output, tmp_path / "probe.log"),
        reuse_existing=False,
    )
    assert result[0]["status"] == "failed"
    assert "did not refresh" in result[0]["error"]


def test_synbios_notebook_covers_monitored_probe_pipeline():
    notebook = json.loads(
        Path("tests/synbios_moe_end_to_end.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required_calls = {
        "cache-probes",
        "validate-probe",
        "probe-pipeline --stage smoke",
        "probe-pipeline --stage pilot",
        "probe-pipeline --stage formal",
        "summarize-probes",
        "pipeline_events.jsonl",
        "heartbeat-seconds",
        "accuracy_by_position_running",
        "data_wait_percent",
        "protocol_version",
    }
    for required in required_calls:
        assert required in source
    assert all(not cell.get("outputs") for cell in notebook["cells"])

    smoke_steps, smoke_jobs, prerequisite = jobs_for_stage(
        load_pipeline_config("configs/synbios_moe/probe_pipeline_notebook_smoke.yaml"),
        "smoke",
    )
    assert smoke_steps == 3
    assert {job.kind for job in smoke_jobs} == {"p", "q"}
    assert prerequisite is None


def test_probe_result_postprocessing(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    base = {
        "kind": "q",
        "attribute": "major",
        "target": "whole",
        "classes": 100,
        "examples": 50,
        "dataset_manifest": {
            "files": {"profiles.jsonl": {"sha256": "same-profile-table"}}
        },
    }
    (left / "q_major_whole.json").write_text(
        json.dumps({**base, "validation_accuracy": [0.25]}), encoding="utf-8"
    )
    (right / "q_major_whole.json").write_text(
        json.dumps({**base, "validation_accuracy": [0.75]}), encoding="utf-8"
    )
    result = summarize_probe_results(
        {"single": left, "multi5_permute": right}, tmp_path / "summary"
    )
    assert len(result["rows"]) == 2
    assert result["comparisons"][0]["delta"] == pytest.approx(0.5)
    persisted = json.loads((tmp_path / "summary" / "summary.json").read_text())
    assert persisted["comparisons"] == result["comparisons"]
    assert (tmp_path / "summary" / "summary.csv").is_file()
    assert (tmp_path / "summary" / "comparison.csv").is_file()


def test_probe_result_postprocessing_rejects_incomplete_stage(tmp_path):
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "p_major_first.json").write_text(
        json.dumps(
            {
                "kind": "p",
                "attribute": "major",
                "target": "first",
                "validation_accuracy": [0.5] * 6,
                "classes": 2,
                "examples": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete probe results"):
        summarize_probe_results(
            {"smoke": validation},
            tmp_path / "summary",
            expected_jobs=[
                ProbeJob("p", "major", "first"),
                ProbeJob("q", "major", "first"),
            ],
        )
