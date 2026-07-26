import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DOCS = (
    ROOT / "README.md",
    ROOT / "reports/README.md",
    ROOT / "reports/engineering/kernels.md",
    ROOT / "reports/engineering/distributed_training.md",
    ROOT / "reports/synbios_moe/README.md",
    ROOT / "reports/synbios_moe/storage_story.md",
    ROOT / "results/README.md",
    ROOT / "results/BENCHMARK_SUMMARY.md",
    ROOT / "docs/guides/artifact_layout.md",
)
LINK_PATTERN = re.compile(r"!?\[[^]]*]\(([^)]+)\)")


def test_canonical_document_links_resolve():
    missing = []
    for document in CANONICAL_DOCS:
        assert document.is_file()
        for raw_target in LINK_PATTERN.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "/")):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, "\\n".join(missing)


def test_superseded_reports_are_short_compatibility_pages():
    compatibility = {
        "reports/operator_bench.md": "engineering/kernels.md",
        "reports/distributed_bench.md": "engineering/distributed_training.md",
        "reports/server_benchmark_resume.md": "engineering/kernels.md",
        "reports/synbios_moe/probes/q_whole_moe_diagnostics.md": "storage_story.md",
    }
    for relative, canonical_target in compatibility.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert len(text) < 1_000
        assert canonical_target in text


def test_result_layout_has_no_new_flat_log_dump():
    root_files = sorted(
        path.name for path in (ROOT / "results/logs").iterdir() if path.is_file()
    )
    assert root_files == ["README.md"]
    for category in ("benchmarks", "experiments", "maintenance", "validation"):
        assert (ROOT / "results/logs" / category).is_dir()


def test_result_catalog_contract_is_present():
    required = (
        "results/catalog/artifacts.json",
        "results/catalog/export_audit.json",
        "results/catalog/retention.json",
        "results/catalog/summary.md",
        "results/tensorboard/index.csv",
        "results/MANIFEST.sha256",
    )
    for relative in required:
        assert (ROOT / relative).is_file(), relative
