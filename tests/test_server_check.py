from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import minitrain.runtime.server_check as server_check


def test_server_check_accepts_complete_linux_cuda_environment(monkeypatch, tmp_path):
    class VersionInfo(tuple):
        major = 3
        minor = 11

    properties = type(
        "Properties",
        (),
        {"name": "NVIDIA GeForce RTX 4090", "major": 8, "minor": 9, "total_memory": 24 * 1024**3},
    )()
    usage = namedtuple("usage", "total used free")(200 * 1024**3, 0, 200 * 1024**3)
    monkeypatch.setattr(server_check, "REQUIRED_MODULES", ())
    monkeypatch.setattr(server_check.sys, "version_info", VersionInfo((3, 11, 0)))
    monkeypatch.setattr(server_check.platform, "system", lambda: "Linux")
    monkeypatch.setattr(server_check.platform, "platform", lambda: "Linux-test")
    monkeypatch.setattr(server_check.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(server_check.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(server_check.torch.cuda, "get_device_properties", lambda _index: properties)
    monkeypatch.setattr(server_check.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(server_check.dist, "is_available", lambda: True)
    monkeypatch.setattr(server_check.dist, "is_nccl_available", lambda: True)
    monkeypatch.setattr(server_check, "_capture", lambda _command: (0, "ok"))
    monkeypatch.setattr(server_check.shutil, "which", lambda _name: "/usr/local/cuda/bin/nvcc")
    monkeypatch.setattr(server_check.shutil, "disk_usage", lambda _path: usage)
    monkeypatch.setattr(server_check, "_find_checkout", lambda: Path.cwd())
    monkeypatch.setattr(server_check.os, "access", lambda _path, _mode: True)
    monkeypatch.setenv("MINITRAIN_STORAGE_ROOT", str(tmp_path))

    status = server_check.collect_server_status(
        expected_gpus=1,
        min_free_disk_gb=100,
        require_nvcc=True,
    )

    assert status["ok"] is True
    assert status["errors"] == []
    assert status["visible_gpus"] == 1
    assert status["gpus"][0]["compute_capability"] == "8.9"
    assert status["storage_root"] == str(tmp_path.resolve())


def test_pyproject_exposes_reproducible_server_bundle_and_checker():
    source = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '"torch==2.5.1"' in source
    assert '"triton==3.1.0;' in source
    assert "server = [" in source
    assert 'include = ["minitrain*", "experiments*"]' in source
    assert 'minitrain-check-server = "minitrain.runtime.server_check:main"' in source


def test_server_setup_runbook_keeps_install_check_and_smoke_commands():
    setup = Path("scripts/bash/setup_server.sh").read_text(encoding="utf-8")
    storage = Path("scripts/bash/setup_storage.sh").read_text(encoding="utf-8")
    runbook = Path("docs/guides/server_setup.md").read_text(encoding="utf-8")
    assert "pip install -e \".[server]\"" in setup
    assert "minitrain-check-server" in setup
    assert "pip check" in setup
    assert ".minitrain-storage.env" in setup
    assert "ALLOW_SYSTEM_DISK_STORAGE" in setup
    assert "MINITRAIN_CUDA_BUILD_ROOT" in storage
    assert "TRITON_CACHE_DIR" in storage
    assert 'ln -s "$ARTIFACTS_ROOT" "$ARTIFACTS_LINK"' in storage
    assert "nvidia-smi topo -m" in runbook
    assert "torchrun --standalone" in runbook
    assert "synbios_moe.sh" in runbook
    assert "synbios_probes.sh" in runbook
