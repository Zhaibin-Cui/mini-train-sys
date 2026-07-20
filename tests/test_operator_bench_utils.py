from __future__ import annotations

import gc
import weakref

import pytest
import torch

import operator_bench_utils as bench_utils
import operator_nsight


class _Marker:
    pass


def test_close_stats_streams_across_chunk_boundaries():
    chunk_elements = 4 * 1024 * 1024
    expected = torch.zeros(chunk_elements + 3)
    actual = expected.clone()
    actual[-1] = 0.25

    stats = bench_utils.close_stats(actual, expected, atol=0.1, rtol=0.0)

    assert stats["correct"] is False
    assert stats["max_abs"] == pytest.approx(0.25)


def test_close_stats_rejects_tensor_metadata_mismatch():
    stats = bench_utils.close_stats(
        torch.zeros(2, dtype=torch.float32),
        torch.zeros(2, dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )

    assert stats == {"correct": False, "max_abs": float("inf"), "max_rel": float("inf")}


def test_release_cache_attempts_empty_cache_after_synchronize_failure(monkeypatch):
    emptied = []
    ipc_collected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda: (_ for _ in ()).throw(RuntimeError("failed CUDA context")),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: ipc_collected.append(True))

    bench_utils.release_cache()

    assert emptied == [True]
    assert ipc_collected == [True]


def test_run_with_cuda_cleanup_drops_failed_frame_locals():
    references = []

    def fail_with_local_object():
        marker = _Marker()
        references.append(weakref.ref(marker))
        raise RuntimeError("expected benchmark failure")

    with pytest.raises(RuntimeError, match="expected benchmark failure"):
        bench_utils.run_with_cuda_cleanup(fail_with_local_object)

    gc.collect()
    assert references[0]() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_run_with_cuda_cleanup_releases_memory_after_failure():
    bench_utils.release_cache()
    baseline = torch.cuda.memory_allocated()

    def fail_after_cuda_allocation():
        tensor = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        assert tensor.numel() > 0
        raise torch.cuda.OutOfMemoryError("synthetic CUDA OOM")

    with pytest.raises(torch.cuda.OutOfMemoryError, match="synthetic CUDA OOM"):
        bench_utils.run_with_cuda_cleanup(fail_after_cuda_allocation)

    assert torch.cuda.memory_allocated() == baseline
    assert torch.cuda.memory_reserved() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bench_sweep_releases_memory_after_provider_failure():
    bench_utils.release_cache()
    baseline = torch.cuda.memory_allocated()

    def make_case(size):
        return bench_utils.BenchCase(
            tensors={"x": torch.empty(size, device="cuda")},
            grad_names=("x",),
        )

    def failing_forward(provider, tensors):
        assert tensors["x"].is_cuda
        raise RuntimeError(f"expected {provider} failure")

    rows = bench_utils.bench_sweep(
        kernel="failure_probe",
        providers=("broken",),
        sizes=(4 * 1024 * 1024,),
        size_label="elements",
        make_case=make_case,
        forward=failing_forward,
        warmup_ms=1,
        rep_ms=1,
    )

    assert rows[0]["status"] == "unavailable"
    assert torch.cuda.memory_allocated() == baseline
    assert torch.cuda.memory_reserved() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nsight_case_releases_memory_after_failure():
    bench_utils.release_cache()
    baseline = torch.cuda.memory_allocated()

    def make_case(size):
        return bench_utils.BenchCase(
            tensors={"x": torch.empty(size, device="cuda")},
            grad_names=("x",),
        )

    def failing_forward(provider, tensors):
        assert tensors["x"].is_cuda
        raise RuntimeError(f"expected {provider} failure")

    spec = operator_nsight.ProfileSpec(
        make_case=make_case,
        forward=failing_forward,
        default_size=4 * 1024 * 1024,
        provider="broken",
    )
    with pytest.raises(RuntimeError, match="expected broken failure"):
        operator_nsight._run_case(spec, size=spec.default_size, mode="fwd")

    assert torch.cuda.memory_allocated() == baseline
    assert torch.cuda.memory_reserved() == 0
