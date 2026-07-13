import pytest

from minitrain.kernels.cuda_ext import build
from minitrain.kernels.cuda_ext.generate_kernels import all_instantiations


@pytest.fixture(autouse=True)
def clear_build_config_cache():
    """Environment-driven build config must be recomputed in every test."""

    build.get_build_config.cache_clear()
    yield
    build.get_build_config.cache_clear()


@pytest.mark.parametrize(
    ("profile", "expected_sources"),
    [("minimal", 4), ("workstation", 24), ("full", 48)],
)
def test_build_profiles_select_complete_forward_backward_pairs(monkeypatch, profile, expected_sources):
    monkeypatch.setenv("MINITRAIN_CUDA_BUILD_PROFILE", profile)
    monkeypatch.delenv("MINITRAIN_CUDA_HEAD_DIMS", raising=False)
    monkeypatch.delenv("MINITRAIN_CUDA_DTYPES", raising=False)

    config = build.get_build_config()
    sources = build._instantiation_sources(config)

    assert len(sources) == expected_sources
    assert len(sources) == 2 * len(config.dtypes) * len(config.head_dims) * 2
    assert all(path.exists() for path in sources)


def test_generator_exposes_expected_reduced_matrix():
    """2 directions x 2 dtypes x 6 buckets x 2 masks equals 48 files."""

    kernels = all_instantiations()
    assert len(kernels) == 48
    assert len({kernel.filename for kernel in kernels}) == 48


def test_explicit_matrix_override_is_normalized(monkeypatch):
    monkeypatch.setenv("MINITRAIN_CUDA_BUILD_PROFILE", "minimal")
    monkeypatch.setenv("MINITRAIN_CUDA_ARCHS", "90;86;86")
    monkeypatch.setenv("MINITRAIN_CUDA_HEAD_DIMS", "128;64;128")
    monkeypatch.setenv("MINITRAIN_CUDA_DTYPES", "bf16;fp16")

    config = build.get_build_config()

    assert config.archs == ("86", "90")
    assert config.head_dims == (64, 128)
    assert config.dtypes == ("fp16", "bf16")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MINITRAIN_CUDA_BUILD_PROFILE", "unknown"),
        ("MINITRAIN_CUDA_ARCHS", "75"),
        ("MINITRAIN_CUDA_HEAD_DIMS", "80"),
        ("MINITRAIN_CUDA_DTYPES", "fp32"),
    ],
)
def test_invalid_build_matrix_is_rejected(monkeypatch, name, value):
    monkeypatch.setenv("MINITRAIN_CUDA_BUILD_PROFILE", "minimal")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        build.get_build_config()
