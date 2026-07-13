# CUDA FlashAttention Extension

This package provides MiniTrain's CUDA-only FlashAttention backend. Dense
forward/backward kernels come from FlashAttention 2.8.4's Ampere implementation;
MiniTrain owns the narrow Python/C++ integration and generated source matrix.

## Supported Contract

- CUDA sm80 or newer; locally verified on sm86.
- `(batch, heads, sequence, head_dim)` Q/K/V with equal shapes and heads.
- fp16 and bf16 selected by the active build profile.
- head dimensions divisible by 8 and at most 256.
- causal and non-causal dense attention.
- dropout and no-dropout forward and backward.
- no GQA/MQA, varlen, KV cache, local window, ALiBi, or softcap.

Unsupported calls follow the backend inheritance chain CUDA -> Triton ->
PyTorch. The no-dropout CUDA specialization is a compile-time branch: it does
not read Philox state, generate random values, or retain a dropout mask.

## Source Map

- `__init__.py`: `CudaOpsBackend` and fallback inheritance.
- `flash_attention.py`: support predicate and PyTorch autograd bridge.
- `build.py`: build profiles, architecture flags, source selection, and JIT cache.
- `generate_kernels.py`: deterministic 48-file instantiation generator.
- `csrc/flash_api_upstream.cpp`: MiniTrain tensor/stride and pybind adapter.
- `csrc/instantiations/*.cu`: thin dtype/head-dim/causal forward/backward files.
- `csrc/third_party/flash_attn/src`: unmodified FlashAttention 2.8.4 kernel headers.
- `csrc/third_party/cutlass/include`: vendored CUTLASS/CUTE headers.
- `csrc/third_party/*LICENSE`: required BSD-3-Clause license texts.

## Build Profiles

| Profile | Dtypes | Head-dim buckets | `.cu` files |
| --- | --- | --- | ---: |
| `minimal` | fp16 | 32 | 4 |
| `workstation` | fp16, bf16 | 32, 64, 128 | 24 |
| `full` | fp16, bf16 | 32, 64, 96, 128, 192, 256 | 48 |

Each bucket has forward/backward and causal/non-causal files. Dropout is
specialized inside every file using upstream's `DROPOUT_SWITCH`, so each object
contains independent dropout and no-dropout kernel trees.

## Local sm86 Compile And Test

Run the small profile first. A four-file shard on the RTX 3050 Laptop GPU took
roughly five to eight minutes with one nvcc worker; subsequent loads of the same
matrix reuse the `.pyd`.

```powershell
$env:MINITRAIN_CUDA_BUILD_PROFILE="minimal"
$env:MINITRAIN_CUDA_ARCHS="86"
$env:MINITRAIN_CUDA_MAX_JOBS="1"
$env:MINITRAIN_CUDA_VERBOSE="1"

python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"

$env:MINITRAIN_RUN_CUDA_EXT_TESTS="1"
python -m pytest tests/test_cuda_flash_attention.py -q
```

Compile the practical local training matrix after the minimal test passes:

```powershell
$env:MINITRAIN_CUDA_BUILD_PROFILE="workstation"
$env:MINITRAIN_CUDA_ARCHS="86"
$env:MINITRAIN_CUDA_MAX_JOBS="1"
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

On a small machine, validate one bucket/dtype before committing to the 24-file
workstation build:

```powershell
$env:MINITRAIN_CUDA_BUILD_PROFILE="minimal"
$env:MINITRAIN_CUDA_HEAD_DIMS="128"
$env:MINITRAIN_CUDA_DTYPES="bf16"
$env:MINITRAIN_CUDA_ARCHS="86"
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

## Server Build

Build every tensor specialization and selected CUDA architectures on a machine
with more CPU RAM and parallel compilation capacity:

```bash
export MINITRAIN_CUDA_BUILD_PROFILE=full
export MINITRAIN_CUDA_ARCHS="80;86;89;90"
export MINITRAIN_CUDA_MAX_JOBS=8
export MINITRAIN_CUDA_VERBOSE=1
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

The newest selected architecture also retains PTX for forward compatibility.
Override profile dimensions directly when a model needs a smaller wheel/cache:

```powershell
$env:MINITRAIN_CUDA_HEAD_DIMS="64;128"
$env:MINITRAIN_CUDA_DTYPES="fp16;bf16"
```

The configuration is hashed into the extension module and build-directory
name. Different matrices cannot accidentally load an incompatible old DLL.

## Generate And Verify Sources

Generated `.cu` files are checked in so the dispatch matrix is visible and
reviewable. Regenerate after changing the matrix, then use `--check` in CI:

```powershell
python minitrain/kernels/cuda_ext/generate_kernels.py
python minitrain/kernels/cuda_ext/generate_kernels.py --check
```

## Training And Benchmark

Select the backend in model configuration:

```yaml
backend:
  ops: cuda
```

The first supported attention call loads the matching extension. Open
`tests/operator_bench.ipynb` after compiling the profile used by its head dim;
the attention sections compare `torch`, `triton`, and `cuda` forward/backward
latency, peak memory, correctness, and speedup.

`--ptxas-options=-v` is always enabled. Treat register and spill output as part
of kernel review, especially before adopting a new head-dim/architecture pair.
The local sm86 hdim32 backward tree reached 253-255 registers and some branches
spilled 24-72 bytes, so it is functionally valid but still requires targeted
sm86 tuning before being called fully optimized for this laptop GPU.
