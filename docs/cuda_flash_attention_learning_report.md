# MiniTrain CUDA FlashAttention Learning Report

## 1. Goal And Scope

MiniTrain needs one production-oriented dense attention operator, not the full
feature surface of upstream FlashAttention. The retained contract is:

- CUDA only;
- fixed-length `(B, H, S, D)` Q/K/V with equal head counts;
- fp16 and bf16;
- causal and non-causal attention;
- dropout and no-dropout;
- native CUDA forward and backward;
- architecture and tensor specialization without branches in model code.

GQA/MQA, variable length, paged KV cache, local attention, ALiBi, softcap, fp8,
and split-KV are deliberately outside this implementation. This removes API
complexity while leaving upstream's proven tiled kernel mathematics intact.

## 2. Why The First Correctness Kernel Was Replaced

The initial local kernel assigned one CTA to one query row and made three passes:
row maximum, softmax denominator, then output. It avoided an `S x S` allocation,
but recomputed every QK dot product in multiple passes and had no CUDA backward
or dropout. It was useful to validate pybind and Windows compilation, but it was
not an industrial FlashAttention kernel.

The current implementation vendors the FlashAttention 2.8.4 Ampere kernel
headers and CUTLASS/CUTE headers. MiniTrain changes the API and build matrix,
not the core tile implementation. This is both faster and lower risk than
reimplementing tensor-core MMA, shared-memory swizzles, online softmax, masking,
Philox dropout, and gradient accumulation from scratch.

## 3. Layered Architecture

The runtime call path is:

```text
Transformer
  -> OpsBackend.attention
  -> CudaOpsBackend
       supported -> MiniTrainCudaFlashAttentionFunction
                    -> pybind C++ adapter
                    -> dtype/head-dim/causal dispatch
                    -> generated CUDA specialization
                    -> upstream launch template and tiled kernel
       unsupported -> TritonOpsBackend
                       unsupported -> TorchOpsBackend / PyTorch SDPA
```

This hierarchy is intentional. Model code has no CUDA shape switches, and each
lower backend remains a valid correctness fallback.

The source ownership boundary is:

- MiniTrain owns `__init__.py`, `flash_attention.py`, `build.py`,
  `generate_kernels.py`, and `flash_api_upstream.cpp`.
- FlashAttention owns the common kernel headers under
  `csrc/third_party/flash_attn/src`.
- NVIDIA CUTLASS owns headers under `csrc/third_party/cutlass/include`.
- Generated `.cu` files are mechanically derived from MiniTrain's matrix and
  preserve upstream's explicit-specialization structure.

## 4. Tensor Layout Mapping

MiniTrain tensors use `(batch, heads, sequence, head_dim)`. Upstream's public API
usually presents `(batch, sequence, heads, head_dim)`, but the CUDA parameter
structure does not require that physical order. It receives element strides:

| Upstream parameter | MiniTrain tensor stride |
| --- | --- |
| `*_batch_stride` | `stride(0)` |
| `*_head_stride` | `stride(1)` |
| `*_row_stride` | `stride(2)` |
| contiguous D | `stride(3) == 1` |

Therefore the C++ adapter maps strides directly. There is no Q/K/V transpose,
contiguous copy, or layout conversion on the normal model path.

## 5. Forward Kernel Logic

Upstream processes query and key/value tiles rather than individual rows. The
important algorithmic ideas are:

1. Load a Q tile and a K tile through coalesced global-memory transactions.
2. Use tensor-core MMA to compute a score tile in fp32 accumulators.
3. Apply causal/non-causal masks at compile time.
4. Update running row maxima and sums using online softmax.
5. Rescale the previous output accumulator when the running maximum changes.
6. Multiply probabilities by a V tile and accumulate output without writing the
   score matrix to global memory.
7. Store output and one fp32 log-sum-exp value per query row.

Memory is linear in Q/K/V/output plus LSE instead of quadratic in sequence
length. The tiled implementation also reuses Q/K/V data in shared memory and
register fragments, unlike the removed correctness kernel's repeated scalar
dot products.

## 6. Causal Specialization

Causal mode is represented by the compile-time `Is_causal` template argument.
Generated files bind both public symbols separately:

```cpp
run_mha_fwd_<cutlass::half_t, 64, false>(...)
run_mha_fwd_<cutlass::half_t, 64, true>(...)
```

The launcher chooses one symbol once. The hot kernel does not branch between
causal and non-causal behavior at runtime. Causal tiles beyond the diagonal are
skipped or masked according to the upstream block rules.

## 7. Dropout And The No-Dropout Fast Path

Dropout is intentionally not encoded in filenames. Upstream's launch template
uses `DROPOUT_SWITCH` to instantiate `Is_dropout=true` and `false` kernel trees
inside each dtype/head/causal translation unit.

For dropout:

- C++ obtains a Philox seed and counter offset from PyTorch's CUDA generator.
- Forward writes two int64 values `(seed, offset)` to a tiny CUDA tensor.
- A tile's logical coordinates deterministically select Philox counters.
- The mask is applied while probabilities are consumed; no full mask is stored.
- Backward receives the saved seed/offset and regenerates the same mask online.

For `dropout_p == 0`:

- Python saves an empty RNG tensor.
- C++ does not reserve generator state.
- `if constexpr (Is_dropout)` removes RNG unpacking, random generation, mask
  predicates, dropout scaling, and return-softmax state from the kernel.

This is stronger than a runtime `if (dropout_p != 0)`: the no-dropout binary
does not carry the random-number work or its live registers.

## 8. Backward Logic And Saved State

Autograd saves Q, K, V, output, LSE, and optional RNG state. It does not save an
attention matrix. Backward recomputes probability tiles from Q/K and LSE, then
forms dV, dP, dS, dQ, and dK using tiled MMA operations.

The adapter allocates:

- final `dq`, `dk`, and `dv` in the input dtype;
- a rounded fp32 `dq_accum` workspace;
- a rounded fp32 softmax-delta workspace.

The current adapter selects upstream's non-deterministic accumulation path.
This is the normal high-throughput mode; a deterministic multi-split dQ mode can
be exposed later if MiniTrain adds that API requirement.

## 9. Why There Are 48 `.cu` Files

The generated matrix is:

```text
2 directions
x 2 dtypes
x 6 head-dimension buckets
x 2 causal modes
= 48 translation units
```

The head buckets are 32, 64, 96, 128, 192, and 256. A head dimension below a
compiled bucket uses masked tail loads/stores. For example, D=80 dispatches to
the 96 bucket if that bucket is linked.

Splitting files has two benefits:

- ninja can compile independent configurations in parallel on a build server;
- one nvcc process does not retain all template trees in RAM simultaneously.

`generate_kernels.py --check` proves checked-in files match the matrix and
detects both missing and orphaned `.cu` files.

## 10. Build Profiles And Architectures

Profiles select linked sources, not runtime behavior:

- `minimal`: 4 files, fp16 + hdim32, compiler and CI smoke tests.
- `workstation`: 24 files, fp16/bf16 + hdim32/64/128.
- `full`: 48 files, all dtypes and buckets.

`MINITRAIN_CUDA_ARCHS` selects cubins. The local default is `86`; a server build
can select `80;86;89;90`. The newest target retains PTX. The current kernel
family is the upstream sm80/Ampere implementation and runs on sm80+; Hopper has
separate upstream kernels and should eventually receive a dedicated sm90 path
for peak H100 performance rather than relying on the Ampere-family kernel.

The config is hashed into the Python module and build directory. Changing a
profile, dtype matrix, head matrix, or architecture list cannot load a stale
binary with missing symbols.

## 11. Compiler Flags

Important nvcc flags are centralized in `build.py`:

- `-O3`, C++17, relaxed constexpr, extended lambda;
- fast math, matching upstream's approximate exponential path;
- CUDA half/bfloat macro undefinitions required by CUTLASS;
- `--ptxas-options=-v` for registers, stack, and spill diagnostics;
- `-allow-unsupported-compiler` and
  `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` on Windows.

Host flags use `/O2 /std:c++17` for MSVC and `-O3 -std=c++17` for GCC/Clang.
MSVC definitions use `/D`; nvcc definitions use `-D` even on Windows.

## 12. Verified Local Evidence

Environment:

- PyTorch 2.5.1+cu121;
- CUDA toolkit 12.1;
- RTX 3050 Laptop GPU, compute capability 8.6;
- Visual C++ 19.44;
- minimal profile, one nvcc worker.

Observed results:

- fp16 buckets D=32, D=64, and D=128 each compiled as an independent four-file
  forward/backward + causal/non-causal shard on sm86.
- bf16 D=128 compiled and linked successfully on the same sm86 toolchain.
- A four-file shard took roughly five to eight minutes with one nvcc worker,
  depending on head dimension.
- causal and non-causal fp16 forward matched PyTorch SDPA exactly for the tested
  `(2, 3, 32, 32)` shape.
- maximum backward absolute error was approximately `1.95e-3`.
- dropout=0.25 reproduced output and Q/K/V gradients exactly when the CUDA seed
  was reset, changed output for a different seed, and remained finite.
- Every verified shard passed the five profile-aware CUDA tests, including an
  explicit fp32 dropout reference built from the exact CUDA keep mask.
- backend fallback tests: 2 passed.

One D=128 fp16 benchmark at `(B=1, H=4, S=1024, D=128)`, causal, no dropout,
reported:

| Provider | Forward p50 | Backward p50 | Backward peak memory |
| --- | ---: | ---: | ---: |
| PyTorch | 0.282 ms | 1.277 ms | 11.02 MB |
| Triton | 0.246 ms | 1.276 ms | 11.02 MB |
| CUDA | 0.244 ms | 0.458 ms | 6.02 MB |

At this one shape CUDA was about 1.16x faster than PyTorch forward and 2.79x
faster backward. This is integration evidence, not a general performance claim;
the notebook sequence sweep is still required.

ptxas showed several hdim32 backward variants at 253-255 registers. Some
causal/dropout combinations spilled roughly 24-72 bytes. This is valuable
evidence, not a harmless warning: the upstream generic Ampere tuning is not
automatically optimal for a small sm86 laptop GPU. Benchmark and Nsight data
must guide any trait changes; core headers should not be edited speculatively.

## 13. Benchmark Workflow

`tests/operator_bench.ipynb` now includes `torch`, `triton`, and `cuda` in both
attention sections. It reports the active CUDA build config before running.

No-dropout benchmark correctness compares forward and backward with PyTorch.
Dropout has two validation paths:

- Triton materializes its debug mask and compares with an explicit reference.
- CUDA uses upstream's sign-bit debug output to recover the exact keep mask,
  then compares output and Q/K/V gradients with an explicit fp32 reference. It
  also checks Philox replay and changed-seed behavior before timing.

Compile a profile containing the notebook's D=128 bucket before execution. If
the bucket is absent, `CudaOpsBackend` intentionally measures the Triton
fallback rather than the CUDA kernel, so always inspect the printed build config.

## 14. Recommended Reading Order

1. `minitrain/model/ops.py`: backend factory.
2. `minitrain/kernels/cuda_ext/__init__.py`: fallback inheritance.
3. `flash_attention.py`: support and autograd contract.
4. `build.py`: source/architecture matrix and compilation.
5. `generate_kernels.py`: why each `.cu` exists.
6. one file under `csrc/instantiations`: thin explicit specialization.
7. `flash_api_upstream.cpp`: tensor strides, workspaces, and RNG state.
8. `flash_fwd_launch_template.h` and `flash_bwd_launch_template.h`: trait and
   compile-time feature dispatch.
9. `flash_fwd_kernel.h`, `flash_bwd_kernel.h`, `softmax.h`, and `dropout.h`:
   tiled device implementation.

## 15. Remaining Industrialization Work

The implementation is operational, but the following evidence is still needed
before calling the complete matrix production-qualified:

- compile bf16 D=32 and D=64, then link/test the combined workstation profile;
- compile the full profile on a high-memory build server;
- verify bf16 and every head bucket, including uneven D values;
- run benchmark sweeps over representative sequence lengths and both masks;
- profile spills, occupancy, tensor-core utilization, and memory throughput;
- stress non-default streams, repeated multithreaded loads, and long training;
- add Linux CI/build evidence and dedicated sm90/Hopper tuning.

These are validation and tuning tasks around a real upstream FlashAttention
kernel, not placeholders for missing CUDA forward or backward functionality.
