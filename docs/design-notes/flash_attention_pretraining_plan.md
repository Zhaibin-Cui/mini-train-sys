# FlashAttention Pretraining Plan

> 历史设计记录：核心实现已经进入 Triton/CUDA backend。当前运行方式见
> [`cuda_ext_run_commands.md`](../kernels/cuda_ext_run_commands.md)，结果见
> [`cuda_flash_attention_learning_report.md`](../kernels/cuda_flash_attention_learning_report.md)。

This note is the handoff document for implementing a Triton FlashAttention-style
training kernel inside MiniTrainSys. A future session should be able to start
from this file, inspect the named files, and continue without rediscovering the
scope.

## Scope

The target is a pretraining-only attention kernel:

```python
y = triton_flash_attn(q, k, v, causal=True, dropout_p=0.0)
```

It is for decoder-only training. It is not an inference-serving project.

In scope:

- dense causal self-attention for pretraining;
- forward and backward;
- bf16 and fp16 CUDA tensors;
- head dimensions 64 and 128 first;
- PyTorch SDPA as the correctness and fallback path;
- benchmarks for latency, memory, and end-to-end training impact.

Out of scope for the first implementation:

- KV cache;
- paged attention;
- decode kernels;
- mixed prefill/decode serving;
- prefix cache;
- speculative decoding;
- arbitrary masks;
- FP8;
- Hopper TMA or Blackwell-specific CuTeDSL optimization.

Those features belong to inference systems such as FlashInfer, vLLM, and
TensorRT-LLM. They should not block a useful pretraining kernel.

## Why Not Start From Neighborhood Attention

The upstream fused-neighborhood-attention implementation is useful
as a Triton engineering example, but it is not the right base for FlashAttention.

That neighborhood implementation allocates full sequence-by-sequence
intermediates:

- `qk_scores` has shape `[batch, heads, seq, seq]`;
- the neighborhood mask has shape `[seq, seq]`;
- backward saves `attn_weights` and creates full attention-gradient tensors.

That design can be faster than a naive PyTorch reference for local attention, but
it is not the FlashAttention memory model. FlashAttention avoids materializing
the full attention matrix. It streams K/V blocks, maintains online softmax state,
and saves log-sum-exp data for backward.

Use existing kernel projects for structure ideas:

- one kernel family per file;
- explicit autograd function;
- focused correctness tests;
- benchmark scripts and reports.

Do not copy its full `[S, S]` allocation pattern for this work.

## Best Starting Point

Start from the official Triton fused attention tutorial:

- URL: `https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html`
- It is a Triton implementation of the FlashAttention v2 algorithm.
- It has forward and backward.
- It uses online softmax state (`m_i`, `l_i`) and stores log-sum-exp metadata.
- It includes autotune configs and benchmark structure.

Treat the tutorial as the algorithmic skeleton, then adapt it to MiniTrainSys
style and tests.

## Repository Integration Points

Current attention path:

- `minitrain/kernels/torch_ops.py`
  - `TorchOpsBackend.attention(...)` calls
    `torch.nn.functional.scaled_dot_product_attention`.
- `minitrain/kernels/triton_ops.py`
  - Triton backend currently overrides some ops and falls back for attention.
- `minitrain/model/transformer.py`
  - model should call only the `OpsBackend`, not Triton directly.
- `tests/operator_bench_utils.py`
  - current operator benchmark harness.
- `reports/operator_bench.md`
  - place to summarize attention kernel benchmark results.
- `docs/training/mixed_precision_plan.md`
  - dtype contract and SDPA backend probe notes.

Expected new files:

- `minitrain/kernels/triton/flash_attention.py`
- `tests/test_triton_flash_attention.py`
- optional: `tests/bench_flash_attention.py`
- optional: update `tests/operator_bench_utils.py` with an `attention` case

Expected existing files to edit:

- `minitrain/kernels/triton_ops.py`
  - route attention to Triton implementation when supported;
  - fall back to PyTorch SDPA otherwise.
- `minitrain/kernels/base.py` or equivalent backend protocol file
  - only if the attention contract needs a minor signature change.
- `reports/operator_bench.md`
  - record benchmark methodology and results.

Keep the model layer backend-agnostic. `MiniTransformer` should not import
`triton` or the Triton kernel module.

## Interface Contract

Start with the same logical contract as `TorchOpsBackend.attention`:

```python
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    dropout_p: float,
) -> torch.Tensor:
    ...
```

Initial supported layout:

```text
q, k, v: [batch, heads, seq, head_dim]
out:     [batch, heads, seq, head_dim]
```

Initial supported values:

- `is_causal=True`;
- `dropout_p=0.0`;
- `q.shape == k.shape == v.shape`;
- `q.dtype in {torch.float16, torch.bfloat16}`;
- CUDA tensors only;
- `head_dim in {64, 128}` first;
- contiguous last dimension.

Unsupported inputs should fall back to PyTorch SDPA, not crash the training run.
For tests that specifically validate the Triton kernel, unsupported inputs should
raise a clear `NotImplementedError` or return a structured "unsupported" reason.

## Algorithm Notes

Forward should follow the FlashAttention pattern:

1. Split Q into blocks of query rows.
2. Keep one Q block resident while streaming K/V blocks.
3. Compute `qk = Q @ K.T * scale`.
4. Apply causal masking inside the block loop.
5. Maintain online softmax row state:
   - running max `m`;
   - running denominator `l`;
   - output accumulator `acc`.
6. Update `acc` as each K/V block is processed.
7. Store output and log-sum-exp metadata.

Backward should follow the recompute pattern:

1. Use saved Q/K/V, output, and log-sum-exp metadata.
2. Recompute QK blocks instead of saving attention weights.
3. Compute dV, dK, and dQ in tiled kernels.
4. Do not save or allocate full `[batch, heads, seq, seq]` attention weights.

If a proposed implementation stores a full attention matrix, it is no longer the
target kernel. Stop and redesign before optimizing.

## Milestones

### M0: Baseline Probe

Goal: know what PyTorch SDPA is doing on the current machine.

Deliverables:

- Add or reuse an SDPA backend probe from `docs/training/mixed_precision_plan.md`.
- Record GPU, driver, PyTorch, CUDA runtime, dtype, shape, and selected SDPA
  backend in benchmark output.

Useful command:

```bash
python - <<'PY'
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

assert torch.cuda.is_available()
q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    torch.cuda.synchronize()

for event in prof.key_averages():
    key = event.key.lower()
    if "scaled_dot_product" in key or "flash" in key or "efficient" in key or "cudnn" in key:
        print(event.key)
PY
```

### M1: Forward Only

Goal: implement Triton forward for a narrow shape set.

Supported matrix:

| batch | heads | seq | head_dim | dtype |
| --- | --- | --- | --- | --- |
| 1, 2 | 8 | 128, 512, 1024 | 64 | fp16, bf16 |
| 1, 2 | 8 | 128, 512, 1024 | 128 | fp16, bf16 |

Correctness:

- Compare against `F.scaled_dot_product_attention(..., is_causal=True)`.
- Use loose initial tolerances, then tighten after dtype policy is clear:
  - fp16: start with `atol=2e-2`, `rtol=2e-2`;
  - bf16: start with `atol=3e-2`, `rtol=3e-2`.
- Check output shape, dtype, finite values, and max absolute error.

Implementation rule:

- No full `[B, H, S, S]` allocation.

### M2: Autograd Backward

Goal: implement enough backward to train.

Deliverables:

- `torch.autograd.Function` wrapper.
- `gradcheck`-style logic where practical, plus direct comparison of `dq/dk/dv`
  against SDPA for fp16/bf16.
- Backward does not save full attention weights.

Correctness checks:

- Compare `dq`, `dk`, `dv`.
- Run multiple upstream gradient patterns:
  - `out.sum()`;
  - random `dy`;
  - loss from a tiny transformer step.

### M3: Backend Integration

Goal: make `backend.ops: triton` use the Triton attention when supported.

Rules:

- `TorchOpsBackend` remains the reference.
- Triton backend calls the custom attention only for supported shapes.
- Unsupported shapes fall back to SDPA.
- Log or expose which attention path ran during benchmarks:
  - `torch_sdpa`;
  - `triton_flash_attention`;
  - `triton_flash_attention_fallback_sdpa`.

### M4: Operator Benchmark

Goal: compare operator latency and memory.

Shape sweep:

| batch | heads | seq | head_dim | dtype |
| --- | --- | --- | --- | --- |
| 1 | 8 | 1024, 2048, 4096, 8192 | 64 | bf16 |
| 2 | 16 | 1024, 2048, 4096 | 64 | bf16 |
| 1 | 8 | 1024, 2048, 4096 | 128 | bf16 |

Metrics:

- forward p50/p95;
- backward p50/p95;
- full forward+backward p50/p95;
- peak CUDA memory;
- max error for output and gradients;
- actual PyTorch SDPA backend selected for the reference.

Do not trust speed numbers unless correctness was checked in the same script or
same report entry.

### M5: End-To-End Training Smoke

Goal: prove the kernel can run in the model path.

Run:

- tiny model, short CUDA smoke;
- one small model for enough steps to observe loss movement;
- compare torch vs triton attention with the same seed and config.

Record:

- loss curve for a short run;
- tokens/sec;
- step time;
- peak memory;
- attention path used;
- whether any fallback occurred.

## Features To Add After M5

Priority order for pretraining:

1. GQA/MQA
   - support `Hq != Hkv`;
   - common in modern LLM pretraining;
   - repeat/interleave K/V logically without materializing repeated K/V if possible.
2. Variable-length packed sequences
   - use `cu_seqlens`-style metadata;
   - needed to avoid cross-document attention leakage when packing documents.
3. Dropout
   - often zero for modern LLMs, but training code should eventually support it;
   - handle RNG determinism carefully with backward recomputation.
4. Sliding window attention
   - only if the model config needs Mistral/Gemma/Qwen-like local layers.
5. Head dimension 256
   - useful later, but not part of the first correctness target.

Avoid adding FP8 or inference-only features before GQA and packed sequences.

## Tests To Write

Create `tests/test_triton_flash_attention.py`.

Minimum tests:

- `test_forward_matches_sdpa`
  - shape and dtype sweep;
  - causal only.
- `test_backward_matches_sdpa`
  - compare `dq`, `dk`, `dv`;
  - random upstream gradient.
- `test_unsupported_falls_back_or_reports`
  - CPU tensors;
  - fp32 tensors;
  - unsupported `head_dim`;
  - `dropout_p > 0` before dropout is implemented.
- `test_no_full_attention_allocation_contract`
  - this may start as a code review checklist in the test docstring;
  - later, memory benchmarks should catch regressions.
- `test_backend_contract_attention`
  - `TorchOpsBackend` and `TritonOpsBackend` return same shape/dtype and close
    values for supported shapes.

Skip GPU-only tests cleanly when CUDA or Triton is unavailable.

## Benchmark Report Template

Add entries to `reports/operator_bench.md` using this shape:

```markdown
## Triton FlashAttention

Environment:
- GPU:
- compute capability:
- driver:
- torch:
- torch CUDA:
- triton:

Kernel status:
- supported shapes:
- unsupported fallback rules:
- SDPA backend selected:

Correctness:
| dtype | shape | max out err | max dq err | max dk err | max dv err |
| --- | --- | --- | --- | --- | --- |

Speed:
| provider | mode | batch | heads | seq | dim | dtype | p50 ms | p95 ms | peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Caveats:
- ...
```

## Common Failure Modes

- Full `[S, S]` allocation slipped back in.
  - This defeats the purpose. Redesign around online softmax.
- Forward passes but backward is numerically loose.
  - Recheck log-sum-exp convention, scale placement, causal mask alignment, and
    dtype casts.
- SDPA reference silently uses math backend.
  - Benchmark metadata must record the actual backend.
- Triton backend is slower for tiny sequence lengths.
  - That is acceptable. The target is long-context pretraining shapes.
- Windows environment blocks Triton work.
  - Use Linux CUDA for serious Triton benchmarking. CPU smoke tests can still run
    on Windows through the torch backend.

## New Session Startup Checklist

When continuing this work in a new session:

1. Read this file.
2. Read `docs/training/mixed_precision_plan.md`, especially the Attention section.
3. Inspect:
   - `minitrain/kernels/torch_ops.py`;
   - `minitrain/kernels/triton_ops.py`;
   - `minitrain/model/transformer.py`;
   - `tests/operator_bench_utils.py`;
   - `reports/operator_bench.md`.
4. Check current git diff before editing.
5. Confirm CUDA, PyTorch, and Triton availability.
6. Start with M0 if benchmark metadata is missing; otherwise continue the first
   incomplete milestone.

The first concrete coding task should be:

```text
Add `minitrain/kernels/triton/flash_attention.py` with a forward-only
FlashAttention-style Triton implementation for causal bf16/fp16 tensors shaped
[B, H, S, D], D in {64, 128}, and tests against PyTorch SDPA.
```
