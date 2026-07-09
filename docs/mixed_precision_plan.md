# Mixed Precision Plan

This note captures the current dtype state and the target design for making
MiniTrainSys run the same configurable precision policy across PyTorch, Triton,
and future CUDA backends. A future session should be able to read this file and
continue the implementation without rediscovering the dtype contract.

## Current State

The project currently trains in fp32 by default.

- `scripts/train.py` builds `MiniTransformer(model_cfg, ops).to(device)` without
  autocast, `GradScaler`, `.half()`, or `.to(torch.bfloat16)`.
- `TrainConfig` has no precision field.
- `Trainer.train_step()` directly calls forward, `loss.backward()`, and
  `optimizer.step()`.
- RoPE caches are initialized as fp32 buffers in `CausalSelfAttention`.
- The default configs use `backend.ops: torch`.
- The Triton backend overrides only RMSNorm, SwiGLU, and RoPE. Attention,
  cross entropy, and fused linear cross entropy still use PyTorch fallback.
- `minitrain/kernels/triton/cross_entropy.py` and
  `minitrain/kernels/triton/fused_linear_ce.py` are stubs.

The default runtime dtypes are therefore:

| Tensor | Current dtype |
| --- | --- |
| `input_ids`, `targets` | `torch.long` |
| model parameters | `torch.float32` |
| activations | `torch.float32` |
| attention Q/K/V | `torch.float32` |
| RoPE cos/sin buffers | `torch.float32` |
| logits | `torch.float32` |
| loss | `torch.float32` |

## AMP Rule To Preserve

Standard PyTorch AMP autocast is not the same as casting the model parameters.

In the target single-GPU/DDP AMP path:

```python
model = model.float()
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss, _ = model(...)
```

Parameters remain fp32 master parameters. Autocast selects lower precision for
eligible PyTorch ops such as matmul/linear, but it does not mutate
`nn.Parameter` storage. For custom Triton autograd functions, autocast does not
automatically rewrite all internal dtypes; the kernels see the tensor dtypes
that are actually passed to them.

Typical standard AMP inputs to a custom RMSNorm are:

| Argument | dtype under bf16 AMP |
| --- | --- |
| activation `x` | `torch.bfloat16` if the previous op produced bf16 |
| RMSNorm `weight` | `torch.float32` |
| upstream grad `dy` | activation dtype, usually bf16/fp16 |

Therefore custom kernels should return gradients matching parameter dtype, not
activation dtype. For RMSNorm:

```text
dweight accumulation: fp32
returned dweight dtype: weight.dtype
```

In standard AMP, `weight.dtype` is fp32, so the gradient is fp32. If a future
mode explicitly casts the model to bf16/fp16, then `weight.dtype` is bf16/fp16
and returning that dtype is consistent with that mode.

## Target Precision Config

Add one explicit precision policy to the runtime config:

```yaml
train:
  precision: fp32   # fp32 | bf16 | fp16
```

Suggested semantics:

| `train.precision` | Parameters | Forward compute | Grad scaling |
| --- | --- | --- | --- |
| `fp32` | fp32 | fp32 | disabled |
| `bf16` | fp32 under AMP | bf16 where eligible | disabled |
| `fp16` | fp32 under AMP | fp16 where eligible | enabled with `torch.amp.GradScaler("cuda")` |

Do not implement global `torch.set_default_dtype()` for training precision.
Keep initialization predictable in fp32 and let autocast or FSDP mixed precision
control compute dtype.

## Backend Dtype Contract

All backends should implement the same logical dtype behavior. This is the
contract tests should enforce.

### RMSNorm

Inputs:

- `x`: activation dtype, one of fp32/fp16/bf16.
- `weight`: parameter dtype. In standard AMP this remains fp32.

Forward:

- compute variance/reduction in fp32;
- multiply by weight with explicit behavior aligned between backends;
- output dtype equals `x.dtype`.

Backward:

- `dx` dtype equals `x.dtype`;
- accumulate `dweight` in fp32;
- return `dweight.to(weight.dtype)`.

Current status:

- PyTorch backend reduces in fp32 and returns activation dtype by casting the
  scale and weight to `x.dtype`.
- Triton backend supports fp32/fp16/bf16 input, stores output in
  `torch.empty_like(x)`, caches `rstd` in fp32, accumulates partial dW in fp32,
  and returns `partial_dw.sum(dim=0).to(dtype=weight.dtype)`.
- Check whether Triton forward should explicitly cast the loaded weight to the
  activation dtype for exact PyTorch parity, or whether both backends should
  define the internal multiply as fp32 and only store back to activation dtype.
  The chosen policy must be documented and tested.

### SwiGLU

Inputs:

- `gate`, `up`: same activation dtype.

Forward:

- compute sigmoid/silu using fp32 for the gate path;
- output dtype equals activation dtype.

Backward:

- gradients returned in the dtype expected by the upstream linear ops. Under
  standard AMP, PyTorch autograd will ultimately accumulate parameter grads for
  linear weights in fp32 master params.

Current status:

- PyTorch backend uses `F.silu(gate) * up` and relies on autocast behavior.
- Triton backend converts `gate` to fp32 for sigmoid/silu, casts silu back to
  `up.dtype`, multiplies by `up`, and stores into `empty_like(gate)`.

### RoPE

Inputs:

- `q`, `k`: activation dtype.
- `cos`, `sin`: should be cast to the same dtype as `q` before calling the
  backend.

Forward/backward:

- output Q/K dtype equals input Q/K dtype.

Current mismatch to fix:

- PyTorch backend casts cos/sin to `q.dtype`.
- Triton RoPE currently loads q/k and casts them to `cos_row.dtype`. Because
  the model creates fp32 cos/sin buffers, a bf16/fp16 activation path can make
  Triton RoPE compute closer to fp32 while PyTorch computes at activation dtype.

Preferred fix:

```python
cos = self.rope_cos[:seq_len].to(device=q.device, dtype=q.dtype)
sin = self.rope_sin[:seq_len].to(device=q.device, dtype=q.dtype)
q, k = ops.rope(q, k, cos, sin)
```

Then both torch and Triton see the same cos/sin dtype.

### Attention

Use `torch.nn.functional.scaled_dot_product_attention` for now. It chooses among
FlashAttention, memory-efficient attention, cuDNN attention, and math depending
on build, GPU, dtype, shape, mask/dropout, and backend availability.

The plan for replacing this path with a pretraining-only Triton
FlashAttention-style kernel is documented in
`docs/flash_attention_pretraining_plan.md`. That work should keep SDPA as the
reference and fallback path.

Target behavior:

- q/k/v dtype follows activation precision.
- bf16/fp16 are required for FlashAttention-style kernels.
- fp32 may fall back to memory-efficient or math depending on the environment.
- Add a profiler utility to log the actual selected SDPA backend during
  benchmark runs.

### Cross Entropy And Fused Linear Cross Entropy

Current backend is PyTorch fallback only.

Target behavior:

- materialized logits may be bf16/fp16 under AMP, but numerically sensitive
  reductions should be fp32;
- fused linear CE should avoid materializing `[tokens, vocab]` logits where
  possible;
- loss output should remain fp32.

Implement these after the global precision policy and dtype tests are in place.

## FSDP Target

The current FSDP strategy does not configure mixed precision. Add config fields
later under `distributed`, for example:

```yaml
distributed:
  mixed_precision:
    param_dtype: bf16
    reduce_dtype: bf16
    buffer_dtype: bf16
```

For FSDP mixed precision, use PyTorch `MixedPrecision` rather than wrapping the
entire trainer in the same way as single-GPU AMP. The exact integration should
live in `minitrain/distributed/fsdp.py` so the trainer stays strategy-agnostic.

Single-GPU/DDP can start with autocast in the trainer. FSDP should own its own
precision policy once the strategy grows beyond the current scaffold.

## Implementation Checklist

1. Add `precision: str = "fp32"` to `TrainConfig`.
2. Update YAML configs with `train.precision: fp32` for current behavior.
3. Add a helper that maps precision strings to AMP settings:
   - `fp32`: autocast disabled, scaler disabled;
   - `bf16`: autocast dtype `torch.bfloat16`, scaler disabled;
   - `fp16`: autocast dtype `torch.float16`, scaler enabled.
4. Update `Trainer` to run forward/loss under autocast when enabled.
5. Update `Trainer` to use `torch.amp.GradScaler("cuda")` for fp16 only.
6. Keep backward outside the autocast region.
7. Update `scripts/train.py` to pass the precision policy into `Trainer`.
8. Log `precision`, AMP dtype, and scaler enabled state in the init event.
9. Cast RoPE cos/sin to `q.dtype` before `ops.rope(...)`.
10. Decide and document the exact RMSNorm weight multiply policy; then align
    PyTorch and Triton implementations.
11. Add dtype sweep tests for `torch` and `triton` providers:
    - fp32;
    - bf16 if CUDA supports it;
    - fp16 if CUDA supports it.
12. Add benchmark dtype as a dimension in `operator_bench_utils.py` and related
    notebooks/reports.
13. Add an SDPA backend probe to training benchmark metadata.
14. Add FSDP mixed precision config after single-GPU/DDP AMP is stable.
15. Implement Triton cross entropy and fused linear CE after the dtype contract
    is covered by tests.

## Minimal Trainer Shape

The trainer should eventually look like this at a high level:

```python
with autocast_context:
    loss, _ = model(input_ids, targets=targets, use_fused_loss=use_fused_loss)

if scaler.is_enabled():
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
```

Keep `optimizer.zero_grad(set_to_none=True)` outside this branch as it is now.

## Verification Commands

Environment and PyTorch precision capability:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("cap", torch.cuda.get_device_capability(0))
    print("bf16 supported", torch.cuda.is_bf16_supported())
PY
```

SDPA backend probe:

```bash
python - <<'PY'
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    torch.cuda.synchronize()

for event in prof.key_averages():
    if (
        "scaled_dot_product" in event.key
        or "flash" in event.key
        or "efficient" in event.key
        or "cudnn" in event.key
    ):
        print(event.key)
PY
```

Expected names:

- `aten::_scaled_dot_product_flash_attention` means PyTorch SDPA used flash.
- `aten::_scaled_dot_product_efficient_attention` means memory-efficient.
- math/cudnn names indicate the corresponding fallback.

## Server Migration Notes

For industry-like large-scale mixed precision benchmarks, prefer Linux servers
with Ampere/Ada/Hopper GPUs. For FlashAttention through PyTorch SDPA, confirm
that the PyTorch build reports FlashAttention availability and that profiler
output shows the flash kernel. For the standalone `flash-attn` package, Linux,
PyTorch 2.2+, CUDA toolkit 12.0+, `ninja`, `packaging`, and `psutil` are the
expected baseline requirements.

Do not trust a benchmark run unless it records:

- GPU name and compute capability;
- driver version;
- PyTorch version and CUDA runtime;
- `train.precision`;
- ops backend;
- distributed strategy;
- actual SDPA backend selected;
- tokens/sec, step time, peak VRAM, and loss.
