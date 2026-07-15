# Configuration

Run config and model config are separate. The supplied presets are intentionally small:

- `train_debug.yaml`: CPU/quick validation, fixed LR, no warmup.
- `train_single.yaml`: single-GPU LLM default, warmup + cosine decay.
- `train_ddp.yaml`: multi-GPU DDP.
- `train_fsdp.yaml`: multi-GPU FSDP.
- `model_default.yaml`: default LLaMA-style dense model (32K vocab, 2K context).
- `model_debug_{dense,moe}.yaml`: tiny architecture checks.
- `model_15m_dense.yaml`: fast end-to-end experiments.
- `model_125m_{dense,moe}.yaml`: representative dense/MoE training.

The default model uses 12 layers, 12 heads, hidden size 768, SwiGLU size 2048,
RMSNorm epsilon `1e-5`, RoPE theta `10000`, no dropout, and untied input/output
embeddings. Debug presets remain deliberately small and are never selected implicitly.

## LR recipes

```yaml
# Industrial LLM default: linear warmup, then cosine to 10% of peak LR.
lr_scheduler: {schedule: cosine, warmup_steps: 100, decay_steps: null, min_lr_ratio: 0.1}

# Fixed LR, no warmup.
lr_scheduler: {schedule: constant, warmup_steps: 0, decay_steps: null, min_lr_ratio: 0.1}

# Fixed LR after warmup.
lr_scheduler: {schedule: constant, warmup_steps: 100, decay_steps: null, min_lr_ratio: 0.1}

# Cosine immediately, without warmup.
lr_scheduler: {schedule: cosine, warmup_steps: 0, decay_steps: null, min_lr_ratio: 0.1}
```

`decay_steps: null` derives the decay horizon from the effective training limit. When both
`max_steps` and `epochs` are set, the first reached limit wins.

## All options

| Key | Values / meaning |
| --- | --- |
| `run.name` | Run/checkpoint/log name. |
| `run.seed` | Python and Torch seed. |
| `backend.ops` | `torch`, `triton`, or `cuda`. |
| `backend.parallel` | `single`, `ddp`, or `fsdp`. |
| `optimizer.name` | `adamw` (currently the supported optimizer). |
| `optimizer.lr` | Peak/fixed learning rate. |
| `optimizer.weight_decay` | Applied only to matrix-like parameters; norms and biases use zero decay. |
| `optimizer.beta1`, `beta2`, `eps` | AdamW numerical settings. LLM defaults: `0.9`, `0.95`, `1e-8`. |
| `optimizer.fused` | `null` auto-selects fused AdamW on CUDA; `true`/`false` forces it. |
| `lr_scheduler.schedule` | `constant` or `cosine`. |
| `lr_scheduler.warmup_steps` | `0` disables warmup; positive values enable linear warmup. |
| `lr_scheduler.decay_steps` | Explicit cosine end step, or `null` for the effective total steps. |
| `lr_scheduler.min_lr_ratio` | Final LR divided by peak LR, from `0` to `1`. |
| `train.batch_size` | Per-process micro-batch size. |
| `train.max_steps` | Total optimizer-step limit, or `null`. |
| `train.epochs` | Total completed-epoch limit, or `null`; cannot be null with `max_steps`. |
| `train.log_interval` | Log every N optimizer steps. |
| `train.use_fused_loss` | Use backend fused LM loss when available. |
| `train.precision` | `fp32`, `bf16`, or `fp16`. |
| `train.grad_clip_norm` | Global gradient norm limit, or `null` to disable. |
| `train.checkpoint_every_epochs` | Save every N completed epochs, or `null`. |
| `train.checkpoint_dir` | Checkpoint root directory. |
| `train.save_final_checkpoint` | Save final/partial state on normal completion. |
| `train.resume_from` | `null`, `latest`, or an explicit `.pt` path. |
| `data.source` | `random` or `tokens`. |
| `data.path` | `.pt`, `.pth`, `.npy`, or uint16 `.bin`; required for `tokens`. |
| `data.num_tokens` | Synthetic token count for `random`. |
| `data.shuffle` | Shuffle samples each epoch. |
| `logging.console` | Enable rank-0 console events. |
| `logging.tensorboard` | Enable rank-0 TensorBoard logging. |
| `logging.log_dir`, `flush_secs` | TensorBoard root and flush interval. |
| `distributed.bucket_cap_mb` | DDP communication bucket size. |
| `distributed.gradient_as_bucket_view` | DDP gradient bucket views. |
| `distributed.sharding_strategy` | FSDP: `full_shard`, `shard_grad_op`, `no_shard`, `hybrid_shard`. |
| `model.vocab_size`, `seq_len` | Vocabulary and context size. |
| `model.n_layers`, `n_heads`, `hidden_size`, `intermediate_size` | Transformer dimensions. |
| `model.dropout` | Dropout probability. |
| `model.ffn_type` | `dense` or `moe`. |
| `model.num_experts`, `experts_per_token` | MoE expert count and top-k routing. |
| `model.router_aux_loss_coef` | MoE load-balancing loss coefficient. |
| `model.router_z_loss_coef` | Router logit stabilization loss coefficient. |
| `model.router_normalize_topk` | Renormalize selected expert weights to sum to one. |
| `model.router_jitter_noise` | Multiplicative training-only router input jitter; `0` disables it. |
| `model.expert_capacity_factor` | Reserved; currently ignored by `TopKRouter.forward`, which always uses dropless routing. |
| `model.expert_min_capacity` | Reserved minimum capacity for a future capacity-aware compact dispatch implementation. |

## Run

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_train.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_train.ps1 -ModelConfig configs/model_debug_moe.yaml -Resume latest
```

Linux:

```bash
bash scripts/run_train.sh
MODEL_CONFIG=configs/model_debug_moe.yaml RESUME=latest bash scripts/run_train.sh
NPROC=8 bash scripts/run_distributed.sh ddp
NPROC=8 bash scripts/run_distributed.sh fsdp
```
