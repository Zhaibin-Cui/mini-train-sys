# RTX 4090 benchmark and validation summary

Snapshot date: 2026-07-21 (Asia/Shanghai). Hardware: 4 × NVIDIA GeForce RTX 4090 24 GB; PyTorch
2.5.1+cu118; CUDA Toolkit 11.8; Triton 3.1.0; BF16.

## Formal SynBioS model

The exact model has 293,494,272 parameters, sequence length 512, 12 transformer layers, eight MoE
experts, and top-2 routing. Capacity cases include forward, fused loss, backward, and AdamW update.

| FSDP local batch | Global batch (4 GPU) | Peak allocated | Throughput | Result |
|---:|---:|---:|---:|---|
| 96 | 384 | 74.56% | 365,034 tok/s | pass |
| 112 | 448 | 86.20% | 370,857 tok/s | pass; selected |
| 120 | 480 | 92.02% | 281,926 tok/s | pass; slower and less margin |
| 128 | 512 | — | — | expected backward OOM |

Batch 112 subsequently completed 55 consecutive full steps at 86.20% peak allocated memory and
368,170 tok/s. It is the persisted formal local batch.

## FSDP weak scaling at local batch 64

| GPUs | Mean throughput | Weak-scaling efficiency |
|---:|---:|---:|
| 1 | 93,302 tok/s | baseline |
| 4 | 344,254 tok/s | 92.24% |

Both four-GPU repeats passed (92.08% and 92.40%); data stall was 0.09–0.12%.

## Generic 125M-class server matrix

The notebook benchmark covered single-device, DDP, and FSDP configurations on the installed
1/4-GPU topology. The extended capacity sweep completed 30 cases with 10 expected OOM boundary
cases. At local batch 32, DDP-4 reached 248,200 tok/s at 92.65% peak allocated memory and FSDP-4
reached 278,760 tok/s at 66.18%. These generic results were used for infrastructure validation;
the exact SynBioS benchmark above is authoritative for formal training.

## Correctness and recovery gates

- Full repository regression after checkpoint/logging optimization: 70 passed, 0 failed.
- Ruff: passed.
- Exact real-data FSDP save/restore advanced step 3 to step 5 with model, AdamW, scheduler, counters,
  and RNG restored.
- LR continuity: step 3 `6.542e-05`, resumed step 4 `8.723e-05`, step 5 `1.090e-04`.
- Formal data: 100,000 accepted synthetic biographies and 7,405,102 tokens; all recorded manifest
  SHA256 values were revalidated.
- Optimized recovery checkpoint: epoch 10/step 320, 3,677,964,267 bytes, atomically committed in
  26.929 seconds without the redundant 1.3 GB probe export.

## Formal convergence pilot status

The first optimized 540-epoch launch was stopped safely at epoch 30/step 960. Loss fell to about
1.10 around epochs 10–20, then persistently increased to 5.38 while unclipped gradient norms rose
to hundreds or thousands (occasionally tens of thousands) and clipping remained continuously
active. This rejects the linearly scaled peak LR `0.004667` for global batch 448; the run is
retained as failure evidence, not presented as a successful formal result. A lower-LR convergence
sweep is required before the formal restart.

## Paper-fidelity restart and corrected GPU telemetry

The rejected linear scaling was removed. The new formal run keeps local/global batch 112/448 for
the validated hardware throughput but fixes AdamW peak LR at `1e-3`, warmup at 1,000 optimizer
steps, and cosine floor at `1e-4`, matching the paper hyperparameters rather than multiplying LR
by `448/96`.

A fresh 64-step 4-GPU FSDP preflight completed without NaN/Inf: loss fell from 10.94644 to
3.68811, the final unclipped grad norm was 1.438, and the final clipping signal was zero. Steady
throughput was 353k-358k token/s. The corrected NVML compute-utilization metric averaged 96.89%
over all logged steps and about 97%-99% after startup; interval peak allocated memory was 86.2%.
The previous `gpu_memory_utilization_percent_max=4.71%` was a post-step PyTorch allocator ratio,
not compute utilization, and has been replaced by explicitly named current/reserved/interval-peak
memory percentages plus NVML compute and memory-controller min/mean/max metrics.

After manifest size/SHA256 verification of all six generated/token data files, the failed active
formal checkpoints and logs were cleared and a new random-initialized FSDP4 run started at step 0.
It completed all 540 epochs / 17,280 optimizer steps and 3,963,617,280 scheduled tokens. Logged
loss fell from 10.94644 to 0.193221 (minimum 0.192083), final grad norm was 0.02456, mean logged
NVML compute utilization was 97.02%, average end-to-end throughput was 312,868 tok/s, and interval
peak allocated memory remained 86.2%. The final atomically committed checkpoint includes the full
FSDP model/Adam recovery state and a separate `model.pt` export. The old failed run remains only
as historical Git evidence and in recoverable mounted-volume trash; it cannot be auto-resumed or
mixed into the successful TensorBoard directory.

Raw evidence is under `benchmarks/`, `validation/`, and `logs/`; exact commands and stopped/failed
runs are preserved in `../HISTORY.md`.

## Single-biography progressive cloze recall

The final `single` checkpoint was evaluated on all 100,000 original training biographies by
removing their six ground-truth fact spans and greedily restoring them in source-text order. Each
earlier model prediction was inserted into the context used for later fields. Across 600,000
fields, strict case-sensitive exact accuracy was 100% for birth date, birth city, university,
major, company, and company city; all 100,000 biographies restored all six fields exactly.
Normalized Levenshtein character similarity and its 50%/80%/90% thresholds were also 100%, but
are only diagnostics because approximate string matching can over-credit semantically wrong near
matches. This training-set test demonstrates exact in-distribution recall, not generalization to
unseen people or templates. See `../reports/synbios_single_cloze_100k.md` for protocol, examples,
limitations, and result paths.

## Multi5+permute pretraining and progressive cloze recall

The augmentation condition used the same 100,000 person profiles as `single`, but rendered five
independently worded and field-permuted biographies per person (500,000 biographies and 37,046,556
tokens). The same model, FSDP4, BF16, global batch 448, and LR schedule completed 108 epochs / 17,388
steps / 3,988,389,888 scheduled tokens. Loss fell from 10.948931 to 0.296150 (minimum logged
0.293688), final grad norm was 0.06513, no experts were dead, no routes were dropped, end-to-end
time was 12,148.31 seconds, and average throughput was 328,308 tok/s.

The final checkpoint was evaluated with the identical progressive original-biography cloze
protocol on all 500,000 training texts. It restored 2,999,746/3,000,000 fields exactly
(99.991533%) and all six fields in 499,813/500,000 biographies (99.9626%). Per-field exact
accuracies were 99.9968% birth date, 99.9966% birth city, 99.9952% university, 99.9710% major,
99.9946% company, and 99.9950% company city. Fuzzy accuracy at threshold 0.90 was 99.991633%,
three fields higher than strict matching, so strict exact remains authoritative.

This validates optimization and near-complete augmented-training-corpus recall. It does not show
held-out improvement over `single`, whose strict training-corpus score was 100%. Document packing
uses a 1,024-document shuffle window: documents remain intact, but the order is a bounded shuffle,
not a uniform global permutation over all 500,000 rows. Full methods and examples are in
`../reports/synbios_multi5_permute_cloze_500k.md`.
