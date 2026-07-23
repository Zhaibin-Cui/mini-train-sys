# RTX 4090 benchmark and validation summary

Snapshot date: 2026-07-23 (Asia/Shanghai). Hardware: 4 × NVIDIA GeForce RTX 4090 24 GB; PyTorch
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

## SynBioS P/Q probe batch capacity

The formal probe workload uses the completed `multi5_permute` checkpoint and the largest-label
`university_whole` task. Two GPUs independently repeated P and two independently repeated Q;
training includes backbone forward, probe backward, and AdamW, while validation is forward-only.
Recommendations maximize mean throughput subject to at most 92% CUDA reserved memory and must not
be the largest tested candidate.

| Workload | Selected batch | Mean examples/s | Max reserved memory |
|---|---:|---:|---:|
| P training | 128 | 580.54 | 43.37% |
| Q training | 768 | 3,938.28 | 35.37% |
| P validation | 512 | 1,502.85 | 10.60% |
| Q validation | 6,144 | 10,699.55 | 14.86% |

The expanded search reports `ready_for_formal=true` with no boundary recommendation. A preceding
matrix separately verified the documented paper-default training batches P=50 and Q=200 on both
replicas (19.33% and 12.50% maximum reserved memory). The selected values are operational batch
choices and do not change the P/Q scientific task, optimizer, step budget, or person-level
train/validation split. See `../reports/synbios_moe/probes/capacity.md` and the exported raw
benchmark directory for the full evidence and limitations.

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

## Allen-Zhu P/Q probe pilots

### Single formal (completed)

The 22-task single formal run completed with zero failures (4,000 first / 12,000 whole steps;
P/Q batch 128/768). On the 50,118-person held-out split, P retains the expected positional
memory pattern: first-token accuracy is near chance at position 0 and reaches approximately
100% at the target's fixed position (birth_date is the fixed-first exception). Q remains near
class-prior accuracy, and whole P is only partially converged. This is a valid single baseline,
not evidence that the augmentation effect is absent; the matched multi5+permute formal run is the
next comparison. Full table and figure: `../reports/synbios_moe/probes/single_formal.md`.

Both 3,000-step, 22-task probe pilots completed all checkpoint-reloaded held-out validations with
zero failures. The person split is disjoint: P validates on 50,000 unseen single biographies or
250,000 unseen augmented biographies, and Q validates on 50,000 unseen names.

The main representation contrast is already decisive. Excluding birth date, whose fact is the
first fixed field, mean P position-0 first-token accuracy rises from 6.71% for `single` to 98.63%
for `multi5_permute`. Mean name-only Q first-token accuracy rises from 12.42% to 98.80%. The
single P heatmap is lower triangular: values stay near chance until the target's fixed position,
whereas augmented P exceeds 97% from position 0 for every first-token attribute.

Whole-attribute probes are not uniformly converged at pilot exposure. Multi5 mean Q whole
held-out accuracy is 32.34%, and some tasks retain large train/validation gaps, so formal
P/Q schedules preserve the paper's approximate sample exposure rather than stopping at 3,000
steps. See `../reports/synbios_moe/probes/pilot_comparison.md` and
`../reports/synbios_moe/probes/formal_protocol.md`.

### Matched formal single vs multi5+permute comparison

Both formal pipelines completed all 22 training heads and 22 checkpoint-reloaded person-held-out
validations with the same profile table, class mappings, model configuration, probe budgets, and
runtime settings. The first-token mechanism is a strong qualitative replication of Allen-Zhu
Part 3.1: excluding fixed-first birth date, mean P0 accuracy rises from 6.76% (`single`) to
98.63% (`multi5_permute`), while the six-attribute name-only Q-first macro rises from 12.83% to
98.79%. Single forms a fixed-order staircase and reaches 99.97% on its target-position diagonal;
multi5+permute is already 97.18%-99.76% accurate at P0.

Whole-attribute readability improves but does not reproduce the paper's dense-GPT2 result. Q-whole
macro rises from 3.18% to 33.15%, versus 92.58% for the paper's bioS multi5+permute row. Multi
Q-whole probe-train recall is 74.51%-98.18%, so its 8.48%-51.04% held-out range is a cross-person
linear-readout generalization gap, not simply failure to fit the probe training split. The formal
outcome is therefore a partial replication: first-token storage/extraction replicated;
whole-attribute linear readability not replicated on the MoE backbone. Canonical report and figures:
`../reports/synbios_moe/probes/formal_comparison.md`.

### Multi5+permute Q-whole inference diagnostics

The completed multi5+permute formal Q heads were evaluated on all 50,118 held-out people without
parameter updates. Inserting the ground-truth first attribute token before a new EOS did not
unlock the unchanged name-only Q-whole head: micro accuracy changed from 33.15% to 32.08%
(-1.06 percentage points). Major and company-city gained 2.38 and 2.73 points, but company lost
10.45 points; 5.38% of baseline errors recovered while 14.06% of baseline-correct predictions
were harmed.

The separate MoE diagnostic selected 162,044 Q-first-correct/Q-whole-wrong multi-token cases.
For same-attribute/same-first-token pairs, the top-2 route branching score
(`t1 overlap - t2 overlap`) was -0.051 in the same-second-token control and +0.154 when second
tokens differed, a +0.205 difference-in-differences. The pair-count-weighted contrast was positive
in all 12 layer aggregates and strongest in layers 0-3. This supports token-conditioned route
branching on the bad-case subset, but does not by itself identify experts as the causal fact store
or attribute the effect to augmentation. Canonical protocol, separate reports, full five-attribute
tables, and retained artifact paths are in
`../reports/synbios_moe/probes/diagnostics/README.md`. The complete colored comparison also includes
all six P positions, both original formal Q baselines, the oracle result, and Allen-Zhu Figure 7
context.
