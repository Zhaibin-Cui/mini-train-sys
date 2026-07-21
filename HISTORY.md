# MiniTrainSys run history

This append-only log records formal training, evaluation, benchmark, probe, and dataset-generation
runs on the experiment server. Times are Asia/Shanghai unless explicitly marked UTC.

## 2026-07-21 03:12:55 — 4-GPU distributed server benchmark

- Status: completed (2026-07-21 03:24 Asia/Shanghai)
- Purpose: Execute the full 1/4-GPU DDP/FSDP weak-scaling and capacity benchmark with cell outputs
  retained in the source notebook.
- Git commit: `bde795e3daae7b92be2a7a6e48c2e7ccb8d7bcb1` (`train` branch; dirty working tree containing
  machine adaptation and benchmark fixes)
- Hardware: 4 × NVIDIA GeForce RTX 4090 24 GB, compute capability 8.9
- Runtime: Python 3.10.12, PyTorch 2.5.1+cu118, Triton 3.1.0, CUDA Toolkit 11.8
- tmux session: `minitrain-distributed-benchmark`
- Command:

  ```bash
  cd /home/ubuntu/mini-train-sys
  source .minitrain-storage.env
  source .venv/bin/activate
  export MPLBACKEND=Agg
  jupyter nbconvert --to notebook --execute --inplace \
    tests/distributed_server_benchmark.ipynb \
    --ExecutePreprocessor.kernel_name=mini-train-sys \
    --ExecutePreprocessor.timeout=-1
  ```

- Notebook: `tests/distributed_server_benchmark.ipynb`
- Runtime log: `artifacts/logs/distributed_server_benchmark_20260721_031255.log`
- Raw results: `artifacts/distributed_benchmark/rtx4090_125m_moe/`
- Git-exported results (populated after completion): `results/distributed_benchmark/`
- Machine overrides: notebook world sizes changed from `(1, 4, 8)` to `(1, 4)`; formal training
  policy defaults to 4-GPU FSDP.
- Result: weak suite 12/12 successful and initial capacity suite 24/24 successful through batch
  32. The notebook was written in place with cell outputs.

## 2026-07-21 03:28 — Extended capacity benchmark

- Status: completed (2026-07-21 03:48 Asia/Shanghai)
- Purpose: Add a true single-device baseline and extend full-training capacity points through
  local batch 128 to determine the largest safe batch for single 1-GPU, DDP 4-GPU, and FSDP
  4-GPU training.
- Git commit: `bde795e3daae7b92be2a7a6e48c2e7ccb8d7bcb1` (`train` branch; dirty working tree with the
  benchmark fixes and machine-specific settings documented above)
- Hardware/runtime: 4 × RTX 4090 24 GB; Python 3.10.12; PyTorch 2.5.1+cu118; Triton 3.1.0
- tmux session: `minitrain-distributed-benchmark`
- Command: same in-place `jupyter nbconvert` command recorded above, using the
  `mini-train-sys` kernelspec.
- Capacity points: `1, 2, 4, 8, 16, 32, 64, 128`
- Valid topologies: single 1-GPU; DDP 1/4-GPU; FSDP 1/4-GPU
- Safety rule: initially 90%; changed by user decision to select the largest successful local
  batch with peak allocated GPU memory at or below 95% (approximately 92% is explicitly accepted
  on this server), then run multi-step stability and checkpoint/resume correctness gates before
  formal training.
- Runtime log: `artifacts/logs/distributed_server_benchmark_20260721_032837.log`
- Raw results: `artifacts/distributed_benchmark/rtx4090_125m_moe/`
- Result: weak suite 15/15 successful; capacity suite 30 successful and 10 expected OOM boundary
  cases. single, DDP 1/4-GPU, and FSDP 1/4-GPU all succeeded through local batch 32; batch 64 and
  128 OOM for each tested topology. FSDP 4-GPU batch 32 reached 278,760 tokens/s at 66.18% peak
  allocated memory. Batch-4 FSDP weak-scaling efficiency was 44.22%, so formal training remains
  blocked pending a larger-batch, formal-model benchmark.
- Git export: `results/distributed_benchmark/rtx4090_125m_moe/` and `results/logs/` (115 files,
  approximately 1.2 MiB total at export time).

## 2026-07-21 03:51:59 — SynBioS formal-model FSDP 4-GPU capacity bracket

- Status: completed (2026-07-21 03:55 Asia/Shanghai)
- Purpose: bracket the full-training local-batch capacity for the exact SynBioS model before a
  non-power-of-two refinement and formal pretraining launch.
- Git commit: `bde795e3daae7b92be2a7a6e48c2e7ccb8d7bcb1` (`train`, dirty with documented benchmark fixes)
- Hardware/runtime: 4 × RTX 4090 24 GB; PyTorch 2.5.1+cu118; Triton 3.1.0; BF16; FSDP full shard
- tmux session: `minitrain-synbios-capacity`
- Command:

  ```bash
  python scripts/run_dist_bench.py run --suite capacity \
    --strategies fsdp --world-sizes 4 \
    --batch-sizes 8 16 32 64 128 256 512 \
    --warmup-steps 3 --measure-steps 5 --repeats 1 \
    --model-config configs/synbios_moe/model.yaml \
    --output artifacts/distributed_benchmark/synbios_moe_fsdp4/capacity_initial
  ```

- Result path: `artifacts/distributed_benchmark/synbios_moe_fsdp4/capacity_initial/`
- Runtime log: `artifacts/logs/synbios_moe_fsdp4_capacity_initial.log`
- Result: local batches 8, 16, 32, and 64 completed full forward, loss, backward, and optimizer
  steps; 128, 256, and 512 reached the expected CUDA OOM boundary. Batch 64 used 12,413 MiB
  peak allocated memory (51.26%) and sustained 343,909 tokens/s. The maximum lies in `[65, 127]`
  and is refined below without restricting candidates to powers of two.

## 2026-07-21 03:57 — SynBioS 8-aligned FSDP capacity refinement

- Status: completed (2026-07-21 03:58 Asia/Shanghai; stopped after the user narrowed the
  selection rule to a safe multiple of eight)
- Purpose: identify the largest safe local batch for the exact SynBioS model on 4-GPU FSDP,
  including backward and optimizer steps, using candidates aligned to multiples of eight.
- Hardware/runtime: 4 × RTX 4090 24 GB; PyTorch 2.5.1+cu118; BF16; FSDP full shard
- Safety rule: largest successful batch with peak allocated memory at or below 95%, followed by
  multi-step formal-config stability and checkpoint/resume validation.
- tmux session: `minitrain-synbios-capacity-refine`
- Command:

  ```bash
  python scripts/run_dist_bench.py run --suite capacity \
    --strategies fsdp --world-sizes 4 \
    --batch-sizes 96 112 120 \
    --warmup-steps 3 --measure-steps 5 --repeats 1 \
    --model-config configs/synbios_moe/model.yaml \
    --output artifacts/distributed_benchmark/synbios_moe_fsdp4/capacity_refine
  ```

- Result path: `artifacts/distributed_benchmark/synbios_moe_fsdp4/capacity_refine/`
- Runtime log: `artifacts/logs/synbios_moe_fsdp4_capacity_refine.log`
- Result: batch 96 succeeded at 74.56% peak allocated memory and 365,034 tokens/s; batch 112
  succeeded at 86.20% and 370,857 tokens/s; batch 120 succeeded at 92.02% but throughput fell to
  281,926 tokens/s. Batch 112 is the provisional safe/performance selection and must pass the
  longer formal-config stability test below before launch. The session was deliberately stopped
  before retaining any non-8-aligned candidate as a selection.

## 2026-07-21 04:00 — SynBioS single-corpus dataset preparation

- Status: completed (2026-07-21 04:00 Asia/Shanghai)
- Purpose: generate the exact 100,000-person `single` corpus and token shards required by the
  formal 4-GPU FSDP pretraining configuration.
- tmux session: `minitrain-synbios-prepare`
- Command:

  ```bash
  python scripts/synbios_moe.py prepare \
    --output artifacts/synbios_moe/single \
    --variant single --num-people 100000 --seed 1337
  ```

- Output: `artifacts/synbios_moe/single/` (mounted storage under `/data`)
- Runtime log: `artifacts/logs/synbios_moe_single_prepare.log`
- Result: generated 100,000 biographies and 7,405,102 training tokens; all 100,000 documents
  passed cleaning and the manifest records content SHA256 hashes. At local batch 112 and four
  ranks, each epoch has 14,463 complete blocks, consumes 14,336 blocks in 32 optimizer steps,
  and drops 127 shuffled blocks (65,024 block tokens, 0.878%) plus the 45-token incomplete tail.

## 2026-07-21 04:00 — SynBioS FSDP batch-112 extended stability benchmark

- Status: completed (2026-07-21 04:01 Asia/Shanghai)
- Purpose: validate the provisional 8-aligned safe batch for 55 consecutive full training steps
  (5 warmup + 50 measured), including backward and AdamW optimizer updates.
- tmux session: `minitrain-synbios-b112-stability`
- Command:

  ```bash
  python scripts/run_dist_bench.py run --suite capacity \
    --strategies fsdp --world-sizes 4 --batch-sizes 112 \
    --warmup-steps 5 --measure-steps 50 --repeats 1 \
    --model-config configs/synbios_moe/model.yaml \
    --output artifacts/distributed_benchmark/synbios_moe_fsdp4/b112_stability
  ```

- Result path: `artifacts/distributed_benchmark/synbios_moe_fsdp4/b112_stability/`
- Runtime log: `artifacts/logs/synbios_moe_fsdp4_b112_stability.log`
- Result: all 55 steps completed. Peak allocated memory was 20,875 MiB (86.20%), peak reserved
  memory was 22,912 MiB, throughput was 368,170 tokens/s, and final random-data loss was finite
  at 10.8419. Local batch 112 is persisted in `single_fsdp_4gpu.yaml`.

## 2026-07-21 04:02 — Formal-config FSDP checkpoint/resume validation

- Status: completed (2026-07-21 04:07 Asia/Shanghai)
- Purpose: run the exact formal model, actual randomized-document dataset, local batch 112, BF16
  Triton backend, and 4-GPU FSDP; save an isolated full model+Adam checkpoint at step 3, restore
  it, and advance to step 5 without touching the formal training checkpoint directory.
- tmux session: `minitrain-synbios-validation`
- Config: `configs/synbios_moe/runs/single_fsdp_4gpu_validation.yaml`
- Output: `artifacts/validation/synbios_moe/`
- Runtime log: `artifacts/logs/synbios_moe_fsdp4_validation.log`
- Command:

  ```bash
  torchrun --standalone --nproc_per_node 4 scripts/train.py --device cuda \
    --smoke-steps 3 --config configs/synbios_moe/runs/single_fsdp_4gpu_validation.yaml \
    --model-config configs/synbios_moe/model.yaml
  torchrun --standalone --nproc_per_node 4 scripts/train.py --device cuda \
    --smoke-steps 5 --resume latest \
    --config configs/synbios_moe/runs/single_fsdp_4gpu_validation.yaml \
    --model-config configs/synbios_moe/model.yaml
  ```
- First attempt result: step-3 DCP checkpoint committed, but the first post-resume optimizer step
  exposed PyTorch 2.5 FSDP/DCP dropping the AdamW `betas` parameter-group option
  (`KeyError: 'betas'`). The restore path now fills only missing non-parameter options from the
  freshly constructed matching optimizer group. Regression result: 9 checkpoint tests and ruff
  passed.
- Final result: restore succeeded and advanced from step 3 to step 5 on the exact model and real
  dataset. Learning rate continued without reset: `6.542e-05` at step 3, `8.723e-05` at step 4,
  and `1.090e-04` at step 5. Both full model+Adam checkpoints are committed in the isolated
  validation directory.

## 2026-07-21 04:10 — SynBioS exact-model FSDP weak-scaling verification

- Status: completed (2026-07-21 04:10 Asia/Shanghai)
- Purpose: measure 1-GPU versus 4-GPU FSDP throughput at the largest common conservative local
  batch, using the exact 293,494,272-parameter SynBioS model and full backward/optimizer steps.
- tmux session: `minitrain-synbios-fsdp-scaling`
- Command:

  ```bash
  python scripts/run_dist_bench.py run --suite weak \
    --strategies fsdp --world-sizes 1 4 --local-batch 64 \
    --warmup-steps 5 --measure-steps 20 --repeats 2 \
    --model-config configs/synbios_moe/model.yaml \
    --output artifacts/distributed_benchmark/synbios_moe_fsdp4/weak_b64
  ```

- Result path: `artifacts/distributed_benchmark/synbios_moe_fsdp4/weak_b64/`
- Runtime log: `artifacts/logs/synbios_moe_fsdp_scaling_b64.log`
- Result: two repeats completed without failure. Mean 1-GPU FSDP throughput was 93,302
  tokens/s; mean 4-GPU FSDP throughput was 344,254 tokens/s. Four-GPU weak-scaling efficiency
  was 92.08% and 92.40% (mean 92.24%), clearing the 80% gate. Four-GPU data stall was only
  0.09–0.12%; no NCCL/NUMA source change is needed.

## 2026-07-21 04:12 — Full pre-launch regression suite

- Status: completed (2026-07-21 04:12 Asia/Shanghai)
- Purpose: execute the full Python regression suite and repository lint after all benchmark,
  config, and FSDP checkpoint fixes; persist machine-readable JUnit and text logs.
- tmux session: `minitrain-prelaunch-tests`
- Command:

  ```bash
  pytest -q --junitxml=artifacts/validation/pytest_full.xml
  ruff check .
  ```

- JUnit: `artifacts/validation/pytest_full.xml`
- Runtime log: `artifacts/logs/prelaunch_full_tests.log`
- Result: 68 tests passed with zero failures/skips and five expected DCP single-process warnings;
  full-repository ruff passed. Dataset revalidation also matched every manifest SHA256 for the
  biography sources, profiles, token shard, and document indices. The formal checkpoint run
  directory was empty and all four GPUs had no compute processes before launch preparation.

## 2026-07-21 04:13 — Export pre-launch results into Git worktree

- Status: completed (2026-07-21 04:13 Asia/Shanghai)
- Purpose: copy all benchmark reports, logs, JUnit, TensorBoard/event logs, and checkpoint
  metadata into `results/` for GitHub. Multi-gigabyte DCP tensor shards and `model.pt` remain on
  mounted storage and are excluded from Git; their COMMITTED/runtime/RNG evidence is exported.
- tmux session: `minitrain-export-results`
- Command: `bash scripts/bash/export_test_results.sh`
- Destination: `results/`
- Result: 1.8 MiB of Git-trackable results exported; full mounted validation checkpoints remain
  intact under `artifacts/validation/`.

## 2026-07-21 04:15 — Formal SynBioS single-corpus 4-GPU FSDP pretraining

- Status: stopped safely at epoch 30 / step 960 (2026-07-21 04:48 Asia/Shanghai)
- Purpose: formal `single` corpus pretraining with the exact SynBioS MoE model, Triton operators,
  BF16, and FSDP full sharding on all four RTX 4090 GPUs.
- tmux session: `minitrain-synbios-single-fsdp4`
- Command:

  ```bash
  AUTO_RESUME=0 NPROC=4 bash scripts/bash/synbios_moe.sh single fsdp
  ```

- Config: `configs/synbios_moe/runs/single_fsdp_4gpu.yaml`
- Model config: `configs/synbios_moe/model.yaml`
- Runtime log: `artifacts/logs/synbios_moe_single_fsdp4_formal.log`
- Checkpoints: `artifacts/synbios_moe/checkpoints/synbios_moe_single_fsdp_4gpu/`
- Metrics: `artifacts/synbios_moe/runs/synbios_moe_single_fsdp_4gpu/`
- Fresh-start checks:
  - no compute processes on any GPU;
  - no pre-existing formal-run checkpoint directory entries;
  - 82 GiB free on `/data`;
  - data manifest SHA256: `144cf49ea607b4a502e5be277dbb63e0e9a08f296596e994cd19d3c6cfb11e25`;
  - token manifest SHA256: `2bdabe8d64941a5e4c51bde05d35618eaa65392b4215a7f382bbc3b7f69541f8`;
  - model config SHA256: `b2a02c5200d6c039d10cb3a802fd105430c3c103b02516e7a96016a98eb4ed01`;
  - run config SHA256: `584a369a317c8795e73f0c57707788a5e867483d1c003dbd27902aebec4b9404`.
- Training contract: seed 1337; local batch 112; global batch 448; 32 optimizer steps/epoch;
  540 epochs; 17,280 total optimizer steps; 3,963,617,280 scheduled training tokens. Each epoch
  drops 127 newly shuffled blocks (0.878%), so no fixed subset is permanently excluded.
- Runtime maintenance (2026-07-21 04:16): PyTorch 2.5 emitted its internal, non-actionable
  `ShardedTensor` to `DTensor` FutureWarning on every DCP operation. A message- and module-scoped
  warning filter was added without hiding other warnings. Training was interrupted only after
  epoch 2 had a committed checkpoint and is resuming model, AdamW, scheduler/LR, counters, and RNG
  from `epoch_000002_step_000000064`; an interrupted epoch-3 temporary directory is not eligible
  for resume and will be safely replaced by the next epoch-3 save.
- Full fresh restart (2026-07-21 04:20, supersedes the maintenance resume above): by user
  decision, the running job was stopped and all current formal training state was removed from
  the active paths. Checkpoints through the last committed epoch, metrics, temporary state, and
  the old console log were moved recoverably to
  `/data/mini-train-sys/artifacts/archive/formal_restart_20260721_0419/`. All GPUs then reported
  0 MiB, 0% utilization, and no compute processes; the active formal checkpoint path was empty.
  Dataset and test artifacts were retained and unchanged. The warning-filter/checkpoint tests
  passed 8/8 and ruff passed while GPUs were empty. The command below is being issued again with
  `AUTO_RESUME=0`, producing a new log and a genuine step-0 run.
- Industrial checkpoint/logging optimization (2026-07-21 04:31): the second fresh job was stopped
  at user request before changing policy. The formal preset now logs every 4 steps; writes atomic
  full FSDP DCP+Adam recovery checkpoints every 10 epochs; retains the newest two plus a 50-epoch
  safety anchor; exports the separate 1.3 GB probe `model.pt` only every 50 epochs and at the final
  save; and records checkpoint duration, bytes, and whether a model export occurred. This caps
  crash rollback at about four compute minutes while eliminating the dominant per-epoch I/O and
  full-state-gather overhead. Regression runs in tmux `minitrain-checkpoint-opt-tests`, with log
  `artifacts/logs/checkpoint_optimization_tests.log` and JUnit
  `artifacts/validation/pytest_checkpoint_optimization.xml`.
- Optimization validation result: 70 tests passed, zero failed, and full ruff passed. The resolved
  formal config is local batch 112, log interval 4, recovery checkpoint interval 10 epochs,
  probe-export interval 50 epochs, and safety interval 50 epochs. Config SHA256 is
  `38dc1c6741253fb5b93f7e33ce6b9faed94df141d07cf71300e1db7b0d0425dc`.
- Training-state reset: 15 GiB of active formal checkpoints, active metrics/log, and the earlier
  15 GiB formal archive were removed from their active paths via the filesystem trash mechanism
  (recoverable from the mounted volume trash). All four GPUs were verified at 0 MiB/0% with no
  compute jobs. The immutable checked dataset and Git-trackable test results were retained.
- Optimized fresh launch command remains:

  ```bash
  AUTO_RESUME=0 NPROC=4 bash scripts/bash/synbios_moe.sh single fsdp
  ```
- Live verification: terminal emitted steps 1, 4, 8, ..., 32 as configured. Epochs 1–9 had no
  checkpoint pause. At epoch 10, the first optimized recovery checkpoint atomically committed
  3,677,964,267 bytes in 26.929 seconds with `exported_model=false`; it contains four FSDP DCP
  shards, optimizer state, runtime/scheduler counters, and four RNG states, with no redundant
  `model.pt`. Training immediately continued into epoch 11. Steady training throughput was about
  348k–369k tokens/s and the tmux session remained healthy.
- Convergence stop: loss initially fell to about 1.10 around epochs 10–20, then rose persistently
  to 5.38 by epoch 30 while the unclipped gradient norm repeatedly reached hundreds to tens of
  thousands and every logged interval clipped gradients. The global-batch linear-scaled peak LR
  `0.004667` is therefore not accepted as a correct formal recipe even though throughput and
  memory were stable. Epoch-30 checkpoint `epoch_000030_step_000000960` committed atomically,
  then the tmux training job was stopped; all GPUs returned to 0 MiB. A lower-LR short convergence
  sweep is required before restarting the 540-epoch formal run.

## 2026-07-21 04:42 — Temporary GitHub server snapshot

- Status: completed; pushed to `origin/train`
- Purpose: preserve the machine adaptations, benchmark implementation/fixes, executed notebook,
  exact-model capacity/scaling results, correctness and recovery validation, formal-run metadata,
  industrial checkpoint policy, and persistent server operating rules while training continues.
- Export command: `bash scripts/bash/export_test_results.sh`
- Git-safe result root: `results/`
- Included: 209+ benchmark/validation/environment/log/event/manifest files, human-readable summary,
  content SHA256 manifest, JUnit, figures, failure/OOM evidence, formal checkpoint runtime/RNG and
  COMMITTED metadata, dataset manifests, and TensorBoard/JSONL events.
- Excluded: raw biographies/token shards, model weights, optimizer/DCP tensor shards, caches,
  credentials, and SSH keys.
- Validation: 70 tests passed, ruff passed, export script syntax and execution passed, no exported
  file exceeded 2 MiB, and the credential-pattern scan returned no match. Source changes pass
  whitespace checks; exported raw third-party/runtime logs deliberately preserve their original
  trailing whitespace as experiment evidence.
- Active services: formal training tmux `minitrain-synbios-single-fsdp4`; TensorBoard tmux
  `minitrain-tensorboard` on TCP 6006.
- Remote snapshot commit: `0967d7f` (`feat(server): persist distributed benchmark snapshot`).

## 2026-07-21 04:51 — Paper-fidelity LR and GPU telemetry correction

- Status: running
- Local/UTC start: 2026-07-21 04:51 Asia/Shanghai / 2026-07-20 20:51 UTC
- Purpose: replace the misleading post-step allocator ratio with interval-sampled NVML GPU
  compute/memory-controller utilization, validate 4-GPU FSDP, and restart the formal `single`
  corpus run from random initialization with the paper optimizer schedule.
- Git at launch: `858cf3f19e247bd8b4b9f4f2cd7c3075e89d214c`; dirty with the telemetry,
  configuration, test, and documentation changes described by this entry.
- Hardware/topology: one node, four RTX 4090 24 GB GPUs, FSDP full shard, BF16, Triton ops.
- Corrected training contract: local batch 112, global batch 448, AdamW peak LR `1e-3`,
  warmup 1,000 optimizer steps, cosine floor `1e-4`, weight decay `0.1`, epsilon `1e-6`,
  gradient clipping threshold 5.0, and no LR/warmup batch scaling. The larger-than-paper batch
  remains an explicit fidelity difference; the paper optimizer hyperparameters are preserved.
- Data contract before reset: generated-data manifest SHA256
  `144cf49ea607b4a502e5be277dbb63e0e9a08f296596e994cd19d3c6cfb11e25`, token-manifest
  SHA256 `2bdabe8d64941a5e4c51bde05d35618eaa65392b4215a7f382bbc3b7f69541f8`.
- Reset scope: after validation, permanently remove only the active failed formal run's
  checkpoints, TensorBoard/JSONL run directory, and console log. Preserve the checked source
  dataset and the Git-exported historical failure evidence required by `AGENTS.md`.
- Validation tmux/session, exact commands, results, destructive reset inventory, and formal
  launch details will be appended below before the run starts.
- Regression gate: initial run found only the obsolete linear-scaling assertion (71 passed,
  1 failed); after updating that contract test, 72 tests passed and full ruff passed in tmux
  `minitrain-fidelity-tests-rerun`. JUnit:
  `artifacts/validation/pytest_fidelity_gpu_telemetry.xml`; log:
  `artifacts/logs/fidelity_gpu_telemetry_tests_rerun.log`.
- NVML sampler check: a two-second CUDA matrix load produced 22 samples with compute utilization
  mean 81.73% and max 99%, while post-workload allocator memory returned to zero. This confirms
  compute utilization is no longer inferred from post-step allocated memory.
- 4-GPU preflight status: running in tmux `minitrain-fidelity-fsdp4-preflight`.
- Preflight command:

  ```bash
  torchrun --standalone --nproc_per_node 4 scripts/train.py --device cuda \
    --config configs/synbios_moe/runs/single_fsdp_4gpu_validation.yaml \
    --model-config configs/synbios_moe/model.yaml --smoke-steps 64
  ```

- Preflight log: `artifacts/logs/fidelity_fsdp4_preflight.log`; isolated result root:
  `artifacts/validation/synbios_moe/runs/synbios_moe_single_fsdp_4gpu_validation/`.
- First preflight attempt failed before CUDA allocation because validation disabled model export
  while inheriting `export_model_every_epochs: 50`. The validation override now explicitly sets
  that interval to null; retry tmux is `minitrain-fidelity-fsdp4-preflight-r2` with the same
  command and log `artifacts/logs/fidelity_fsdp4_preflight_r2.log`.
- Preflight result: completed 64/64 optimizer steps with no NaN/Inf. Loss fell from 10.94644 to
  3.68811; grad norm ended at 1.438 and the per-step clipping signal ended at zero after one
  transient maximum of 70.656. Steady throughput reached 353k-358k token/s. NVML compute
  utilization averaged 96.89% across all logged points (about 97%-99% after startup), and the
  corrected interval peak allocated-memory ratio was 86.2%. All four GPUs returned to 0 MiB.
  Event directory:
  `artifacts/validation/synbios_moe/runs/synbios_moe_single_fsdp_4gpu_validation/20260721-045343/`.
- Dataset integrity gate: all six files named by the generation/token manifests passed exact size
  and SHA256 verification; 100,000 biographies, 100,000 document-index entries, and 7,405,102
  training tokens remain unchanged.
- Fresh reset completed at 2026-07-21 04:55 Asia/Shanghai: the exact active failed-run targets
  (11,033,892,817 checkpoint bytes in 34 files, 4,009,995 run-log bytes in two files, and the
  40,097-byte console log) were moved to the mounted filesystem trash after direct deletion was
  blocked by the host safety policy. Their active paths are absent, so no checkpoint/log can be
  resumed or mixed into the new TensorBoard run; historical Git exports and the source dataset
  remain. `/data` has 42 GiB available and all four GPUs are empty.
- Formal fresh-launch status: starting from random initialization in tmux
  `minitrain-synbios-single-fsdp4`; no `--resume` argument and `AUTO_RESUME=0`.
- Formal command:

  ```bash
  AUTO_RESUME=0 NPROC=4 bash scripts/bash/synbios_moe.sh single fsdp
  ```

- Formal console log: `artifacts/logs/synbios_moe_single_fsdp4_formal.log`; checkpoints:
  `artifacts/synbios_moe/checkpoints/synbios_moe_single_fsdp_4gpu/`; TensorBoard/JSONL:
  `artifacts/synbios_moe/runs/synbios_moe_single_fsdp_4gpu/`. Run-config SHA256:
  `0f75102bde77e16b419b91e60f211f9b30c47c7cafca572e2bce5e3c5abfb0b2`.
- Result-export correction: the first post-reset export exposed that `rsync --delete` would mirror
  active artifact pruning into Git and remove the prior failed-run evidence. No commit occurred.
  The exporter is now append-only for benchmarks, logs, validation, formal runs, and checkpoint
  metadata; the tracked epoch-10/20/30 metadata and old `20260721-042956` events were restored
  before regenerating `results/MANIFEST.sha256`.
- Formal live verification: the new TensorBoard event file contains every corrected telemetry tag
  (NVML compute min/mean/max, memory-controller min/mean/max, and allocator
  current/reserved/interval-peak percentages). At epoch 7 the loss was about 1.11, grad norm about
  0.14, LR `2.0e-4` in the 1,000-step warmup, steady compute utilization about 98%, and interval
  peak allocated memory 86.2%. This is a fresh trajectory, not a resume of the rejected run.
- End/status: 2026-07-21 08:28 Asia/Shanghai / 2026-07-21 00:28 UTC; `completed`, exit code 0.
- Final result: completed all 540 epochs / 17,280 optimizer steps and 3,963,617,280 scheduled
  tokens. Logged loss fell from 10.94644 at step 1 to 0.193221 at step 17,280 (minimum logged
  0.192083); final grad norm was 0.02456, final LR was `1.00000008e-4`, average end-to-end
  throughput was 312,868 tok/s, mean logged NVML compute utilization was 97.02%, and interval peak
  allocated memory remained 86.2%. Only 0.197% of logged steps clipped gradients.
- Final retained checkpoint: `epoch_000540_step_000017280`, atomically `COMMITTED`, with four DCP
  model/Adam shards, runtime/scheduler state, four RNG states, and the final `model.pt` export.
  Retention also preserves epoch 500 as the safety anchor and epoch 530 as the previous recovery
  point; the active formal checkpoint root is 12 GiB.
- Final evidence: TensorBoard event file 27,376,320 bytes, JSONL event stream 31,135,172 bytes,
  and console log 783,380 bytes under the paths above. All Git-safe evidence is exported under
  `results/formal_runs/synbios_moe/` and `results/logs/`; tensor payloads remain on `/data`.

## 2026-07-21 13:33 — Completed formal-run GitHub snapshot

- Status: completed; payload pushed to `origin/train`.
- Local/UTC start: 2026-07-21 13:33 Asia/Shanghai / 2026-07-21 05:33 UTC.
- Purpose: export and push every Git-safe server result and process artifact after the successful
  540-epoch SynBioS run, especially complete TensorBoard/JSONL metrics, console/test/service logs,
  validation reports, dataset manifests, checkpoint recovery metadata, and content hashes.
- Source Git revision: `858cf3f19e247bd8b4b9f4f2cd7c3075e89d214c` on `train`, initially synchronized
  with `origin/train`; working tree contains the completed GPU telemetry/config/test changes and
  generated evidence described in the preceding run entry.
- Export command: `bash scripts/bash/export_test_results.sh`; append-only destination `results/`.
- Validation tmux: `minitrain-prepush-20260721`; log:
  `artifacts/logs/prepush_validation_20260721-1333.log`.
- Validation command:

  ```bash
  python -m pytest -q && python -m ruff check minitrain scripts tests
  ```

- Planned exclusions: raw datasets/token shards, caches, trash, virtualenv contents, `model.pt`,
  and multi-gigabyte DCP/optimizer shards. Their retained locations, sizes, hashes/manifests, and
  `COMMITTED`/runtime/RNG evidence are preserved where allowed by the repository policy.
- Validation result: 72 tests passed with zero failures in 22.18 seconds; Ruff passed. Exit status
  was zero and the complete output is retained in the validation log above.
- Export result: 254 content-hashed files totaling 64 MiB. This includes the successful formal
  run's 31,135,172-byte JSONL stream and 27,376,320-byte TensorBoard event, validation and data
  preparation TensorBoard events, the early single-GPU smoke events, console/service/test logs,
  JUnit, checkpoint runtime/RNG/COMMITTED metadata, manifests, benchmark raw data, CSV, and plots.
- Safety audit: every `results/MANIFEST.sha256` entry verified; no staged file reaches GitHub's
  100 MiB hard limit; credential scans found no private key, cloud key, GitHub/OpenAI token,
  bearer token, or assignment-style secret. `git diff --check` passed for source/documentation;
  raw generated logs/events retain their original formatting as provenance.
- Remote preflight: `git fetch origin` completed and `HEAD...origin/train` was `0 0` before the
  snapshot commit. TensorBoard remains available in tmux `minitrain-tensorboard` on TCP 6006.
- Push result: completed at 2026-07-21 13:37 Asia/Shanghai / 2026-07-21 05:37 UTC. Payload commit
  `aef675c` (`feat(server): persist completed SynBioS formal run`) was pushed successfully;
  local and `origin/train` were verified at `0 0` divergence immediately afterward.

## 2026-07-21 13:46 — Single-bio progressive cloze evaluation pilot

- Status: running.
- Local/UTC start: 2026-07-21 13:46 Asia/Shanghai / 2026-07-21 05:46 UTC.
- Purpose: measure the completed `single` model by removing the six actual fact spans from each
  original biography and greedily filling them in original textual order. Generated earlier facts
  replace earlier holes before later facts are predicted; only the source's non-fact text is kept.
- Git at launch: `a117d336c84ebd24b588c8a4f572b10896825ee1`; dirty only with the new cloze
  evaluator, CLI entry, and its passing unit test.
- Backbone checkpoint:
  `artifacts/synbios_moe/checkpoints/synbios_moe_single_fsdp_4gpu/epoch_000540_step_000017280/`
  (`COMMITTED`, final `model.pt`, 540 epochs / 17,280 optimizer steps).
- Data: the first 1,000 of 100,000 deterministic `single` biographies; six fields per biography,
  exact case-sensitive value match, greedy decoding until the original sentence period, maximum
  16 generated tokens per field. This pilot calibrates throughput before the full-data run.
- Hardware: one RTX 4090 (`CUDA_VISIBLE_DEVICES=0`), inference-only frozen backbone, batch 16.
- tmux: `minitrain-single-cloze-1k`; log:
  `artifacts/logs/single_cloze_pilot_1000_20260721-1346.log`.
- Command:

  ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/synbios_moe.py cloze-evaluate \
    --data artifacts/synbios_moe/single \
    --model-config configs/synbios_moe/model.yaml \
    --checkpoint artifacts/synbios_moe/checkpoints/synbios_moe_single_fsdp_4gpu/epoch_000540_step_000017280 \
    --device cuda --examples 1000 --batch-size 16 --max-new-tokens 16 \
    --sample-biographies 20 \
    --output artifacts/synbios_moe/results/single_cloze_eval/pilot_1000.json \
    --log-dir artifacts/synbios_moe/results/single_cloze_eval/operation_logs \
    --log-interval 5 --tensorboard
  ```
- First attempt status: stopped during the second batch after the first 16 biographies exposed an
  invalid string-level GPT-2 BPE boundary (`0%` was discarded, not a model result). Splitting at a
  fact's first character retokenized a trained token such as `" July"` into a standalone space;
  a generated EOS also made the reconstructed string non-encodable. No result JSON was published.
- Corrected protocol: remove the exact contiguous fact token spans from the original training BPE
  sequence, keep all original non-fact tokens unchanged, insert generated token IDs progressively,
  and score both strict exact match and normalized Levenshtein character similarity (mean and
  >=50%/80%/90% rates). Retry tmux/log: `minitrain-single-cloze-1k-r2` and
  `artifacts/logs/single_cloze_pilot_1000_20260721-1346-r2.log`.
- Retry r2 verified the corrected metric on its first 80 biographies: strict field accuracy was
  95.625% and six-field biography accuracy was 73.75%. It was intentionally stopped because the
  progress reporter received cumulative rather than per-batch item/token counts, inflating only
  its throughput counters (not accuracy). That bookkeeping was fixed and revalidated with 18/18
  SynBioS tests plus Ruff. Final retry tmux/log: `minitrain-single-cloze-1k-r3` and
  `artifacts/logs/single_cloze_pilot_1000_20260721-1346-r3.log`.
- Retry r3 completed 1,000 biographies in 80.40 seconds and demonstrated 100% exact recall for
  birth date, university, major, company, and company city. Its apparent 78.5% birth-city score
  was a scoring-boundary artifact: the `calls {value} a birthplace.` template caused correct
  generations such as `Oakland4, MA a birthplace` to be compared against only `Oakland4, MA`.
  The result is retained as `pilot_1000_period_delimiter_r3.json`, not used as the final metric.
- Final r4 stops generation at each hole's actual right-hand literal delimiter (usually `.`, or
  ` a birthplace.` for that template), while still withholding the delimiter from the causal
  prompt. This scores only text belonging inside the original hole. Final tmux/log:
  `minitrain-single-cloze-1k-r4` and
  `artifacts/logs/single_cloze_pilot_1000_20260721-1346-r4.log`.
- Pilot final result: completed at 2026-07-21 13:55 Asia/Shanghai in 79.13 seconds. All 6,000
  fields were exact, normalized character similarity was 100%, all >=50%/80%/90% fuzzy rates
  were 100%, all 1,000 biographies had 6/6 correct, and no generation hit the token limit.
  Final JSON: `artifacts/synbios_moe/results/single_cloze_eval/pilot_1000.json`; TensorBoard/JSONL:
  `artifacts/synbios_moe/results/single_cloze_eval/operation_logs/`.

## 2026-07-21 13:57 — Full 100k single-bio progressive cloze evaluation

- Status: completed successfully at 2026-07-21 14:08 Asia/Shanghai.
- Local/UTC start: 2026-07-21 13:57 Asia/Shanghai / 2026-07-21 05:57 UTC.
- Purpose/protocol: extend the validated pilot unchanged to all 100,000 original `single`
  biographies (600,000 progressively filled fact holes), using four disjoint contiguous shards.
- Code/checkpoint/data provenance is identical to the pilot entry above; only `--start-index`,
  `--examples 25000`, and `--batch-size 128` differ. Each process owns one RTX 4090 and one frozen
  model replica; no gradients, checkpoint writes, or inter-process aggregation affect inference.
- tmux sessions: `minitrain-cloze-full-gpu0` through `minitrain-cloze-full-gpu3`.
- Outputs/logs:
  `artifacts/synbios_moe/results/single_cloze_eval/full_100k/shard_{0,1,2,3}.json` and
  `artifacts/logs/single_cloze_full_gpu{0,1,2,3}_20260721-1357.log`.
- Per-shard command template:

  ```bash
  CUDA_VISIBLE_DEVICES=<gpu> python scripts/synbios_moe.py cloze-evaluate \
    --data artifacts/synbios_moe/single \
    --model-config configs/synbios_moe/model.yaml \
    --checkpoint artifacts/synbios_moe/checkpoints/synbios_moe_single_fsdp_4gpu/epoch_000540_step_000017280 \
    --device cuda --start-index <0|25000|50000|75000> --examples 25000 \
    --batch-size 128 --max-new-tokens 16 --sample-biographies 5 \
    --output artifacts/synbios_moe/results/single_cloze_eval/full_100k/shard_<gpu>.json \
    --log-dir artifacts/synbios_moe/results/single_cloze_eval/full_100k/operation_logs \
    --log-interval 20 --tensorboard
  ```
- First full-run attempt: all four processes completed with 100% exact accuracy, but the aggregate
  range validator correctly rejected the outputs because CLI wiring had passed `--start-index` to
  the legacy evaluator rather than the new cloze evaluator. All four therefore covered 0–24,999.
  GPU0's shard remains valid; GPU1–3 duplicate JSON/logs were renamed with
  `_duplicate_range0_r1` and retained as non-independent evidence, never counted toward 100k.
- Range fix: moved `start_index=args.start_index` to `command_cloze_evaluate`, reran 18/18 SynBioS
  tests and Ruff, and confirmed the CLI exposes the option. GPU1–3 are relaunched as tmux sessions
  `minitrain-cloze-full-gpu{1,2,3}-r2` with the same commands/ranges and logs suffixed `-r2`.
- Final validated ranges were exactly 0–24,999, 25,000–49,999, 50,000–74,999, and
  75,000–99,999. The range-aware aggregator rejected overlaps/gaps before publishing the summary.
- Final result: all 600,000 fields were strict case-sensitive exact matches, every field type was
  100%, all 100,000 biographies restored 6/6 fields, no generation hit the token limit, mean
  normalized Levenshtein character similarity was 100%, and fuzzy rates at 50%/80%/90% were all
  100%. Parallel wall time was 419.48 seconds (238.39 biographies/s across four GPUs).
- Machine-readable summary:
  `artifacts/synbios_moe/results/single_cloze_eval/full_100k/summary.json`; human conclusion with
  two worked progressive-fill examples and fuzzy-metric caveats:
  `reports/synbios_single_cloze_100k.md`.
- Interpretation: the final checkpoint learned the intended training corpus extremely well and
  can exactly recall every recorded fact under its original template/context. This is strong
  evidence that optimization was effective, but because the evaluated people and templates are
  the training set it is not evidence of unseen-person or unseen-template generalization.

## 2026-07-21 14:09 — Progressive-cloze validation and result export

- Status: completed; ready for the repository snapshot.
- Full regression: 74 tests passed with zero failures in 12.20 seconds; the five emitted warnings
  are the expected single-process `torch.distributed.checkpoint` fallbacks. Ruff passed.
- tmux/log: `minitrain-cloze-verify-20260721` and
  `artifacts/logs/cloze_full_validation_20260721-1409.log`; exit status was zero.
- Export command: `bash scripts/bash/export_test_results.sh`. The exporter now includes
  `artifacts/synbios_moe/results/`, so aggregate/shard JSON, rejected-attempt evidence, JSONL
  progress, and TensorBoard event files are retained under
  `results/formal_runs/synbios_moe/results/` rather than existing only on the mounted volume.
- Exported cloze evidence: 32 files totaling 648 KiB. `results/MANIFEST.sha256` verifies every
  exported file, source/documentation diff checks pass, and no exported file exceeds 90 MiB.
- Remote preflight at 14:19 Asia/Shanghai: `git fetch origin` succeeded and
  `HEAD...origin/train` was `0 0` before creating the new snapshot.
- Push result: commit `596bffb` (`feat(eval): add progressive biography cloze recall`) was pushed
  successfully to `origin/train`; local and remote divergence was verified as `0 0` afterward.

## 2026-07-21 14:25 — Replace the P/Q pretraining gate with original-bio cloze

- Status: completed and validated.
- Clarified scope: P/Q probe training, held-out validation, task scheduling, and result comparison
  remain unchanged. Only their pretraining-readiness gate changes.
- Requested gate: remove the six exact fact spans from original biographies, restore them
  progressively in source order with the frozen checkpoint, and gate on strict generated field
  accuracy instead of teacher-forced attribute-token next-token accuracy.
- Implementation: the P/Q pipeline now evaluates up to 10,000 original biographies with the same
  token-aligned progressive cloze evaluator and gates on strict `micro_field_accuracy >= 0.90`.
  Fuzzy similarity is retained in the gate JSON but cannot cause a pass.
- Cache safety: only results carrying the progressive-cloze protocol, matching common pipeline
  identity, and a numeric strict field metric can be reused. Pipeline protocol version increased
  from 2 to 3 so pre-change stages cannot silently satisfy the new gate.
- Default gate configuration: first 10,000 original biographies, batch 128, greedy generation up
  to 16 tokens per field, and strict field threshold 0.90. The completed full-corpus evaluation
  already establishes that this checkpoint scores strict 100% on that subset; fuzzy thresholds
  are never consulted by the gate decision.
- Validation: 75 repository tests passed in 11.87 seconds with only the five expected
  single-process DCP warnings; Ruff passed across `minitrain`, `experiments`, `scripts`, and
  `tests`. tmux/log: `minitrain-cloze-gate-verify-20260721` and
  `artifacts/logs/cloze_probe_gate_validation_20260721-1433.log`.
- Push result: commit `bb93aaa` (`feat(probe): gate pipeline on strict biography cloze`) was
  pushed to `origin/train`; local/remote divergence was verified as `0 0` afterward.
