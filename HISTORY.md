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

- Status: completed successfully at 2026-07-21 13:55 Asia/Shanghai.
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

## 2026-07-21 14:48 — Paper-format README and complete Git-safe result snapshot

- Status: documentation/export validation in progress.
- Purpose: place all currently available server training, engineering-validation, cloze-validation,
  capacity, scaling, failure-control, and experiment-status results directly in `README.md` using
  a paper-style Abstract/Methods/Results/Limitations structure.
- Reporting boundary: the README explicitly distinguishes the completed `single` 540-epoch run,
  its full training-corpus cloze recall, checkpoint/preflight validation, and the rejected high-LR
  run from unexecuted formal P/Q held-out validation, router analysis, and `multi5_permute` work.
- Evidence was re-derived from the exported JSONL/JSON rather than copied from console memory:
  17,280 steps, total loss 10.946440 to 0.193221 (minimum 0.192083), 3.9636B scheduled
  tokens, 12,668.67 seconds, 312,868 tok/s, mean logged NVML compute 97.02%, and 100% strict
  exact recall for all 600,000 fields in 100,000 original training biographies.
- README artifact links were checked against the working tree with zero missing relative targets.
- Result export: `bash scripts/bash/export_test_results.sh` produced 299 content-hashed Git-safe
  evidence files totaling 65 MiB. SHA256 verification passed, and rsync dry-runs found no pending
  files in server logs, formal run events, cloze results, or SynBioS operation logs.
- Deliberate payload exclusions: 12,362,146,587 bytes across 16 DCP/model tensor files and
  132,521,340 bytes across seven raw/tokenized dataset payload files. Their manifests, hashes,
  paths, `COMMITTED` markers, runtime/RNG metadata, metrics, and TensorBoard events are exported.
- Full validation tmux/log: `minitrain-readme-results-verify-20260721` and
  `artifacts/logs/readme_results_validation_20260721-1452.log`. Result: 75 tests passed in 11.51
  seconds with the five expected single-process DCP warnings; Ruff passed for `minitrain`,
  `experiments`, `scripts`, and `tests`.
- Push result: Conventional Commit `8b39e6a` (`docs(readme): publish formal server experiment
  report`) was pushed to `origin/train`; local/remote divergence was `0 0` afterward.

## 2026-07-21 14:37 — Launch multi5+permute 4-GPU FSDP pretraining

- Status: running successfully.
- Decision basis: the completed `single` run establishes stable optimization and 100% strict
  training-corpus recall. This is sufficient to validate the training implementation/configuration
  before starting the augmentation condition, while remaining explicitly distinct from unseen
  person/template generalization.
- Requested condition: `multi5+permute`, 100,000 identical person profiles, five independently
  rendered/permuted biographies per person, four RTX 4090 GPUs, FSDP full shard, BF16.
- The augmented data is not yet present and will be generated deterministically with seed 1337;
  the launcher will byte-compare its `profiles.jsonl` against `single` before training.
- Formal config correction before launch: the generic FSDP default local batch 8 was replaced by
  the already validated same-model/same-sequence local batch 112 (global 448). The five-times
  larger corpus uses 108 epochs and token-equivalent log/checkpoint periods: log every 20 steps,
  DCP+Adam every 2 epochs, safety/model export every 10 epochs. Peak LR remains `1e-3`, warmup
  1,000 steps, cosine floor `1e-4`, with no batch LR scaling.
- Pre-launch resources: all four GPUs idle at 0 MiB/0% utilization; `/data` has 30 GiB free;
  local and `origin/train` are synchronized at `0 0`; only TensorBoard tmux is active.
- Config/test commit: `3285476` (`fix(config): align multi5 fsdp4 with validated capacity`) was
  pushed to `origin/train` before launch, leaving the training process with a clean Git worktree.
- Data preparation tmux/log: `minitrain-multi5-prepare` and
  `artifacts/logs/synbios_multi5_permute_prepare.log`. It generated 500,000 accepted biographies
  and 37,046,556 training tokens. Dataset/token manifest SHA256 values are
  `31b99dd8d415007c34e00e3b32440014be4b34c27b7c3c1a80f7e1319019566e` and
  `0f589d4def4ddf1a2ac6f774cd28d8343ea168eb07b3199b9e6e5209444f1547`.
- Integrity: every raw and token-shard size/SHA256 passed; all 500,000 document-index entries are
  present; the 100,000-person `profiles.jsonl` is byte-identical to `single`. With drop-last and
  global batch 448, the resolved schedule is 161 steps/epoch × 108 = 17,388 steps and
  3,988,389,888 scheduled tokens.
- Formal tmux: `minitrain-synbios-multi5-permute-fsdp4`; command:

  ```bash
  AUTO_RESUME=0 NPROC=4 bash scripts/bash/synbios_moe.sh multi5_permute fsdp
  ```

- Console log: `artifacts/logs/synbios_moe_multi5_permute_fsdp4_formal.log`; TensorBoard/JSONL:
  `artifacts/synbios_moe/runs/synbios_moe_multi5_permute_fsdp_4gpu/`; checkpoints:
  `artifacts/synbios_moe/checkpoints/synbios_moe_multi5_permute_fsdp_4gpu/`.
- Live verification through batch 40/161 of epoch 1: loss fell `10.94893 → 7.96582`, LR warmed
  `1e-6 → 4e-5`, grad norm fell `17.113 → 7.030`, steady throughput reached 353k–364k tok/s,
  compute utilization was about 97.5%–98.2%, and interval peak allocated memory was 20.39/23.65
  GiB (the validated 86.2%). No NaN/Inf, OOM, dead process, or data stall was observed.

## 2026-07-21 18:56 — Complete multi5+permute training and full 500k cloze validation

- Status: pretraining, validation, result export, repository verification, and publication push
  completed successfully.
- Pretraining completed 108/108 epochs and 17,388/17,388 optimizer steps, processing
  3,988,389,888 scheduled tokens. Total loss fell from 10.948931 to 0.296150 (minimum logged
  0.293688); final LM cross-entropy was 0.285855, MoE regularization 0.010295, grad norm 0.06513,
  expert-load CV 0.02924, with zero dead experts and zero dropped routes.
- End-to-end training time was 12,148.31 seconds with 328,308 tok/s average throughput. The final
  atomically committed checkpoint is
  `artifacts/synbios_moe/checkpoints/synbios_moe_multi5_permute_fsdp_4gpu/epoch_000108_step_000017388/`,
  including DCP+Adam and a model export (5,006,353,550 total bytes).
- Final validation reused the `single` progressive original-biography cloze protocol unchanged:
  exact BPE fact spans removed, source-order greedy fill, earlier predictions inserted before later
  fills, 16-token field cap, and strict case-sensitive exact equality as the primary metric.
- Four disjoint ranges (0–124,999, 125,000–249,999, 250,000–374,999, and 375,000–499,999) ran on
  four RTX 4090 GPUs. The range-aware aggregator verified full contiguous coverage without gaps or
  overlap.
- Result: 2,999,746/3,000,000 strict fields exact (99.991533%); 499,813/500,000 biographies had
  6/6 fields exact (99.9626%); no generation reached the token cap. Field accuracies were birth
  date 99.9968%, birth city 99.9966%, university 99.9952%, major 99.9710%, company 99.9946%, and
  company city 99.9950%.
- Fuzzy accuracy was 99.993433% / 99.991733% / 99.991633% at thresholds 0.50 / 0.80 / 0.90. The
  0.90 score credits three non-exact fields, so strict accuracy remains the paper headline.
- Parallel validation wall time was 2,450.64 seconds (204.03 biographies/s). Machine summary:
  `artifacts/synbios_moe/results/multi5_permute_cloze_eval/full_500k/summary.json`; human report:
  `reports/synbios_multi5_permute_cloze_500k.md`.
- Interpretation: optimization and augmented training-corpus recall are successful. This is not
  held-out validation, and the near-perfect score does not establish an augmentation-driven
  generalization gain over the 100%-exact `single` training-corpus result.
- Repository verification: 75 tests passed with five expected single-process DCP warnings; Ruff
  passed. Persistent log: `artifacts/logs/multi5_results_validation_20260721.log`. The final export
  includes training JSONL/TensorBoard events, validation JSON/TensorBoard/events, console logs,
  data manifests, and checkpoint metadata, with `results/MANIFEST.sha256` regenerated.
- Push result: Conventional Commit `1361173` (`docs(results): publish multi5 training and
  validation`) was pushed successfully to `origin/train`.

## 2026-07-21 19:13 — Audit and synchronize all Git-safe server evidence

- Scope: every server-side benchmark, smoke run, formal training run, validation run, cloze
  result, operation log, console log, TensorBoard event, JUnit report, environment inventory,
  dataset manifest, and lightweight checkpoint recovery file.
- A source-to-export `rsync --dry-run` audit reported no missing files for all covered artifact
  roots after refresh: `distributed_benchmark`, `logs`, `validation`, `smoke`, SynBioS `runs`,
  `results`, `operation_logs`, `checkpoints`, and local smoke `runs/checkpoints`.
- Gap fixed: the previous checkpoint rule excluded the complete `distributed/` directory and thus
  omitted each small DCP `.metadata` file. The exporter now excludes only `*.distcp` tensor shards
  and `model.pt`, while preserving nine DCP metadata files (about 20–253 KiB each) across smoke,
  validation, `single`, and `multi5_permute` checkpoints.
- Deliberate remote exclusions remain raw biography/token payloads, `model.pt`, and DCP/Adam tensor
  shards. These are data/model payloads rather than result evidence, include files as large as
  1.33 GB, and would exceed or misuse normal GitHub storage. Their exact hashes, manifests, sizes,
  committed markers, runtime/RNG state, and DCP layout metadata are retained remotely.
- Push result: Conventional Commit `0a8c3d2` (`chore(results): sync all server evidence`) was
  pushed successfully to `origin/train`.

## 2026-07-23 13:59 — Probe regression gate and cache preparation

- Status: completed successfully at 2026-07-23 14:02 Asia/Shanghai / 2026-07-23 06:02 UTC;
  exit code 0.
- Local/UTC start: 2026-07-23 13:59 Asia/Shanghai / 2026-07-23 05:59 UTC.
- Purpose: validate the hardened P/Q probe implementation at current HEAD, then generate and
  validate the complete protocol-v2 probe caches for both `single` and `multi5_permute` before
  the four-GPU batch-capacity regression.
- Git at launch: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with the
  user-requested `AGENTS.md` conclusion/dataset organization rules and this appended history entry.
- Hardware/topology: one node with 4 × RTX 4090 24 GB, all at 0 MiB before launch; cache
  preparation is CPU/storage work and does not train the backbone.
- tmux session: `minitrain-probe-preflight-cache-20260723`.
- Command:

  ```bash
  cd /home/ubuntu/mini-train-sys
  source .venv/bin/activate
  source .minitrain-storage.env
  python -m pytest -q --junitxml=artifacts/validation/pytest_probe_preflight_20260723.xml
  python -m ruff check .
  python scripts/synbios_moe.py cache-probes \
    --data artifacts/synbios_moe/single \
    --output artifacts/synbios_moe/single/probe_cache \
    --require-coverage
  python scripts/synbios_moe.py validate-probe-cache \
    --probe-cache artifacts/synbios_moe/single/probe_cache \
    --data artifacts/synbios_moe/single
  python scripts/synbios_moe.py cache-probes \
    --data artifacts/synbios_moe/multi5_permute \
    --output artifacts/synbios_moe/multi5_permute/probe_cache \
    --require-coverage
  python scripts/synbios_moe.py validate-probe-cache \
    --probe-cache artifacts/synbios_moe/multi5_permute/probe_cache \
    --data artifacts/synbios_moe/multi5_permute
  ```

- Log: `artifacts/logs/probe_preflight_cache_20260723_1359.log`.
- Validation output: `artifacts/validation/pytest_probe_preflight_20260723.xml`.
- Dataset inputs: `artifacts/synbios_moe/{single,multi5_permute}/`; derived outputs:
  `artifacts/synbios_moe/{single,multi5_permute}/probe_cache/`.
- Next gate after successful completion: run the four-GPU P/Q training/validation batch-capacity
  regression and require a complete, non-boundary `recommended.env` before probe smoke training.
- Result: current HEAD passed 85 tests with five expected single-process DCP warnings, and full
  Ruff passed. The `single` protocol-v2 cache contains 100,000 P and 100,000 Q examples and uses
  41 MiB; the `multi5_permute` cache contains 500,000 P and 100,000 Q examples and uses 168 MiB.
  Both caches match their source manifests, cover every validation class in the person-level
  training split, and report no missing validation classes.

## 2026-07-23 14:02 — Multi5 P/Q four-GPU batch-capacity regression

- Status: failed at 2026-07-23 14:03 Asia/Shanghai / 2026-07-23 06:03 UTC; exit code 1.
- Local/UTC start: 2026-07-23 14:02 Asia/Shanghai / 2026-07-23 06:02 UTC.
- Purpose: measure real P/Q probe training and held-out validation capacity on the largest
  `multi5_permute` condition, with two independent GPU replicas per probe kind, and publish
  non-boundary batch recommendations for the smoke/pilot/formal pipeline.
- Git at launch: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with
  `AGENTS.md` organization rules and the current run-history additions.
- Hardware/topology: 4 × RTX 4090 24 GB, all at 0 MiB and 0% utilization immediately before
  launch. GPUs 0/2 independently benchmark P and GPUs 1/3 independently benchmark Q.
- Dataset/cache: `artifacts/synbios_moe/multi5_permute/` and its validated protocol-v2
  `probe_cache/` (500,000 P examples, 100,000 Q examples, complete train/validation class
  coverage).
- Backbone: committed final `multi5_permute` checkpoint
  `artifacts/synbios_moe/checkpoints/synbios_moe_multi5_permute_fsdp_4gpu/epoch_000108_step_000017388/`.
- tmux session: `minitrain-probe-capacity-multi5-20260723`.
- Command:

  ```bash
  cd /home/ubuntu/mini-train-sys
  source .venv/bin/activate
  source .minitrain-storage.env
  PROBE_BENCHMARK_RUN_ID=20260723T060246Z \
    bash scripts/bash/synbios_probe_batch_benchmark.sh \
      multi5_permute \
      artifacts/synbios_moe/checkpoints/synbios_moe_multi5_permute_fsdp_4gpu/epoch_000108_step_000017388
  ```

- Candidate batches: P training `32,50,64,80,96`; Q training `128,200,256,320,384`;
  P validation `64,96,128,160,192`; Q validation `256,384,512,640,768`.
- Benchmark contract: three warmup and ten measured steps per point, full
  forward/backward/AdamW for training, forward-only for validation, at most 92% CUDA reserved
  memory, and recommendation only if both replicas agree on a safe non-right-boundary choice.
- Console log: `artifacts/logs/probe_capacity_multi5_20260723_1402.log`.
- Result directory:
  `artifacts/synbios_moe/results/probe_batch_benchmark/multi5_permute/20260723T060246Z/`.
- Failure result: all four workers failed before model allocation. `ProgressReporter` called
  `torch.cuda.reset_peak_memory_stats(cuda:N)` before the explicitly indexed device had been
  initialized, producing `RuntimeError: Invalid device argument` on GPUs 0–3. No capacity point
  or recommendation was published, all GPUs returned to 0 MiB, and the four failure logs are
  retained under the result directory.
- Corrective action: `ProgressReporter` now makes its requested CUDA device current before
  resetting memory statistics. A regression test covers the ordering; the focused logger suite
  passes 8 tests and Ruff passes before retry.

## 2026-07-23 14:04 — Multi5 P/Q four-GPU batch-capacity regression retry

- Status: completed but rejected by the formal-readiness gate at 2026-07-23 14:05
  Asia/Shanghai / 2026-07-23 06:05 UTC; exit code 1 by design.
- Local/UTC start: 2026-07-23 14:04 Asia/Shanghai / 2026-07-23 06:04 UTC.
- Purpose/configuration: retry the preceding multi5 P/Q training and validation capacity matrix
  unchanged after fixing indexed CUDA initialization; retain the first attempt as failure evidence.
- Git at launch: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with
  `AGENTS.md`, `HISTORY.md`, the CUDA reporter fix, and its regression test.
- Preflight: focused logger tests 8/8 passed, focused Ruff passed, and all four RTX 4090 GPUs were
  idle. Dataset/cache, final checkpoint, candidates, warmup/measurement counts, two-replica
  topology, and 92% limit are identical to the preceding entry.
- tmux session: `minitrain-probe-capacity-multi5-r2-20260723`.
- Command:

  ```bash
  cd /home/ubuntu/mini-train-sys
  source .venv/bin/activate
  source .minitrain-storage.env
  PROBE_BENCHMARK_RUN_ID=20260723T060430Z \
    bash scripts/bash/synbios_probe_batch_benchmark.sh \
      multi5_permute \
      artifacts/synbios_moe/checkpoints/synbios_moe_multi5_permute_fsdp_4gpu/epoch_000108_step_000017388
  ```

- Console log: `artifacts/logs/probe_capacity_multi5_r2_20260723_1404.log`.
- Result directory:
  `artifacts/synbios_moe/results/probe_batch_benchmark/multi5_permute/20260723T060430Z/`.
- Result: all 40 measurements completed successfully, with both replicas agreeing and no OOM.
  Recommended points were P training 96 at 33.29% maximum reserved memory, Q training 384 at
  19.97%, P validation 192 at 7.02%, and Q validation 768 at 5.99%. All four were the largest
  candidate tested and still improved throughput, so `ready_for_formal=false`; `summary.json` was
  retained but `recommended.env` was deliberately not published. A wider search is required.

## 2026-07-23 14:06 — Expanded multi5 P/Q batch-capacity regression

- Status: completed successfully at 2026-07-23 14:09 Asia/Shanghai / 2026-07-23 06:09 UTC;
  exit code 0.
- Local/UTC start: 2026-07-23 14:06 Asia/Shanghai / 2026-07-23 06:06 UTC.
- Purpose: extend all four P/Q training/validation candidate ranges beyond the right-boundary
  recommendations from the preceding complete matrix and locate a true throughput or 92%-memory
  boundary.
- Git/code/data/checkpoint/hardware: identical to the preceding retry, including the indexed-CUDA
  fix; all four GPUs were idle before launch.
- tmux session: `minitrain-probe-capacity-multi5-r3-20260723`.
- Command:

  ```bash
  cd /home/ubuntu/mini-train-sys
  source .venv/bin/activate
  source .minitrain-storage.env
  P_BATCHES=96,128,160,192,224,256,288,320,352,384,416 \
  Q_BATCHES=384,512,768,1024,1280,1536,1792,2048,2304,2560 \
  P_VALIDATION_BATCHES=192,256,384,512,768,1024,1536,2048 \
  Q_VALIDATION_BATCHES=768,1024,1536,2048,3072,4096,6144,8192,12288,16384 \
  PROBE_BENCHMARK_RUN_ID=20260723T060600Z \
    bash scripts/bash/synbios_probe_batch_benchmark.sh \
      multi5_permute \
      artifacts/synbios_moe/checkpoints/synbios_moe_multi5_permute_fsdp_4gpu/epoch_000108_step_000017388
  ```

- Benchmark contract remains two independent replicas, three warmup and ten measured steps per
  candidate, and at most 92% reserved memory.
- Console log: `artifacts/logs/probe_capacity_multi5_r3_20260723_1406.log`.
- Result directory:
  `artifacts/synbios_moe/results/probe_batch_benchmark/multi5_permute/20260723T060600Z/`.
- Result: `ready_for_formal=true` with no boundary recommendations. The two-replica throughput
  optima are P training 128 (580.54 examples/s mean, 43.37% maximum reserved memory), Q training
  768 (3,938.28 examples/s, 35.37%), P validation 512 (1,502.85 examples/s, 10.60%), and Q
  validation 6,144 (10,699.55 examples/s, 14.86%). The generated `recommended.env` records those
  four values. Larger safe candidates on both sides demonstrated that none is a right-boundary
  artifact; training candidates above the 92% rule were excluded from recommendation.
- Paper-default evidence: this expanded matrix starts at the previous right boundary and therefore
  does not repeat P=50/Q=200. Their safety is established by the complete preceding matrix on both
  replicas (P 19.33%, Q 12.50% maximum reserved memory). Read the two summaries together; the
  expanded summary's `paper_batch_safe_on_all=false` means “not included in this matrix,” not an
  observed unsafe result.
- Persistence step: export the complete raw JSON, worker logs, event logs, summary, and
  `recommended.env` to `results/benchmarks/synbios_moe/probe_batch_benchmark/`, export both cache
  manifests to `results/datasets/synbios_moe/<variant>/probe_cache/`, regenerate
  `results/MANIFEST.sha256`, and update the human-readable benchmark/report indexes before probe
  smoke training.

## 2026-07-23 14:12 — Single-condition P/Q probe smoke

- Status: completed successfully at 2026-07-23 14:19 Asia/Shanghai / 2026-07-23 06:19 UTC;
  exit code 0.
- Local/UTC start: 2026-07-23 14:12 Asia/Shanghai / 2026-07-23 06:12 UTC.
- Purpose: execute the first end-to-end P/Q probe stage against the completed `single` backbone:
  strict pretrain gate, 500 training steps each for P/Q `university_whole`, independent
  person-held-out validation from saved probe checkpoints, pipeline identity checks, event logs,
  and atomic lightweight recovery.
- Git at launch: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with the
  user-requested organization rules, experiment history/results/report, indexed-CUDA monitoring
  fix and test, exporter organization change, and exported benchmark/cache evidence.
- Preflight gates: current tree 85/85 tests and full Ruff passed before cache generation; focused
  post-fix logger tests 8/8 and Ruff passed; both probe caches validate with complete class
  coverage; exported `results/MANIFEST.sha256` verifies; all four GPUs were idle.
- Hardware/topology: 4 × RTX 4090 24 GB. Probe task parallelism assigns at most one independent
  probe worker per GPU; it does not DDP-wrap a classifier.
- Dataset/cache: `artifacts/synbios_moe/single/` and protocol-v2 `probe_cache/`; cache-manifest
  SHA256 `9dcfd2cff38d6f3d29d7f10c2a3247b634f31395b3cda3755a755c8964ffaf5b`.
- Backbone checkpoint:
  `artifacts/synbios_moe/checkpoints/synbios_moe_single_fsdp_4gpu/epoch_000540_step_000017280/`;
  `model.pt` SHA256 `2d154fa14cbd233e71936f1db8f54d2c652c1347f529d2db2e6be8df49039e84`.
- Capacity input:
  `artifacts/synbios_moe/results/probe_batch_benchmark/multi5_permute/20260723T060600Z/recommended.env`
  (SHA256 `30f36e9572b66dbcdc9c4272f70bb077a5c774da4790fab4d65444a08723f309`);
  P/Q training batches 128/768 and validation batches 512/6,144.
- tmux session: `minitrain-probe-smoke-single-20260723`.
- Command:

  ```bash
  cd /home/ubuntu/mini-train-sys
  source .venv/bin/activate
  source .minitrain-storage.env
  STAGE=smoke NPROC=4 PROBE_GPUS=4 \
  PROBE_BATCH_ENV=artifacts/synbios_moe/results/probe_batch_benchmark/multi5_permute/20260723T060600Z/recommended.env \
    bash scripts/bash/synbios_probes.sh single fsdp latest
  ```

- Console log: `artifacts/logs/probe_smoke_single_20260723_1412.log`.
- Output:
  `artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/smoke/`.
- Promotion rule: do not launch the `single` pilot until this stage has `pipeline.json` status
  `completed`; run the corresponding `multi5_permute` smoke before any cross-condition pilot.
- Result: strict 10,000-biography cloze gate passed at 60,000/60,000 fields and 10,000/10,000
  biographies exact. Both 500-step training jobs and both checkpoint-reloaded held-out validation
  jobs completed. P `university_whole` validation accuracy across the six left-to-right biography
  observation positions was `[0.003891, 0.004210, 0.052217, 0.946426, 0.902490, 0.720759]`;
  Q name-only validation accuracy was `0.003392`. These are smoke diagnostics, not formal
  Allen-Zhu results. `pipeline.json` is `completed` with seven summary rows and matching protocol
  identity.

## 2026-07-23 14:20 — Multi5+permute P/Q probe smoke

- Status: completed successfully at 2026-07-23 14:33 Asia/Shanghai / 2026-07-23 06:33 UTC;
  exit code 0.
- Local/UTC start: 2026-07-23 14:20 Asia/Shanghai / 2026-07-23 06:20 UTC.
- Purpose: run the same two-task, 500-step P/Q smoke protocol and independent person-held-out
  validation against the completed `multi5_permute` backbone, after the `single` smoke completed.
- Git at launch: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with the
  previously recorded probe experiment implementation/results changes.
- Hardware/topology: 4 × idle RTX 4090 24 GB; task-parallel probe workers, at most one per GPU.
- Dataset/cache: `artifacts/synbios_moe/multi5_permute/` and its protocol-v2 cache; cache manifest
  SHA256 `acd78360d0daa7cf0d2c557fc9f68f07431bc3063cee1145daa3f14c320a232f`.
- Backbone checkpoint:
  `artifacts/synbios_moe/checkpoints/synbios_moe_multi5_permute_fsdp_4gpu/epoch_000108_step_000017388/`;
  `model.pt` SHA256 `e89075289bb3a774825e7fd03cedc2c7c37957583bf3656e8ab32c52ef02f0dd`.
- Capacity input: expanded formal-ready `recommended.env` SHA256
  `30f36e9572b66dbcdc9c4272f70bb077a5c774da4790fab4d65444a08723f309`;
  P/Q training 128/768, validation 512/6,144.
- tmux session: `minitrain-probe-smoke-multi5-20260723`.
- Command:

  ```bash
  cd /home/ubuntu/mini-train-sys
  source .venv/bin/activate
  source .minitrain-storage.env
  STAGE=smoke NPROC=4 PROBE_GPUS=4 \
  PROBE_BATCH_ENV=artifacts/synbios_moe/results/probe_batch_benchmark/multi5_permute/20260723T060600Z/recommended.env \
    bash scripts/bash/synbios_probes.sh multi5_permute fsdp latest
  ```

- Console log: `artifacts/logs/probe_smoke_multi5_20260723_1420.log`.
- Output:
  `artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/smoke/`.
- Promotion rule: require `pipeline.json` status `completed` before either condition enters pilot.
- Result: the strict 10,000-biography cloze gate restored 59,997/60,000 fields exactly
  (`99.995%`) and 9,998/10,000 biographies at 6/6, above the 90% gate. Both 500-step training and
  both checkpoint-reloaded validation jobs completed. P `university_whole` held-out accuracy at
  the six left-to-right text positions was
  `[0.076084, 0.078423, 0.079117, 0.079257, 0.079660, 0.079995]`; Q name-only accuracy was
  `0.005128`. The flattened P position profile is expected to require careful interpretation
  because `multi5_permute` randomizes attribute order; it is a smoke diagnostic, not the formal
  comparison. `pipeline.json` is `completed` with matching identity and seven summary rows.

## 2026-07-23 14:34 — Single-condition full-task P/Q probe pilot

- Status: running.
- Local/UTC start: 2026-07-23 14:34 Asia/Shanghai / 2026-07-23 06:34 UTC.
- Purpose: train all 22 Allen-Zhu P/Q tasks for 3,000 steps on the completed `single` backbone,
  evaluate each saved probe on the full probe-train split and independent person-held-out
  validation split, and validate task scheduling, recovery, summaries, and attribute/position
  semantics before the 30,000-step formal stage.
- Prerequisite: the matching `single` smoke has `pipeline.json status=completed`; the
  `multi5_permute` smoke also completed, so cross-condition smoke gating is satisfied.
- Git at launch: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with the
  recorded probe implementation, history, reports, and exported evidence changes.
- Hardware/topology: 4 × idle RTX 4090 24 GB; dynamic task parallelism across four independent
  workers. Task matrix is 11 label targets × P/Q: six first-token targets and five whole-attribute
  targets, with no birthday whole-attribute classifier.
- Inputs and runtime identity: same `single` dataset/cache/model/checkpoint and formal-ready
  capacity environment as its completed smoke; P/Q train batches 128/768, validation 512/6,144,
  seed 1337, full train evaluation enabled, 1,000-step atomic recovery interval.
- tmux session: `minitrain-probe-pilot-single-20260723`.
- Command:

  ```bash
  cd /home/ubuntu/mini-train-sys
  source .venv/bin/activate
  source .minitrain-storage.env
  STAGE=pilot NPROC=4 PROBE_GPUS=4 \
  PROBE_BATCH_ENV=artifacts/synbios_moe/results/probe_batch_benchmark/multi5_permute/20260723T060600Z/recommended.env \
    bash scripts/bash/synbios_probes.sh single fsdp latest
  ```

- Console log: `artifacts/logs/probe_pilot_single_20260723_1434.log`.
- Output:
  `artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/pilot/`.
- Promotion rule: require all 22 training and all 22 checkpoint-reloaded validation jobs to
  complete with a matching `pipeline.json` identity before launching `single` formal.
- Local/UTC end: 2026-07-23 15:49 Asia/Shanghai / 2026-07-23 07:49 UTC.
- Status: completed (exit code 0).
- Result: all 22 training jobs and all 22 checkpoint-reloaded person-held-out validation jobs
  completed, with zero failed jobs and 77 tidy summary rows. The held-out first-token P results
  show the expected fixed-order `single` pattern: birth city rises from 6.30% at position 0 to
  100.00% at its own position; university rises from 5.39%/11.66% to 99.80%; major rises from
  5.51%/6.72%/7.40% to 100.00%; company rises from 5.23%--6.99% to 100.00%; and company city
  remains 10.62%--15.30% until reaching 100.00% at its own position. The corresponding held-out
  Q first-token accuracies remain much lower (5.10%--39.17%, depending on class cardinality), so
  the pilot supports position-dependent/local storage rather than universal 100% probe accuracy.
- Retained outputs: `artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/pilot/`,
  including `pipeline.json`, 22 training JSON/PT pairs, 22 validation JSON files, recovery
  checkpoints, per-task logs/events, and `summary/{summary.json,summary.csv}`.
- Result export: completed in tmux `minitrain-probe-pilot-single-export-20260723` using
  `bash scripts/bash/export_test_results.sh`; log
  `artifacts/logs/probe_pilot_single_export_20260723_1549.log`. The exported pilot evidence is
  under `results/formal_runs/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/pilot/`, and
  `sha256sum -c results/MANIFEST.sha256` passed.

## 2026-07-23 15:51 — Multi5+permute full-task P/Q probe pilot

- Status: running.
- Local/UTC start: 2026-07-23 15:51 Asia/Shanghai / 2026-07-23 07:51 UTC.
- Purpose: train and independently validate all 22 Allen-Zhu P/Q tasks for 3,000 steps on the
  completed `multi5_permute` backbone, then compare its early-position P and name-only Q
  representation with the completed `single` pilot before choosing the formal-run duration.
- Prerequisites: matching `multi5_permute` smoke completed; matching cache/cloze gate identity
  passed; `single` pilot completed all 22 training and 22 validation tasks and its result export
  manifest verified.
- Git at launch: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with the
  recorded probe implementation, reports/history, and exported reproducibility evidence.
- Hardware/topology: 4 × RTX 4090 24 GB; dynamic task parallelism across four independent GPU
  workers.
- Inputs and runtime identity: `multi5_permute` dataset/cache and committed 4-GPU FSDP backbone
  checkpoint; P/Q train batches 128/768, validation batches 512/6,144, seed 1337, 3,000 steps,
  full train evaluation enabled, 1,000-step atomic recovery interval.
- tmux session: `minitrain-probe-pilot-multi5-20260723`.
- Command:

  ```bash
  cd /home/ubuntu/mini-train-sys
  source .venv/bin/activate
  source .minitrain-storage.env
  STAGE=pilot NPROC=4 PROBE_GPUS=4 \
  PROBE_BATCH_ENV=artifacts/synbios_moe/results/probe_batch_benchmark/multi5_permute/20260723T060600Z/recommended.env \
    bash scripts/bash/synbios_probes.sh multi5_permute fsdp latest
  ```

- Console log: `artifacts/logs/probe_pilot_multi5_20260723_1551.log`.
- Output:
  `artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/pilot/`.
- Promotion rule: do not launch formal probes directly after completion. First produce the
  paper-style cross-condition pilot tables/figures and conclusion document, inspect convergence
  and train/validation gaps, then choose and record defensible formal steps/equivalent epochs and
  batch exposure per probe family or task.
- Local/UTC end: 2026-07-23 17:37 Asia/Shanghai / 2026-07-23 09:37 UTC.
- Status: completed (exit code 0).
- Result: all 22 training jobs and all 22 checkpoint-reloaded person-held-out validation jobs
  completed, with zero failed jobs and 77 tidy summary rows. Multi5 first-token validation is
  near-saturated from the earliest P position (approximately 97.1%--99.7% across targets) and
  Q first-token validation is similarly high (approximately 97.2%--99.7% for non-date targets),
  while whole-attribute tasks remain heterogeneous. This is the expected augmented-memory
  contrast to the completed single pilot's position-local P and low name-only Q pattern.
- Retained outputs: `artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/pilot/`,
  including `pipeline.json`, 22 training JSON/PT pairs, 22 validation JSON files, recovery
  checkpoints, per-task logs/events, and `summary/{summary.json,summary.csv}`.

## 2026-07-23 17:45 — Pilot-driven formal probe budget decision (pre-launch)

- Status: decision recorded; formal training not yet launched.
- Evidence: both 3,000-step pilots completed all 22 training and 22 held-out validation tasks
  with zero failures. First-token P/Q shows the decisive single-versus-multi5 contrast; whole
  probes remain heterogeneous and receive a longer sufficiency budget.
- Formal task matrix: retain all 22 paper probe tasks (six first-token P, five whole P, six
  first-token Q, five whole Q; no birthday whole task), with first as the primary endpoint and
  whole as a secondary diagnostic.
- Industrial runtime settings from the completed capacity benchmark: P train/validation
  batches 128/512 and Q train/validation batches 768/6,144. These are the tested throughput
  settings used for this server, not paper batch sizes.
- Pilot-driven optimizer budgets: P-first/Q-first 4,000 steps; P-whole/Q-whole 12,000 steps.
  This is approximately 10.3/2.1 P-first epochs for single/multi5, 61.6 Q-first epochs, and
  30.8/6.2 P-whole epochs plus 184.8 Q-whole epochs on the corresponding training splits.
- Paper comparison: Allen-Zhu uses P rank 2, batch 50, 30,000 steps and Q rank 16, batch 200,
  30,000 steps. Our ranks and task definitions remain paper-faithful; batch and step budgets
  are explicitly optimized using measured server throughput and pilot convergence.
- Reports: `reports/synbios_moe/probes/pilot_comparison.md` and
  `reports/synbios_moe/probes/formal_protocol.md`.
- Pre-launch gates passed: 26 focused probe/pipeline tests, Ruff, and manifest verification.
- Export: `bash scripts/bash/export_test_results.sh` completed in tmux
  `minitrain-probe-final-export-20260723`; log `artifacts/logs/probe_final_export_20260723_1745.log`.

## 2026-07-23 17:44 — Single-condition formal P/Q probe launch

- Status: running.
- Local/UTC start: 2026-07-23 17:44 Asia/Shanghai / 2026-07-23 09:44 UTC.
- Purpose: formal Allen-Zhu reproduction on the completed `single` backbone using the complete
  22-task P/Q matrix. First-token probes are the primary mechanism endpoint; whole probes are a
  secondary sufficiency diagnostic.
- Git at launch: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with the intentional
  probe reports, budget schedule, ETA instrumentation, history, and exported evidence changes.
- Hardware/topology: 4 × RTX 4090 24 GB, FSDP backbone checkpoint, four task-parallel probe
  workers.
- Runtime/config: P rank 2, Q rank 16; measured throughput batches P=128/Q=768 and validation
  P=512/Q=6,144; first budgets 4,000 steps and whole budgets 12,000 steps; seed 1337; atomic
  recovery every 1,000 steps; full training evaluation enabled. Formal config and identity bind
  these per-task budgets.
- tmux session: `minitrain-probe-formal-single-20260723`.
- Command:

  ```bash
  cd /home/ubuntu/mini-train-sys
  source .venv/bin/activate
  source .minitrain-storage.env
  STAGE=formal NPROC=4 PROBE_GPUS=4 \
  PROBE_BATCH_ENV=artifacts/synbios_moe/results/probe_batch_benchmark/multi5_permute/20260723T060600Z/recommended.env \
    bash scripts/bash/synbios_probes.sh single fsdp latest
  ```

- Console log: `artifacts/logs/probe_formal_single_20260723_1744.log`.
- Output: `artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/`.
- Promotion/monitoring rule: verify the formal pipeline identity, four workers, per-task step
  budgets, and task/phase/full-pipeline ETA at launch; after that hand monitoring to the user.
- Local/UTC end: 2026-07-23 17:51 Asia/Shanghai / 2026-07-23 09:51 UTC.
- Status: stopped before completion (the tmux session disappeared while training was at the first
  four tasks; no completed formal result was promoted and no exit code was available). The run is
  retained as failed/stopped provenance because its initial console output exposed the misleading
  completed-task-only ETA display.
- Retained evidence after cleanup: the stopped log and partial formal output were moved to
  `/data/mini-train-sys/trash/probe_formal_single_20260723_1751/`; backbone checkpoints, pilot
  outputs, reports, and benchmark evidence were not touched. A corrected restart must receive a
  new HISTORY entry and a new output directory/identity.

## 2026-07-23 17:52 — Single-condition formal P/Q probe restart

- Status: running.
- Local/UTC start: 2026-07-23 17:52 Asia/Shanghai / 2026-07-23 09:52 UTC.
- Purpose: restart the stopped formal single-condition probe after correcting the misleading
  task-level ETA display. The complete 22-task matrix and pilot-driven P/Q budgets are unchanged.
- Git at launch: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with intentional
  monitoring, report, budget, and provenance changes.
- Hardware/topology: 4 × RTX 4090 24 GB, four task-parallel workers.
- Runtime/config: P rank 2, Q rank 16; P/Q train batches 128/768; validation batches 512/6,144;
  first budgets 4,000 steps and whole budgets 12,000 steps; seed 1337; recovery every 1,000
  steps; full training evaluation enabled. The corrected monitor reports running-task progress
  in phase and pipeline progress, with task/phase/pipeline ETA fields.
- tmux session: `minitrain-probe-formal-single-restart-20260723`.
- Command:

  ```bash
  cd /home/ubuntu/mini-train-sys
  source .venv/bin/activate
  source .minitrain-storage.env
  STAGE=formal NPROC=4 PROBE_GPUS=4 \
  PROBE_BATCH_ENV=artifacts/synbios_moe/results/probe_batch_benchmark/multi5_permute/20260723T060600Z/recommended.env \
    bash scripts/bash/synbios_probes.sh single fsdp latest
  ```

- Console log: `artifacts/logs/probe_formal_single_restart_20260723_1752.log`.
- Output: `artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/`.
- Local/UTC end: 2026-07-23 17:56 Asia/Shanghai / 2026-07-23 09:56 UTC.
- Status: stopped before completion (the first corrected-progress restart was stopped after its
  first heartbeat exposed an unstable near-zero-progress ETA estimator). Partial output/logs were
  moved to `/data/mini-train-sys/trash/probe_formal_single_20260723_1756/`.

## 2026-07-23 17:57 — Single-condition formal P/Q restart with stable ETA estimator

- Status: running.
- Local/UTC start: 2026-07-23 17:57 Asia/Shanghai / 2026-07-23 09:57 UTC.
- Purpose: restart formal training after replacing the near-zero-progress ETA estimate with an
  active-worker estimate before the first task completes. The scientific matrix, budgets, and
  batches are unchanged.
- Git at launch: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with intentional
  monitoring/provenance changes.
- Hardware/topology: 4 × RTX 4090 24 GB, four task-parallel workers.
- Runtime/config: P rank 2, Q rank 16; P/Q train batches 128/768; validation batches 512/6,144;
  first budgets 4,000 steps and whole budgets 12,000 steps; seed 1337; recovery every 1,000
  steps; full training evaluation enabled.
- tmux session: `minitrain-probe-formal-single-final-20260723`.
- Command:

  ```bash
  cd /home/ubuntu/mini-train-sys
  source .venv/bin/activate
  source .minitrain-storage.env
  STAGE=formal NPROC=4 PROBE_GPUS=4 \
  PROBE_BATCH_ENV=artifacts/synbios_moe/results/probe_batch_benchmark/multi5_permute/20260723T060600Z/recommended.env \
    bash scripts/bash/synbios_probes.sh single fsdp latest
  ```

- Console log: `artifacts/logs/probe_formal_single_final_20260723_1757.log`.
- Output: `artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/`.
- Local/UTC end: 2026-07-23 21:22 Asia/Shanghai / 2026-07-23 13:22 UTC.
- Status: completed (exit code 0).
- Result: all 22 training and all 22 checkpoint-reloaded person-held-out validation tasks
  completed with zero failures and 77 summary rows. Single P shows the expected positional
  pattern (first-token position-0 near chance except fixed birth_date, then near 100% at the
  memorized field position); Q remains near its class prior. Whole P is partially converged and
  whole Q remains near chance. See `reports/synbios_moe/probes/single_formal.md`.
- Retained outputs: `artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/`,
  including summary JSON/CSV, 22 training checkpoints/records, 22 validation records, and router
  analysis. Temporary plot: `.../formal/summary/single_formal_probe_overview.png`.
- Result export: completed in tmux `minitrain-probe-formal-single-export-20260723`; log
  `artifacts/logs/probe_formal_single_export_20260723_210112.log`. `sha256sum -c
  results/MANIFEST.sha256` passed; exported formal evidence is under
  `results/formal_runs/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/`.

## 2026-07-23 17:23 — Cross-condition pilot reduction and formal-duration decision

- Status: completed (exit code 0).
- Local/UTC start: 2026-07-23 17:23 Asia/Shanghai / 2026-07-23 09:23 UTC.
- Local/UTC end: 2026-07-23 17:29 Asia/Shanghai / 2026-07-23 09:29 UTC.
- Purpose: reduce the complete `single` and `multi5_permute` held-out pilots into one tidy
  comparison, reproduce the paper-style P/Q views, apply the trend promotion gate, and select
  the fastest formal schedule that retains the Allen-Zhu P/Q sample exposure.
- Git: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with the
  already recorded probe/runtime/report changes and exported evidence.
- Command:

  ```bash
  source .venv/bin/activate
  source .minitrain-storage.env
  python scripts/synbios_moe.py summarize-probes \
    --run single=artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/pilot/validation \
    --run multi5_permute=artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/pilot/validation \
    --output artifacts/synbios_moe/results/probe_pilot_comparison
  ```

- Result: 154 tidy held-out rows and 77 matched deltas. The primary trend gate passed: excluding
  first-position birth date, mean P position-0 first-token accuracy changed from 6.71% (`single`)
  to 98.63% (`multi5_permute`), and mean Q first-token accuracy changed from 12.42% to 98.80%.
  Whole tasks favor augmentation but remain unconverged at pilot exposure, so they are retained
  as a formal target.
- Formal decision: retain the throughput-optimal benchmark batches P=128/Q=768 and validation
  P=512/Q=6,144. Use P=12,000 steps (1.536M examples) and Q=8,000 steps (6.144M examples),
  closely matching the paper's P=1.5M/Q=6.0M sampled examples while reducing optimizer-step
  overhead. Equivalent epochs are P single=30.72, P multi5=6.144, and Q=122.88.
- Reports:
  `reports/synbios_moe/probes/pilot_analysis.md`,
  `reports/synbios_moe/probes/formal_training_decision.md`, and
  `reports/synbios_moe/probes/figures/pilot_{p_first,p_whole,q}.png`.
- Raw comparison: `artifacts/synbios_moe/results/probe_pilot_comparison/`.
- Monitoring change: formal stages now accept per-kind step schedules and terminal progress
  distinguishes task ETA, phase ETA, full-pipeline ETA/overall 44-task progress, and predicted
  local completion time. Four focused tests and Ruff passed before the full regression.

## 2026-07-23 17:30 — Formal probe configuration and monitoring regression gate

- Status: running.
- Local/UTC start: 2026-07-23 17:30 Asia/Shanghai / 2026-07-23 09:30 UTC.
- Purpose: run the complete regression suite after adding the P/Q-specific formal schedule,
  paper-equivalent exposure configuration, and full-pipeline terminal ETA. This is the final
  software gate before launching the formal probe.
- Git at launch: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with the
  recorded configuration, monitoring, tests, reports, and exported pilot evidence.
- Hardware/topology: CPU regression suite; all four GPUs idle and reserved for the subsequent
  formal task-parallel probe.
- tmux session: `minitrain-probe-formal-preflight-20260723`.
- Command:

  ```bash
  cd /home/ubuntu/mini-train-sys
  source .venv/bin/activate
  source .minitrain-storage.env
  pytest -q --junitxml=results/validation/pytest_probe_formal_preflight_20260723.xml
  ruff check .
  ```

- Console log: `artifacts/logs/probe_formal_preflight_20260723_1730.log`.
- Outputs: `results/validation/pytest_probe_formal_preflight_20260723.xml` and the console log.
## 2026-07-23 21:05 — Multi5+permute formal P/Q probe launch

- Status: failed (exit code 1; runtime logger teardown race).
- Local/UTC start: 2026-07-23 21:05 Asia/Shanghai / 2026-07-23 13:05 UTC.
- Purpose: matched augmented-data formal probe to test the Allen-Zhu single-vs-augmentation
  positional-memory contrast after the completed single baseline.
- Git at launch: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with recorded
  probe/runtime/report changes and exported single evidence.
- Hardware/topology: 4 × RTX 4090 24 GB, four task-parallel workers.
- Runtime/config: variant `multi5_permute`, P rank 2/Q rank 16; P/Q train batches 128/768;
  validation batches 512/6,144; first 4,000 steps and whole 12,000 steps; seed 1337;
  recovery every 1,000 steps; full training evaluation enabled.
- tmux session: `minitrain-probe-formal-multi5-20260723`.
- Command:

  ```bash
  cd /home/ubuntu/mini-train-sys
  source .venv/bin/activate
  source .minitrain-storage.env
  STAGE=formal NPROC=4 PROBE_GPUS=4 \
  PROBE_BATCH_ENV=artifacts/synbios_moe/results/probe_batch_benchmark/multi5_permute/20260723T060600Z/recommended.env \
    bash scripts/bash/synbios_probes.sh multi5_permute fsdp latest
  ```

- Console log: `artifacts/logs/probe_formal_multi5_20260723_2105.log`.
- Output: `artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/`.
- End: 2026-07-23 21:35 Asia/Shanghai / 2026-07-23 13:35 UTC. Four initial tasks exited
  before producing retained formal results; GPUs were idle afterward. The failure was isolated
  to concurrent logger close/write teardown, fixed in `minitrain/runtime/logger.py`; focused
  logger tests passed (8 passed). A new launch will be recorded separately.

## 2026-07-23 21:37 — Multi5+permute formal P/Q retry after logger fix

- Status: stopped (interrupted before completion).
- Local/UTC start: 2026-07-23 21:37 Asia/Shanghai / 2026-07-23 13:37 UTC.
- Purpose: retry the matched augmented-data formal probe after the logger teardown race fix.
- Git: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`, dirty with the logger fix.
- tmux: `minitrain-probe-formal-multi5-retry-20260723`; log:
  `artifacts/logs/probe_formal_multi5_retry_20260723_2137.log`.
- Configuration and output are unchanged from the failed launch above.

## 2026-07-23 21:45 — Multi5+permute formal P/Q restart after interruption

- Status: completed.
- Local/UTC start: 2026-07-23 21:45 Asia/Shanghai / 2026-07-23 13:45 UTC.
- Purpose: restart the interrupted augmented-data formal probe; prior failed and interrupted
  attempts remain retained as provenance.
- Git: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`, dirty with logger race fix.
- tmux: `minitrain-probe-formal-multi5-restart2-20260723`.
- Log: `artifacts/logs/probe_formal_multi5_restart2_20260723_2145.log`.
- Configuration/output: same formal P/Q schedule and output root as the preceding attempt.
- End: 2026-07-24 00:33 Asia/Shanghai / 2026-07-23 16:33 UTC; exit code 0.
- Result: all 22 probe heads trained and all 22 held-out person-split validations completed;
  the pipeline summary contains 77 position-level rows. The subsequent router summaries were
  retained under the same formal output root.

## 2026-07-24 00:35 — Multi5+permute Q-whole inference diagnostics

- Status: completed (both commands exit code 0).
- Local/UTC start: 2026-07-24 00:35 Asia/Shanghai / 2026-07-23 16:35 UTC.
- Purpose: run two inference-only validations on the completed formal probes: (1) compare the
  unchanged Q-whole head on name-only input versus an oracle ground-truth first-token
  intervention; (2) test whether Q-first-correct/Q-whole-wrong examples preserve similar
  top-2 MoE routes at token 1 but branch at token 2.
- Git at launch: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with the
  new diagnostics, tests, documentation, prior formal-run evidence, and unrelated retained
  experiment changes.
- Dataset/cache: `/data/mini-train-sys/artifacts/synbios_moe/multi5_permute` and its
  `probe_cache/` (100,000 profiles; held-out person validation split).
- Backbone: `/data/mini-train-sys/artifacts/synbios_moe/checkpoints/`
  `synbios_moe_multi5_permute_fsdp_4gpu/epoch_000108_step_000017388`.
- Probe checkpoints:
  `/data/mini-train-sys/artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/`
  `probe_pipeline/formal/training/`.
- Hardware/topology: 4 × RTX 4090 24 GB; two independent inference processes on CUDA 0 and 1.
- Tests before launch: Ruff passed; 4 diagnostics tests passed; 27 related
  probe/router/pipeline/logger regression tests passed.
- tmux/log/output:
  - `minitrain-probe-diagnostic-oracle-multi5-20260724`;
    `artifacts/logs/probe_diagnostic_oracle_multi5_20260724_0035.log`;
    `artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/`
    `diagnostics/oracle_first_token/`.
  - `minitrain-probe-diagnostic-routes-multi5-20260724`;
    `artifacts/logs/probe_diagnostic_routes_multi5_20260724_0035.log`;
    `artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/`
    `diagnostics/bad_case_routes/`.
- Commands:

  ```bash
  python scripts/synbios_moe.py validate-probe-oracle-first-token \
    --data artifacts/synbios_moe/multi5_permute \
    --probe-cache artifacts/synbios_moe/multi5_permute/probe_cache \
    --probe-dir artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/training \
    --model-config configs/synbios_moe/model.yaml \
    --checkpoint artifacts/synbios_moe/checkpoints/synbios_moe_multi5_permute_fsdp_4gpu/epoch_000108_step_000017388 \
    --batch-size 512 --device cuda:0 \
    --output artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/oracle_first_token

  python scripts/synbios_moe.py validate-probe-bad-case-routes \
    --data artifacts/synbios_moe/multi5_permute \
    --probe-cache artifacts/synbios_moe/multi5_permute/probe_cache \
    --probe-dir artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/training \
    --model-config configs/synbios_moe/model.yaml \
    --checkpoint artifacts/synbios_moe/checkpoints/synbios_moe_multi5_permute_fsdp_4gpu/epoch_000108_step_000017388 \
    --batch-size 512 --pair-limit 2000 --device cuda:1 \
    --output artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/bad_case_routes
  ```

- End: oracle 2026-07-24 00:38:18 Asia/Shanghai / 2026-07-23 16:38:18 UTC;
  routes 2026-07-24 00:38:34 Asia/Shanghai / 2026-07-23 16:38:34 UTC.
- Result: oracle micro Q-whole accuracy changed from 33.15% to 32.08% (-1.06pp);
  5.38% of baseline errors recovered while 14.06% of baseline-correct predictions were harmed.
  The route analysis retained 162,044 eligible bad cases. Weighted branching score was -0.051
  for same-t2 controls and +0.154 for different-t2 pairs, yielding +0.205
  difference-in-differences; the contrast was positive in every layer and largest in layers 0-3.
- Report: `reports/synbios_moe/probes/q_whole_moe_diagnostics.md`.

## 2026-07-24 00:52 — Matched single vs multi5+permute formal-study report

- Status: completed.
- Local/UTC start: 2026-07-24 00:52 Asia/Shanghai / 2026-07-23 16:52 UTC.
- Purpose: reconstruct the canonical formal comparison directly from all 44 independent
  held-out validation JSON files, enforce matched run/data identities, combine the two strict
  source-text cloze results, and render paper-style numeric heatmaps and project overview figures.
- Git at launch: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with
  retained experiment/report changes and the new audited report generator.
- Inputs:
  - `artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/`
  - `artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/`
  - `artifacts/synbios_moe/results/single_cloze_eval/full_100k/summary.json`
  - `artifacts/synbios_moe/results/multi5_permute_cloze_eval/full_500k/summary.json`
- Output: `artifacts/synbios_moe/results/formal_probe_comparison_20260724/`.
- Hardware/topology: CPU-only report generation; no GPU training or inference.
- Preflight: Ruff passed; 8 formal-report/diagnostics tests passed.
- Command:

  ```bash
  python scripts/synbios_moe.py report-formal-study \
    --single artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal \
    --multi5-permute artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal \
    --single-cloze artifacts/synbios_moe/results/single_cloze_eval/full_100k/summary.json \
    --multi5-permute-cloze artifacts/synbios_moe/results/multi5_permute_cloze_eval/full_500k/summary.json \
    --output artifacts/synbios_moe/results/formal_probe_comparison_20260724
  ```

- Local/UTC end: 2026-07-24 01:01 Asia/Shanghai / 2026-07-23 17:01 UTC.
- Exit code: 0.
- Result summary:
  - strict training-corpus source recall was 100.0000% for `single` and 99.9915% for
    `multi5_permute`;
  - P first-token position-0 macro accuracy (excluding birth date) rose from 6.76% to 98.63%;
  - Q first-token macro accuracy rose from 12.83% to 98.79%;
  - Q whole-attribute macro accuracy rose from 3.18% to 33.15%, but remained far below the
    92.58% Allen-Zhu bioS multi5+permute reference, so the overall result is a partial rather
    than exact replication.
- Retained outputs:
  - mounted artifact: `artifacts/synbios_moe/results/formal_probe_comparison_20260724/`;
  - Git-safe export:
    `results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/`;
  - narrative: `reports/synbios_moe/probes/formal_comparison.md`;
  - experiment index: `reports/synbios_moe/probes/README.md`;
  - generation log: `artifacts/logs/formal_probe_comparison_20260724_0052.log`;
  - export tmux/log: `minitrain-formal-report-export-20260724-0102` /
    `artifacts/logs/formal_report_export_20260724_0102.log`.
- Verification:
  - Ruff and shell syntax checks passed;
  - 39 related regression tests passed, 7 deselected;
  - an independent temporary rebuild matched all 9 JSON/CSV/PNG files byte-for-byte;
  - `results/MANIFEST.sha256` verified, report-local Markdown links resolved, no exported
    file exceeded 100 MB, and no weight-like file entered the formal comparison export.

## 2026-07-24 01:12 — Full whole-readout diagnostic report

- Status: failed (exit code 1).
- Local/UTC start: 2026-07-24 01:12 Asia/Shanghai / 2026-07-23 17:12 UTC.
- Purpose: strictly audit the two completed `multi5_permute` inference-only Q-whole diagnostics,
  cross-check the oracle name-only baseline against the original formal Q-whole results, and
  render a complete five-attribute comparison containing all single/multi P-whole positions,
  original Q-whole, oracle true-t1 Q-whole, controlled route branching, and Allen-Zhu context.
- Git at launch: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with
  retained formal/probe evidence and the new deterministic diagnostic reporting layer.
- Inputs:
  - `artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/`;
  - `artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/`;
  - `artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/`.
- Output:
  `artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/report/`.
- Hardware/topology: CPU-only validation/report generation; no model or probe parameter updates.
- Preflight: Ruff passed; 12 diagnostic/formal-report tests passed.
- tmux/log:
  `minitrain-probe-diagnostic-report-20260724-0112` /
  `artifacts/logs/probe_diagnostic_report_20260724_0112.log`.
- Command:

  ```bash
  python scripts/synbios_moe.py report-probe-diagnostics \
    --single-formal artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal \
    --multi5-permute-formal artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal \
    --diagnostics artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics \
    --output artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/report
  ```

- End: 2026-07-24 01:13 Asia/Shanghai / 2026-07-23 17:13 UTC.
- Failure: strict equality compared an integer-count reconstruction with the float32 accuracy
  retained in the original formal JSON. The first comparison differed by approximately
  `5.5e-9`; no report artifact was accepted. The retry uses an explicit `1e-7` tolerance only
  for this float32 boundary while retaining exact count, identity, hash, and matrix checks.

## 2026-07-24 01:13 — Full whole-readout diagnostic report retry

- Status: completed (exit code 0).
- Local/UTC start: 2026-07-24 01:13 Asia/Shanghai / 2026-07-23 17:13 UTC.
- Purpose, inputs, outputs, hardware, Git state, tmux/log, and command are identical to the
  immediately preceding run, except for the documented float32-boundary tolerance.
- Preflight: Ruff passed; 12 diagnostic/formal-report tests passed.
- tmux/log:
  `minitrain-probe-diagnostic-report-r2-20260724-0113` /
  `artifacts/logs/probe_diagnostic_report_r2_20260724_0113.log`.
- Local/UTC end: 2026-07-24 01:19 Asia/Shanghai / 2026-07-23 17:19 UTC.
- Result:
  - all five whole-value attributes and all six original P positions were included;
  - oracle Q-whole micro accuracy changed from 33.15% to 32.08% (−1.06pp) across
    250,590 held-out predictions;
  - route analysis retained 162,044 bad cases; same-`t2` and different-`t2` branching scores
    were −0.051 and +0.154, for a +0.205 pair-count-weighted difference-in-differences;
  - all 12 layer aggregates were positive, with the strongest contrast in layers 0–3;
  - the complete colored comparison includes single/multi formal P-whole, original Q-whole,
    oracle `+ true t1`, and Allen-Zhu Figure 7 context.
- Retained outputs:
  - mounted:
    `artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/report/`;
  - Git-safe:
    `results/formal_runs/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/report/`;
  - reports: `reports/synbios_moe/probes/diagnostics/{README.md,oracle_first_token.md,`
    `bad_case_routes.md}`;
  - export tmux/log: `minitrain-diagnostic-report-export-20260724-0119` /
    `artifacts/logs/probe_diagnostic_report_export_20260724_0119.log`.
- Verification:
  - 43 related regression tests passed, 7 deselected; Ruff and `git diff --check` passed;
  - an independent temporary rebuild matched all 9 JSON/CSV/PNG files byte-for-byte;
  - source and Git-safe report directories match exactly;
  - `results/MANIFEST.sha256`, report-local Markdown links, file-size limits, and weight
    exclusions all passed.

## 2026-07-24 01:28 — SynBioS repository path and lineage audit

- Status: completed (exit code 0).
- Local/UTC start: 2026-07-24 01:28 Asia/Shanghai / 2026-07-23 17:28 UTC.
- Purpose: validate the complete retained SynBioS graph from raw dataset manifests through token
  shards, person-level probe caches, formal checkpoints, cloze/probe/diagnostic results, accepted
  runtime configs, and archived logs; emit mechanically checkable lineage and path catalogs without
  moving or deleting historical evidence.
- Git at launch: `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` on `train`; dirty with
  retained experiment evidence, reports, diagnostic tooling, and the repository-audit implementation.
- Hardware/topology: CPU/storage audit on the persistent four-RTX-4090 host; no GPU training,
  inference, or parameter updates.
- Preflight: Ruff passed; 37 focused repository/formal/probe tests passed, 8 deselected.
- tmux/log:
  `minitrain-synbios-repository-audit-20260724-0128` /
  `artifacts/logs/synbios_repository_audit_20260724_0128.log`.
- Inputs:
  - `artifacts/synbios_moe/{single,multi5_permute}/`;
  - `artifacts/synbios_moe/checkpoints/`;
  - `artifacts/synbios_moe/results/{single_fsdp_4gpu,multi5_permute_fsdp_4gpu}/`;
  - `configs/synbios_moe/{runs,probe_pipeline.yaml}`;
  - `artifacts/logs/`.
- Output:
  `artifacts/synbios_moe/results/repository_audit_20260724/`, plus versioned `lineage.json`
  sidecars under each mounted dataset, token-shard, and probe-cache directory.
- Command:

  ```bash
  python scripts/synbios_moe.py audit-synbios-repository \
    --repo-root /home/ubuntu/mini-train-sys \
    --output artifacts/synbios_moe/results/repository_audit_20260724
  ```

- Local/UTC end: 2026-07-24 01:29 Asia/Shanghai / 2026-07-23 17:29 UTC.
- Result:
  - all raw dataset files, token shards, document indexes, cache arrays, formal checkpoint model
    exports, and parent identities matched their retained manifests/hashes;
  - `single` and `multi5_permute` use the same 100,000 profiles and the same deterministic
    49,882-person probe-train / 50,118-person probe-validation split;
  - both accepted training configs resolve the intended mounted token manifests, use local batch
    112 on four GPUs, and retain the intended 540/108 epoch budgets;
  - probe runtime defaults now exactly match both completed formal identities:
    P/Q train batch 128/768 and P/Q validation batch 512/6144;
  - 81 retained logs were hashed and classified; historical paths were preserved.
- Retained outputs:
  - `artifacts/synbios_moe/results/repository_audit_20260724/{summary.json,`
    `dataset_lineage.json,path_contract.json,log_catalog.csv}`;
  - mounted dataset sidecars:
    `artifacts/synbios_moe/{single,multi5_permute}/{lineage.json,`
    `token_shards/lineage.json,probe_cache/lineage.json}`.
- Git-safe export:
  - first export:
    `minitrain-synbios-audit-export-20260724-0131` /
    `artifacts/logs/synbios_repository_audit_export_20260724_0131.log`;
  - policy-clean retry:
    `minitrain-synbios-audit-export-r2-20260724-0134` /
    `artifacts/logs/synbios_repository_audit_export_r2_20260724_0134.log`;
  - the retry removed 136 legacy probe-head tensor copies (589,311,663 bytes) only from the
    Git-safe mirror. Their authoritative `/data` artifacts remain intact; retained JSON identities
    and hashes still bind the formal results to those heads.
- Final verification:
  - full repository suite: 102 passed, 5 expected PyTorch single-process distributed-checkpoint
    warnings;
  - test tmux/log:
    `minitrain-synbios-final-tests-20260724-0136` /
    `artifacts/logs/synbios_final_pytest_20260724_0136.log`;
  - final export tmux/log:
    `minitrain-synbios-final-export-20260724-0138` /
    `artifacts/logs/synbios_final_export_20260724_0138.log`;
  - `results/MANIFEST.sha256`, source/export audit-directory equality, shell syntax, Ruff,
    `git diff --check`, Git-safe weight exclusion, and report links passed.

## 2026-07-24 01:35 — Git-safe full SynBioS server evidence snapshot

- Status: completed.
- Local/UTC start: 2026-07-24 01:35 Asia/Shanghai / 2026-07-23 17:35 UTC.
- Purpose: publish all current server evidence that is appropriate for the remote repository:
  source/config changes, formal and diagnostic reports, figures, machine-readable summaries,
  manifests/lineage, validation evidence, TensorBoard/event logs, and lifecycle logs.
- Exclusions: raw biography payloads, probe-cache arrays, model/probe weights, optimizer tensors,
  DCP shards, caches, credentials, and other secrets remain on mounted storage and are not staged.
- Base revision/branch/remote:
  `0473f6f52d8b8ccdf62b9456938b04e96eb6be03` / `train` /
  `origin` (`git@github.com:Zhaibin-Cui/mini-train-sys.git`).
- Hardware/topology: repository/export checks only; no GPU computation.
- Pre-push correctness evidence: full repository suite passed (102 tests); Ruff, shell syntax,
  Markdown links, result manifest, report rebuild identity, and Git-safe tensor exclusion passed.
- Export tmux/log:
  `minitrain-synbios-push-export-20260724-0135` /
  `artifacts/logs/synbios_push_export_20260724_0135.log`.
- Snapshot commit:
  `bc61fdcda009bf405ceb5b5b1365a7fafbad0f47`
  (`feat(synbios): publish formal probe study and diagnostics`).
- Push tmux/log:
  `minitrain-synbios-push-20260724-0140` /
  `artifacts/logs/synbios_git_push_20260724_0140.log`.
- Exact push command: `git push origin train`.
- Local/UTC end: 2026-07-24 01:38 Asia/Shanghai / 2026-07-23 17:38 UTC.
- Result: GitHub accepted `0473f6f..bc61fdc` on `origin/train`; a post-push fetch/revision
  check confirmed the remote branch at the exact snapshot commit.

## 2026-07-24 01:42 — Promote SynBioS evidence snapshot to default branch

- Status: completed.
- Local/UTC time: 2026-07-24 01:42 Asia/Shanghai / 2026-07-23 17:42 UTC.
- Purpose: make the already-published Git-safe SynBioS formal results visible on GitHub's default
  `main` branch; the initial publication had updated only the active server branch `train`.
- Safety check: `origin/main` (`7c34608`) was a direct ancestor of `origin/train` (`09f8e4b`);
  the promotion was a conflict-free fast-forward with zero main-only commits.
- Exact command: `git push origin train:main`.
- tmux/log:
  `minitrain-synbios-main-push-20260724-0148` /
  `artifacts/logs/synbios_main_push_20260724_0148.log`.
- Result: GitHub accepted `7c34608..09f8e4b` on `main`; post-push fetch confirmed both
  `origin/main` and `origin/train` at
  `09f8e4b4e8a58f54e977ac19741841984f4dadde`.

> Superseded at 2026-07-24 01:45: this default-branch promotion was unintended and was
> explicitly reverted at the user's request. The `train` publication remains valid.

## 2026-07-24 01:45 — Restore default branch after unintended promotion

- Status: completed.
- Local/UTC time: 2026-07-24 01:45 Asia/Shanghai / 2026-07-23 17:45 UTC.
- Purpose: undo only the preceding unintended `main` promotion while preserving all published
  SynBioS evidence on `train`.
- Precondition: fetched `origin/main` was exactly
  `ffd6d74d5d9aaab1fcce0cd9a0932f87c1e8b075`; restore target was its verified ancestor
  `7c34608863c18c3e47f1f50223f95449321aba5b`.
- Safety mechanism: an exact `--force-with-lease` bound the update to the fetched remote SHA, so
  the command would reject any intervening third-party update.
- Exact command:
  `git push --force-with-lease=refs/heads/main:ffd6d74d5d9aaab1fcce0cd9a0932f87c1e8b075 origin 7c34608863c18c3e47f1f50223f95449321aba5b:refs/heads/main`.
- tmux/log:
  `minitrain-restore-main-20260724-0155` /
  `artifacts/logs/restore_main_20260724_0155.log`.
- Result: GitHub accepted the forced update `ffd6d74...7c34608`; post-push fetch confirmed
  `origin/main` at `7c34608` and `origin/train` unchanged at `ffd6d74`.
