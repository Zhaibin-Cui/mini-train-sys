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

- Status: running from a fresh step 0 (automatic resume explicitly disabled)
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

## 2026-07-21 04:42 — Temporary GitHub server snapshot

- Status: preparing push to `origin/train`
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
