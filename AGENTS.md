# Local server operating rules

This checkout runs on a persistent multi-GPU server. Follow these rules for all future work in
this repository.

## Long-running work

- Run every training job, benchmark, notebook batch execution, CUDA extension build, dataset
  preparation job, and any other command that may outlive an SSH/API session inside `tmux`.
- Give each tmux session a descriptive, unique name such as `minitrain-train-single`,
  `minitrain-ddp4`, or `minitrain-cuda-build`.
- Before starting project commands, change to the repository root, activate `.venv`, and source
  `.minitrain-storage.env`.
- Write stdout and stderr to a timestamped file under `artifacts/logs/`, which is stored on the
  mounted data disk. Do not rely only on tmux scrollback.
- After a server benchmark finishes, copy its complete result directory (raw JSON, logs, CSV,
  figures, and summaries) into `results/benchmarks/`. This path is intentionally
  versioned and should be included when preparing a Git commit or push.
- Run `bash scripts/bash/export_test_results.sh` after every benchmark/validation cycle so all
  Git-appropriate results are copied from the mounted disk into the repository before handoff.
- Before creating a session, check `tmux list-sessions` and do not overwrite or kill an existing
  user session.
- After launching work, verify that the tmux pane is still running and report the session name,
  log path, and attach command (`tmux attach -t <name>`).
- Do not run long experiments directly in a transient shell. Short read-only checks and quick
  smoke tests that finish in a few seconds may run directly.

## Formal run history

- Every formal training, evaluation, benchmark, probe, or dataset-generation run must be recorded
  in the repository-root `HISTORY.md`. This requirement applies whether the run is launched by an
  agent or manually on the server.
- Add the history entry when the run is launched, not after it is forgotten. Record the local and
  UTC start time, purpose, exact command, Git commit and dirty state, tmux session, log path,
  artifact/result paths, hardware topology or GPU count, and important configuration overrides.
- When the run ends, update the same entry with end time, status (`completed`, `failed`, or
  `stopped`), exit code when available, result summary, and links/paths to retained outputs.
- Never rewrite or delete prior run records. Append new entries and preserve failed/stopped runs;
  they are part of the experiment provenance.

## Result persistence and temporary remote snapshots

- Persist as much reproducibility evidence as practical in Git after every benchmark, validation,
  formal-run milestone, and before every commit or push. Run
  `bash scripts/bash/export_test_results.sh`, review `results/MANIFEST.sha256`, and include the
  resulting small/medium artifacts in the same commit as the code/configuration that produced them.
- Keep `results/` structured by purpose: `benchmarks/`, `validation/`, `formal_runs/`, `datasets/`,
  `environment/`, `logs/`, and `smoke/`. Maintain `results/BENCHMARK_SUMMARY.md` as the human-readable
  index; do not leave important conclusions discoverable only by opening raw JSON files.
- Record raw and aggregate metrics, exact commands/configs, environment and topology inventories,
  Git revision/dirty state, notebook cell outputs, JUnit reports, event logs, checkpoint runtime/RNG
  metadata, `COMMITTED` markers, data manifests, content hashes, and failure/OOM logs whenever they
  are available.
- Never commit secrets, SSH/private keys, access tokens, raw dataset payloads, model weights,
  optimizer tensor shards, DCP `distributed/` directories, or caches. For excluded large artifacts,
  persist their manifest, size, checksum, logical path, and retention status instead.
- Before pushing a temporary server snapshot, check for files over GitHub's size limit, scan the
  staged diff for credentials, run the relevant tests, and record the pushed commit in `HISTORY.md`.
  Temporary snapshot commits are valuable provenance and must not be silently rewritten or removed.

## Storage and machine profile

- Keep source code and `.venv` in this checkout on the system disk.
- Keep datasets, checkpoints, logs, experiment results, package caches, Triton caches, and CUDA
  build outputs under the paths configured by `.minitrain-storage.env` on `/data`.
- This host has four RTX 4090 24 GB GPUs, NVIDIA driver 525.105.17, and CUDA Toolkit 11.8. Match
  runtime and build choices to the detected machine state; re-check hardware before changing
  CUDA or PyTorch versions.
- Use 4-GPU FSDP for formal training experiments by default. DDP may still be run for explicit
  comparison benchmarks and short diagnostics, but do not choose DDP for a formal run unless the
  user explicitly requests it.
- Before a formal training launch, use the latest capacity benchmark to select the largest tested
  local batch whose run completed successfully and whose peak allocated GPU memory is at most 95%
  (the user explicitly accepted approximately 92% on this server).
  Persist that batch in the matching run YAML and record the benchmark evidence and configuration
  change in `HISTORY.md`; do not launch formal training with an older conservative default.
- Performance results are not a correctness proof. Before formal FSDP training, require the full
  regression suite, Torch-vs-Triton operator correctness, short single/DDP/FSDP training checks,
  FSDP checkpoint save/resume, and a multi-step stability run at the selected batch. Record these
  gates in `HISTORY.md`; do not launch formal training if any required gate fails.
