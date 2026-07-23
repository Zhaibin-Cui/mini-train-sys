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

## Conclusions and report organization

- Keep conclusions deliberate, structured, and easy to audit. `results/BENCHMARK_SUMMARY.md` is the
  canonical cross-run index, while `reports/README.md` indexes experiment-specific reports. Update
  both when a run changes a headline result, accepted configuration, failure boundary, or next step.
- Organize new reports by experiment, condition, and stage when more than one file is needed, using
  paths such as `reports/<experiment>/<condition>/<stage>.md`. Preserve existing published paths;
  do not move historical reports merely to satisfy a newer layout convention.
- Every conclusion document must state, in order: the question or hypothesis, exact compared
  conditions, run/checkpoint and dataset identity, primary metrics, supporting artifact paths,
  interpretation, limitations or threats to validity, and the next decision/action. Clearly label
  training-set recall, held-out validation, smoke checks, failed runs, and formal results.
- Put comparison tables and aggregate metrics in the human-readable report, not only in raw JSON,
  TensorBoard, logs, or notebook cells. Link each headline number to its machine-readable summary
  and the matching `HISTORY.md` entry. Do not duplicate incompatible headline values across files;
  if a conclusion is superseded, retain it as historical evidence and mark what supersedes it.
- Keep raw outputs, intermediate summaries, plots, and final conclusions separate. Raw evidence
  belongs under `results/`; narrative conclusions belong under `reports/`; commands, lifecycle,
  failures, and provenance belong in `HISTORY.md`.

## Dataset and derived-data organization

- Treat every dataset condition and derived cache as a versioned experiment input. Large SynBioS
  payloads remain under `artifacts/synbios_moe/<variant>/` on `/data`; their Git-safe mirrors belong
  under `results/datasets/synbios_moe/<variant>/`. Keep derived probe data under the owning
  condition, for example `artifacts/synbios_moe/<variant>/probe_cache/`, rather than in an
  unlabelled shared directory.
- Every dataset or cache must have an authoritative manifest recording its format/protocol version,
  generator command and seed, source/parent dataset identity, sample and token counts, split
  definition, important preprocessing or augmentation settings, file sizes and SHA256 hashes.
  Derived manifests must identify the parent manifest hash so lineage is mechanically checkable.
- Store train/validation/test split semantics explicitly and distinguish person-level splits from
  document-level or training-corpus evaluation. Never describe a training-corpus recall result as
  held-out validation. Check for split overlap, missing ranges, duplicate shards, and class
  coverage before a formal run.
- Do not mix files from different seeds, variants, protocol versions, tokenizers, or partial reruns
  in one logical dataset directory. Write new versions to a new path or rebuild atomically; validate
  the manifest before use. Preserve failed or obsolete manifests as provenance, clearly marked, and
  never let a pipeline silently reuse them.
- Do not commit raw dataset payloads or caches. Export their manifests, compact statistics, lineage,
  checksums, validation results, and retention locations to Git, and include the dataset status in
  the relevant report and `results/BENCHMARK_SUMMARY.md`.

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
