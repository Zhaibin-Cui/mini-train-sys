# Local repository rules

## Before running work

- Work from the repository root and load `.minitrain-storage.env` on the experiment server.
- Put long training, benchmark, build, and dataset jobs in a named `tmux` session.
- Keep large datasets, checkpoints, caches, and raw logs under the configured `artifacts/` paths.
- Check the current GPU and disk state before starting a formal job.

## Experiments

- Run smoke and pilot gates before formal training or probes.
- Use the matching YAML config; do not hide formal settings in an ad-hoc shell command.
- Do not rebuild a dataset after checkpoints exist unless the old checkpoints are archived.
- `single` and `multi5_permute` must share the same profile-table hash.
- A probe split contains people seen during pretraining unless a report explicitly says otherwise.

## Evidence

- Keep raw machine output in `results/`; keep interpretation in `reports/`.
- Every retained conclusion must link its dataset lineage, config, command, implementation,
  checkpoint or run identity, machine-readable result, and report.
- `results/catalog/studies/synbios_moe.json` is the formal SynBioS map.
- Regenerate the result catalog and `results/MANIFEST.sha256` after changing exported evidence.
- Do not commit raw datasets, model weights, optimizer shards, caches, secrets, or credentials.

## Code and verification

- Keep SynBioS code grouped under `pretraining/`, `probes/`, and `mechanisms/`.
- Keep formal performance work under `benchmarks/` and runnable demonstrations under `examples/`.
- Do not enable postponed annotations through a future import.
- Verify changed commands directly, then run Ruff, forbidden-content search, and `git diff --check`.

## Local notes

- `HISTORY.md` records only the purpose, final command, and retained result of relevant runs.
- Do not add retry diaries, timestamps, tmux transcripts, dirty-worktree descriptions, or discarded
  conclusions.
- `AGENT.md`, `AGENTS.md`, and `HISTORY.md` are local and gitignored.
