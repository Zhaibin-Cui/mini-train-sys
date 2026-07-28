# Experiment results

`results/` is the Git-safe export of the server experiment volume. Evidence is organized by the
stage that produced it:

| Stage | Contents |
|---|---|
| [`pretraining/`](pretraining/) | Dataset and token-shard lineage, formal training events, and checkpoint metadata |
| [`cloze/`](cloze/) | Source-biography recall for `single` and `multi5_permute`, including shard summaries and evaluation logs |
| [`benchmarks/`](benchmarks/) | Kernel, backend-capacity, distributed-scaling, probe-capacity, and executed-notebook evidence |
| [`probes/`](probes/) | Probe caches, formal P/Q runs, mechanism diagnostics, and matched comparisons |

Cross-stage records stay under [`catalog/`](catalog/): the
[SynBioS study index](catalog/studies/synbios_moe.json), exported-file catalog, retention policy,
TensorBoard index, retention inventory, and export audits. [`environment/`](environment/) records the
server hardware and software stack.

Start with [`SUMMARY.md`](SUMMARY.md) for headline measurements. The reports under
[`../reports/`](../reports/) explain how the retained files support each conclusion.

## Export contract

Large mutable payloads—raw biographies, token arrays, model weights, optimizer shards, probe heads,
and per-example route records—remain on `/data`. Git retains their manifests and lineage together
with compact metrics, figures, events, and checkpoint metadata.

Refresh and verify the snapshot with:

```bash
bash scripts/bash/export_results.sh
python scripts/build_results_manifest.py --results results --check
```

Every exported file except the manifest itself is covered by [`MANIFEST.sha256`](MANIFEST.sha256).
TensorBoard events remain beside the run that produced them; use
[`catalog/tensorboard/index.csv`](catalog/tensorboard/index.csv) to locate them.
