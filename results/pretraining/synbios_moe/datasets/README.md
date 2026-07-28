# SynBioS pretraining data

Raw biographies and token arrays remain under `/data/mini-train-sys/artifacts/synbios_moe/`.
This directory retains the manifests and lineage required to reproduce the pretraining inputs.

| Condition | People | Biographies | Train tokens |
|---|---:|---:|---:|
| `single` | 100,000 | 100,000 | 7,405,102 |
| `multi5_permute` | 100,000 | 500,000 | 37,046,556 |

Both conditions use seed 1337 and the same `profiles.jsonl`. Each condition contains:

- `manifest.json`: generation settings, raw-file sizes, and SHA-256 hashes;
- `lineage.json`: source dataset identity;
- `token_shards/manifest.json`: tokenizer identity, document counts, offsets, and shard hashes;
- `token_shards/lineage.json`: the parent-manifest link used by pretraining.

Probe-cache manifests are kept under
[`../../../probes/synbios_moe/cache/`](../../../probes/synbios_moe/cache/), because they are derived
for the P/Q experiments rather than consumed by pretraining.
