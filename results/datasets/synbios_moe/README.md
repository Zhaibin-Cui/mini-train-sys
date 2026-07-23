# SynBioS dataset catalog

这里仅保存 Git-safe manifest、lineage、统计和校验结果；raw biographies、token shards 与
probe cache 数组保留在 `/data/mini-train-sys/artifacts/synbios_moe/`，不进入 Git。

| condition | people | biographies | train tokens | P examples | Q examples |
|---|---:|---:|---:|---:|---:|
| `single` | 100,000 | 100,000 | 7,405,102 | 100,000 | 100,000 |
| `multi5_permute` | 100,000 | 500,000 | 37,046,556 | 500,000 | 100,000 |

两组使用相同的 `profiles.jsonl`（SHA256
`7d239f046cb5e16ac3d8d7636b6901a2430f2ccb8dc1179063e4eaed92256da1`）和 seed
1337。probe 按 person 固定划分为 49,882 train / 50,118 validation；由于所有 100,000
个人都出现在 backbone 的预训练语料中，这里的 validation 是 **held-out probe
supervision**，不是未被 backbone 见过的人或文档。预训练 token manifest 的 validation
split 为空。

每个 condition 目录包含三层证据：

- `manifest.json`：raw dataset 身份、生成设置、文件大小与 SHA256；
- `token_shards/{manifest.json,lineage.json}`：tokenizer、document index、shard
  hashes、token/document counts 和 parent manifest hash；
- `probe_cache/{manifest.json,lineage.json}`：P/Q 缓存 schema、任务类别覆盖、split
  语义、所有 cache 文件 hashes 和 parent manifest hash。

完整跨层校验结果见
[`repository_audit_20260724`](../../formal_runs/synbios_moe/results/repository_audit_20260724/summary.json)；
对应生命周期与命令见仓库根目录 `HISTORY.md` 的
“2026-07-24 01:28 — SynBioS repository path and lineage audit”。
