# Cloze validation

The cloze stage checks whether each pretrained model can reproduce the six facts in its source
biographies before representation probing begins.

| Condition | Retained result | Coverage |
|---|---|---:|
| `single` | [`synbios_moe/single/full_100k/`](synbios_moe/single/full_100k/) | 100,000 biographies / 600,000 fields |
| `multi5_permute` | [`synbios_moe/multi5_permute/full_500k/`](synbios_moe/multi5_permute/full_500k/) | 500,000 biographies / 3,000,000 fields |

Each condition directory contains the aggregate summary, contiguous shard results, and operation
logs. Console logs copied from the server are in [`synbios_moe/logs/`](synbios_moe/logs/).
