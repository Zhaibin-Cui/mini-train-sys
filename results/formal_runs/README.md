# Formal experiment evidence

`synbios_moe/` contains Git-safe evidence for:

- `runs/`: pretraining JSONL and TensorBoard events;
- `checkpoints/`: `COMMITTED`, runtime/RNG, and DCP layout metadata without tensor shards;
- `results/`: cloze, formal P/Q, comparison figures, diagnostics, and compact summaries;
- `operation_logs/`: dataset-preparation operation events.

Published result paths remain stable. Use
[`reports/synbios_moe/README.md`](../../reports/synbios_moe/README.md) for the readable map and
[`results/tensorboard/index.csv`](../tensorboard/index.csv) to locate event files.
