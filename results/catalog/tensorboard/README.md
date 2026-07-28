# 📈 TensorBoard index

TensorBoard events remain inside their owning run directory. This prevents a formal `single`
curve from being silently merged with `multi5_permute`, pilot, smoke, or probe-head events.

[`index.csv`](index.csv) records every exported event file with condition, stage, relative path,
and byte size. To inspect one stage, use the corresponding owner directory:

```bash
tensorboard --logdir results/pretraining/synbios_moe --port 6006 --bind_all
# or: results/cloze/synbios_moe
# or: results/probes/synbios_moe
```

For a cleaner view, choose one path from `index.csv` and pass only that condition/run directory.
Console logs follow the same stage-based layout.
