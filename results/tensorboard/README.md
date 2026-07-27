# 📈 TensorBoard index

TensorBoard events remain inside their owning run directory. This prevents a formal `single`
curve from being silently merged with `multi5_permute`, pilot, smoke, or probe-head events.

[`index.csv`](index.csv) records every exported event file with condition, stage, relative path,
and byte size. To inspect all SynBioS training and probe events:

```bash
tensorboard --logdir results/formal_runs/synbios_moe --port 6006 --bind_all
```

For a cleaner view, choose one path from `index.csv` and pass only that condition/run directory.
Console logs are separately organized under `results/logs/`.
