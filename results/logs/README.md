# 🧾 Exported server logs

每个长时间 tmux 任务先写入 `artifacts/logs/`，导出时按用途放入四个稳定类别：

| Directory | Contents |
|---|---|
| `benchmarks/` | Kernel builds/scans、backend capacity、FSDP scaling |
| `experiments/` | SynBioS preparation、training、cloze、probes、diagnostics |
| `validation/` | Regression、preflight、fidelity 和 correctness gates |
| `maintenance/` | Export、环境安装、repository snapshot 与 cleanup |

`HISTORY.md` 仍是权威 lifecycle 记录；文件名本身不能证明一次运行是 formal 或成功。
全仓库机器索引见 [`../catalog/artifacts.json`](../catalog/artifacts.json)。
