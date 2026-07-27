# Notebook 与 Benchmark 工具

## 教学与端到端 Notebook

| Notebook | 建议何时读 |
|---|---|
| `example_training.ipynb` | 第一次理解 config、LR、Dense/MoE、checkpoint |
| `synbios_moe_end_to_end.ipynb` | 分阶段验证数据→主训练→评估→单 probe/独立 val→P/Q pipeline→router→恢复与监控产物 |
| `distributed_server_benchmark.ipynb` | 在 1/4/8×RTX 4090 服务器测 DDP/FSDP |

## Kernel Notebook

| Notebook/工具 | 用途 |
|---|---|
| `operator_bench.ipynb` | 通用算子正确性和 shape sweep |
| `operator_bench_linux_server.ipynb` | 逐 shape 隔离的 RTX 4090 工业 dense 扫描 |
| `moe_operator_bench_linux_server.ipynb` | 逐 shape 子进程隔离的 Router 与 fused MoE |
| `dense_operator_bench_runner.py` | 正式 293M 与 Mixtral-class dense profile runner |
| `operator_bench_utils.py` | notebook 共用测量/落盘工具 |
| `operator_nsight.py` | Nsight 命令行入口 |

从仓库根目录启动 Jupyter。原始 benchmark 写入
`tests/benchmark_results/<gpu>/<operator>/<timestamp>.json`；不要把 smoke 数字当成
正式性能结论。

工业规模 runner 会把每个 shape 放在独立 CUDA 子进程中；OOM、timeout 和 unsupported
都会保留为证据，不会污染后续 case。

服务器批量执行 notebook 时使用 `scripts/run_server_notebook.py`，不要原地覆盖源
notebook。正式端到端 backend 比较通过 `scripts/run_dist_bench.py present-backend`
生成固定工作量表，通过 `present-capacity` 生成固定 reserved-VRAM 预算表。
