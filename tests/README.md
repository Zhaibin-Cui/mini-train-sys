# 测试与 Notebook

## 自动测试

先安装 `pip install -e ".[data,dev]"`；`dev` 已包含 pytest、Jupyter、pandas 和 matplotlib，
`data` 提供 SynBioS 使用的 tiktoken。

```bash
python -m pytest -q
python -m ruff check minitrain scripts tests
```

`test_*.py` 验证数据边界、配置、DDP/FSDP 包装契约、DCP 恢复、日志和 SynBioS。
Linux CI 可运行多进程 Gloo 测试；真实 NCCL/FSDP 性能只能在目标 GPU 服务器验收。

`test_distributed_bench_utils.py` 使用合成报告验证服务器 benchmark 的命令构造、重复聚合、
quality gates、capacity frontier、Notebook 高层调用，以及 CSV/JSON/PNG 展示产物落盘；
它不伪造真实 GPU 性能结果。

其中 `test_model_training_metrics.py` 验证 Dense/MoE loss 拆分、每层专家分布和兼容接口；
`test_runtime_logger.py` 验证 TensorBoard 标量、专家 ratio histogram、固定色标 heatmap，以及
probe 训练健康指标和 pipeline 终端格式；`test_synbios_moe.py` 验证 probe 的逐位置准确率、
全局梯度范数、DataLoader 等待指标、任务 started/heartbeat/finished 事件，并检查端到端
notebook 始终覆盖独立 validation、smoke/pilot/formal 和后处理入口。
它还检查 pipeline identity 会绑定所有正式输入、重复任务配置被拒绝、返回成功但未落盘的
子进程被判失败、summary 的磁盘 JSON 包含 comparison，以及服务器一键入口没有绕过三阶段 gate。

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
