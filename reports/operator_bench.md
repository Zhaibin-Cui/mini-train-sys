# Operator benchmark

通用 dense 算子入口仍是 `tests/operator_bench.ipynb`，MoE 的 RTX 4090 正式入口是
`tests/moe_operator_bench_linux_server.ipynb`。

MoE notebook 只保留三步：构造 `MoeOperatorBenchmark`、运行 scan、展示结果。复杂的
shape、子进程隔离、失败保留、汇总和绘图集中在
`tests/moe_operator_bench_runner.py`；通用计时与正确性实现仍由
`tests/operator_bench_utils.py` 提供。

扫描包含：

- router：1,024 到 262,144 tokens；
- project-formal fused MoE：E=8、H=768、I=1024、K=2，最大 57,344 tokens，
  对应正式 local batch 112 × sequence 512；
- Mixtral-class 参考：E=8、H=4096、I=14336、K=2，扫描 16、64、256 tokens。

每个 shape 在新的 CUDA 子进程中执行 Torch/Triton forward、backward 和 full-step
正确性、P50/P95 延迟及峰值显存。OOM、timeout 或 kernel fault 只会形成一个失败 case，
不会污染 Jupyter kernel。

## 之前命令行执行容易失败的原因

旧 notebook 在一个 Jupyter CUDA 进程里直接构造 E=8、H=4096、I=14336 的 BF16 专家
权重。仅两组权重就约 2.82 GB（2.625 GiB）；correctness 同时保留 candidate、Torch
reference 和两边梯度，权重相关存储可超过 10 GiB，再叠加输出、临时 workspace 和
Jupyter 单元残留引用。某个大 case OOM 后，CUDA context 继续被后续 cell 复用，所以会
出现连锁错误，而不只是单个点失败。

另外，旧 notebook metadata 使用泛化的 `python3` kernel；命令行所在虚拟环境与
Jupyter kernel 可能不是同一个 Python。`nbconvert --inplace` 在中断时还可能让源
notebook 留下不完整输出。新的 runner 分别用显式 `mini-train-sys` kernel、逐 shape
子进程和临时 executed 副本解决这三个问题。

服务器不要再使用 `nbconvert --execute --inplace`。标准命令是：

```bash
python scripts/run_server_notebook.py \
  tests/moe_operator_bench_linux_server.ipynb \
  --kernel mini-train-sys \
  --output-dir artifacts/notebooks \
  --timeout-seconds -1
```

安全执行器使用当前虚拟环境的 Python、显式指定 kernel、捕获 cell error，并在成功后
原子发布 executed notebook。失败时保留 failed notebook、错误清单和执行日志。

正式结果写入 `artifacts/operator_benchmark/rtx4090_24gb/`，包含 raw JSON、逐 case
日志、`summary.csv`、`summary.json` 和统一风格图片。实验完成后按 `AGENTS.md` 导出到
`results/benchmarks/`。
