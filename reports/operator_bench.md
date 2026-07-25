# Operator benchmark

通用 dense 算子的开发/完整正确性入口仍是 `tests/operator_bench.ipynb`；RTX 4090
工业 shape 的隔离执行入口是 `tests/operator_bench_linux_server.ipynb`，MoE 正式入口是
`tests/moe_operator_bench_linux_server.ipynb`。

两本 server notebook 都只保留三步：构造 runner、运行 scan、展示结果。复杂的
shape、子进程隔离、失败保留、汇总和绘图分别集中在
`tests/dense_operator_bench_runner.py` 和 `tests/moe_operator_bench_runner.py`；
通用计时与正确性实现仍由
`tests/operator_bench_utils.py` 提供。

扫描包含：

- router：1,024 到 524,288 tokens，包含正式 57,344-token 点；
- project-formal fused MoE：E=8、H=768、I=1024、K=2，最大 57,344 tokens，
  对应正式 local batch 112 × sequence 512；
- Mixtral-class 参考：E=8、H=4096、I=14336、K=2，扫描到 512 tokens；dense
  Mixtral-class 附录扫描到 1,024 tokens。末档用于确认原扫描边界之外的趋势。

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

## RTX 4090 正式结果

下一轮不再把算子结果和端到端训练结果混在一起。算子主表固定覆盖：

| 优化实现 | Kernel |
|---|---|
| Triton | RMSNorm、RoPE、SwiGLU、CrossEntropy、FusedLinearCrossEntropy、FlashAttention、Router postprocess、Fused MoE |
| native CUDA | FlashAttention |

每个 kernel 选择一个预先定义、最接近项目 workload 的代表 shape，同时报告
forward/backward/full-step P50/P95、相对 Torch speedup、latency reduction、peak-memory
reduction 和正确性。raw sweep 继续完整保留。CUDA facade 对 attention 之外的算子仍会
fallback，不能写成八个手写 CUDA kernel。

主 profile 使用正式 293.49M MoE 的 57,344 tokens/rank、H=768、I=1024、D=64、
vocab 50,257、E=8、K=2；Mixtral-class H=4096/I=14336 作为工业维度附录。完整工业
shape 若超出 4090 24 GB，就扫描到所有 backend 的共同安全上限。headline 选择吞吐最优
或平台拐点，不选择“显存占用最大”的点；搜索边界必须向外扩一档。

正式扫描已完成并通过质量门。相对 shape-matched Torch 的 full-step headline：

| Kernel | Backend | Speedup | Peak allocated reduction |
|---|---|---:|---:|
| RMSNorm | Triton | 6.51× | 78.7% |
| RoPE | Triton | 2.57× | 14.3% |
| SwiGLU | Triton | 1.42× | 20.0% |
| CrossEntropy | Triton | 4.54× | 83.3% |
| FusedLinearCrossEntropy | Triton | 1.39× | 94.0% |
| FlashAttention | Triton | 1.22× | 44.1% |
| FlashAttention | native CUDA | 1.15× | 22.1% |
| Router postprocess | Triton | 2.43× | 65.2% |
| Fused MoE | Triton | 1.46× | 5.3% |

这里的 Torch attention 不是朴素 eager 基线。项目的 Torch backend 调用
`torch.nn.functional.scaled_dot_product_attention`；在本机 PyTorch 2.5.1+cu118、
RTX 4090、BF16、causal、D=64 配置下，现场 profiler 确认 forward dispatch 到
`aten::_scaled_dot_product_flash_attention` / `aten::_flash_attention_forward`，
backward dispatch 到 `aten::_scaled_dot_product_flash_attention_backward` /
`aten::_flash_attention_backward`。因此上表 Triton 1.22× 和 native CUDA 1.15×
均是相对 **PyTorch fused Flash-SDPA** 的提升，不能描述成相对 naive attention。
机器可读 dispatch 证据见
`results/benchmarks/operator_benchmark/resume_summary/torch_attention_backend.json`。

除两个 loss 外均使用完整 57,344-token formal shape；显式 loss 因 Torch 在大 shape OOM，
按预注册规则使用所有 backend 共同完成的 8,192-token 吞吐最优点。Triton fused loss
另行在 57,344 tokens 完成容量测量：full P50 163.12 ms、peak allocated delta
356.1 MiB，但该点没有伪造 Torch speedup。

规范报告、CSV、JSON 和图位于
`artifacts/operator_benchmark/resume_summary/`，Git-safe 镜像由结果导出脚本写入
`results/benchmarks/operator_benchmark/resume_summary/`。完整执行契约见
[`scripts/server_benchmark_codex_prompt.md`](../scripts/server_benchmark_codex_prompt.md)。
