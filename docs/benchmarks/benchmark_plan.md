# Benchmark 共同约定

## 先正确，再计时

每个优化算子先与 Torch baseline 比较 forward、backward、dtype 和边界 shape。性能 case
必须记录是否真正进入优化 kernel；fallback 结果不能标成 Triton/CUDA 性能。

## 算子 benchmark

- 固定 seed 和输入；
- warmup 后 CUDA synchronize；
- 报告 P50/P95，不只报最快一次；
- 记录 allocated/reserved 峰值；
- sweep dtype、shape、stride 和 masked tail；
- raw JSON 包含 GPU、CUDA、PyTorch、Git revision 和命令参数。

入口为 `tests/operator_bench.ipynb`、`tests/moe_operator_bench.ipynb` 和
`tests/operator_bench_utils.py`。

## 训练 benchmark

固定模型、数据顺序、精度和 local/global batch。记录 step time、tokens/s、显存、
data wait、loss 和实际 backend。不要把 compile 首次开销混入 steady-state。

## 分布式 benchmark

- weak scaling：固定每卡 batch，看 step time 是否稳定、吞吐是否按 N 扩展；
- fixed-global-batch：固定 global batch，看 strong scaling，但需说明每卡 batch 变化；
- capacity：每个 batch 用独立进程，OOM 是容量边界数据；
- 同时记录 NCCL topology、最慢 rank 时间和全系统显存。

当前 1/4/8 卡协议与验收门槛见 [distributed_benchmark.md](distributed_benchmark.md)。
