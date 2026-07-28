# Benchmark 共同约定

## 先正确，再计时

每个优化算子先与 Torch baseline 比较 forward、backward、dtype 和边界 shape。性能 case
必须记录是否真正进入优化 kernel；fallback 结果不能标成 Triton/CUDA 性能。

## 算子 benchmark

- 固定 seed 和输入；
- warmup 后 CUDA synchronize；
- 报告 P50/P95，不只报最快一次；
- 报告 forward、backward-only 和 full forward+backward；
- 同时报告相对 Torch 的 speedup、latency reduction 和 peak-memory reduction；
- 记录 allocated/reserved 峰值；
- sweep dtype、shape、stride 和 masked tail；
- raw JSON 包含 GPU、CUDA、PyTorch、Git revision 和命令参数。

简历主表覆盖 RMSNorm、RoPE、SwiGLU、CrossEntropy、FusedLinearCrossEntropy、
FlashAttention、Router postprocess 和 Fused MoE。代表 shape 的选择规则必须在看到结果
前确定；raw sweep 全部保留，不能事后只挑最快的点。

工业 profile 选择顺序：

1. 优先使用当前正式 293.49M SynBioS MoE 的实际 shape：local batch 112、sequence
   512、57,344 tokens/rank、H=768、I=1024、12 heads、D=64、vocab 50,257、E=8、K=2；
2. 用 Mixtral-class H=4096/I=14336/E=8/K=2 检查工业维度；
3. 若完整 profile 超过 4090 24 GB，以相同候选网格寻找各 backend 的共同安全上限；
4. 在 ≤92% peak-reserved VRAM 内选择吞吐最优点或性能平台拐点，而不是选择显存占用
   最大的点；若最优点落在扫描边界，必须扩展一档。

显式 CrossEntropy 的 formal `tokens × vocab` logits 若无法由 Torch reference 完成，
应保留 OOM，并在最大共同安全 shape 比较速度；不得把只有优化 backend 能运行的 shape
伪装成成对 speedup。FusedLinearCrossEntropy 是否能避免该显式 logits 空间，作为独立
空间结论报告。

开发期完整正确性入口为 `benchmarks/operators/operator_bench.ipynb`；正式隔离执行入口为
`benchmarks/operators/operator_bench_linux_server.ipynb` 和
`benchmarks/operators/moe_operator_bench_linux_server.ipynb`，runner 为
`benchmarks/operators/dense_runner.py`、`benchmarks/operators/moe_runner.py`。
CUDA backend 当前只有 attention 是 native CUDA；
其他算子的 CUDA 名称若发生 fallback，不能当作 CUDA kernel 性能。

## 训练 benchmark

固定模型、数据顺序、精度和 local/global batch。记录 step time、tokens/s、显存、
data wait、loss 和实际 backend。不要把 compile 首次开销混入 steady-state。

Backend 比较必须同时回答两个不同问题：

1. **固定工作量**：相同模型、batch、warmup、measure steps 和 repeats，比较吞吐与
   allocated/reserved memory，报告相对 Torch 的速度提升和显存节省；
2. **固定空间**：给所有 backend 同一个 peak-reserved VRAM 上限，在重复完成的安全
   case 中分别选择最高吞吐 batch，报告同样显存预算下的 tokens/s 提升。

固定工作量结果不能冒充固定空间结果。固定空间选择以 reserved memory 为物理预算，
不能用 allocated memory 替代。

内存优化还必须扣除静态训练状态：在共同 batch `N` 上，使用
`peak allocated(N) - peak allocated(batch 1)` 作为激活内存增长主指标，并报告相对
Torch 的 reduction 和每新增 local sample 的斜率。该差分抵消相同 backend 内的权重、
梯度、优化器及大部分 FSDP 静态状态；reserved 差分只解释 allocator，原始 reserved
峰值仍负责容量安全。吞吐是端到端产出率，直接比较 tokens/s，不做 batch-1 扣减。

正式端到端 backend benchmark 固定使用当前 293,494,272 参数 SynBioS MoE；125M
通用模型只保留为基础设施历史证据，不进入简历主对比。

## 分布式 benchmark

- weak scaling：固定每卡 batch，看 step time 是否稳定、吞吐是否按 N 扩展；
- fixed-global-batch：固定 global batch，看 strong scaling，但需说明每卡 batch 变化；
- capacity：每个 batch 用独立进程，OOM 是容量边界数据；
- 同时记录 NCCL topology、最慢 rank 时间和全系统显存。

当前 1/4/8 卡协议与验收门槛见 [distributed_benchmark.md](distributed_benchmark.md)。
正式 SynBioS backend 简历 benchmark 的执行入口与两阶段交付规范见
运行入口和输出位置见 [`benchmarks/README.md`](../../benchmarks/README.md)。
