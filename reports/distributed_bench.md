# SynBioS MoE 的 FSDP 扩展效率

问题是：同一个正式模型、同一个每卡 batch，从 1 张 RTX 4090 扩到 4 张后，token
吞吐能否接近四倍。

比较条件完全一致：293.49M 参数的 SynBioS MoE、序列长度 512、BF16、FSDP full
shard、local batch 64，以及 benchmark preset 产生的固定随机 token 流。1 卡与 4 卡
各运行两次，每次预热 5 step、测量 20 step。这里测的是训练计算与通信 scaling，不把
语料 I/O 差异混进主指标。

| GPU | global batch | 平均吞吐 | 相对单卡 | 弱扩展效率 | 每卡峰值显存 |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 93,302 tok/s | 1.00× | 100.00% | 14,940 MB |
| 4 | 256 | 344,254 tok/s | 3.69× | 92.24% | 12,413 MB |

![FSDP weak scaling](../results/benchmarks/synbios_moe_fsdp4/weak_b64/presentation/weak_overview.png)

结论：4 卡没有达到理论 4.00×，但达到 3.69×，平均弱扩展效率为 92.24%，两次重复
分别为 92.08% 和 92.40%。4 卡的数据等待仅占 0.10%，因此主要损失来自 FSDP
通信与同步，而不是 dataloader。结果通过重复完成、效率 ≥80%、data stall ≤5% 和
显存余量 ≥10% 四项门槛。

机器可读证据位于
`results/benchmarks/synbios_moe_fsdp4/weak_b64/weak_summary.json`，聚合表、质量门槛
和图片位于同目录的 `presentation/`。运行入口是
`scripts/run_dist_bench.py`，notebook 入口是
`tests/distributed_server_benchmark.ipynb`，正式记录见 `HISTORY.md` 的
“SynBioS exact-model FSDP weak-scaling verification”。

容量扫描是另一个问题：4 卡 local batch 112 达到 370,857 tok/s、每卡峰值
20,875 MB；local batch 120 虽然仍能运行，但吞吐下降到 281,926 tok/s。因此正式
训练选择 112，不把“能塞进显存”误当成“性能最优”。

限制：这是单机四卡 PCIe 拓扑、随机 token 输入上的弱扩展结果，不等价于真实语料端到端
吞吐、跨节点扩展，也不能外推到不同模型尺寸或不同互联。Torch/Triton/CUDA 的下一轮
正式 launch 对比则显式使用 `multi5_permute` token shards。

## 下一轮 backend 对比口径

正式 293.49M SynBioS MoE 的 Torch/Triton/CUDA 对比将分成两个独立 headline：

1. 固定 common batch：比较同样 token 工作量下的吞吐、step time 和
   allocated/reserved 显存，回答相对 Torch 节省多少内存、提升多少速度；
2. 固定每卡 92% peak-reserved VRAM 上限：每个 backend 选择重复完成的最高吞吐 batch，
   回答同样显存空间下提升多少 tokens/s。

两种口径尚未产生正式数据。容量聚合由
`scripts/run_dist_bench.py present-capacity` 完成，完整执行计划见
[`scripts/server_benchmark_codex_prompt.md`](../scripts/server_benchmark_codex_prompt.md)。
该对比只使用当前正式约 300M MoE 和真实 `multi5_permute` token shards；125M 通用模型
和 Mixtral-class shape 不进入整体训练 headline。
