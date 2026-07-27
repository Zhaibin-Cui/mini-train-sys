# 给服务器 Codex 的精简 Benchmark 提示词

把下面整段发给服务器上的 Codex。目标不是继续扩展实验范围，而是产出两组可以放进简历、
项目 README 和技术报告的可信数字：

1. 我优化了哪些核心 kernel，以及各 kernel 的正确性、速度提升和空间节省；
2. 正式 293M SynBioS MoE 在 4×RTX 4090 FSDP 下：
   - 相同训练工作量时节省多少显存、提升多少吞吐；
   - 相同显存上限时可以提升多少 token 训练吞吐。

---

你现在位于 `mini-train-sys` 仓库。先完整阅读根目录 `AGENTS.md`，严格遵守 tmux、挂载盘、
日志、`HISTORY.md` 和结果导出规则。机器预期为 4 × RTX 4090 24 GB。

## 总规则

1. 整个 benchmark 只分两个阶段：**kernel microbenchmark** 和
   **formal end-to-end benchmark**。ground-truth-t1 fresh P probe 属于另一条科学实验线，不要混入
   本 benchmark。
2. 每阶段启动前只做一次必要预检，然后给出准确命令、预计耗时和北京时间。得到明确同意
   后启动；阶段完成后停下来汇报，不要自动进入下一阶段。
3. 长任务必须在独立 tmux 中运行，stdout/stderr 写入
   `artifacts/logs/<timestamp>.log`。启动时写 `HISTORY.md`，结束后补全状态和结果。
4. 不得隐藏 OOM、timeout、fallback、正确性失败或慢结果。所有正式对比必须使用相同
   shape、dtype、warmup、measure steps 和重复次数。
5. 每阶段结束后运行 `bash scripts/bash/export_test_results.sh`。原始 JSON/CSV/log、
   executed notebook、图片和聚合报告全部保留。
6. 最终只需要一次总汇报和一个简历友好的规范报告，不要生成多份互相重复的结论文档。
7. 算子 benchmark 使用工业 workload 优先，而不是为了好跑缩成 toy shape。第一优先级
   是本项目正式 293.49M MoE 的真实 per-rank/per-layer shape；再加入 Mixtral-class
   工业维度。如果完整工业 shape 超过单张 4090 24 GB，就用相同候选网格逐档逼近所有
   对比 backend 的共同安全上限。最大显存占用不是优化目标：主结果选择正确性通过、
   ≤92% peak-reserved VRAM 内的吞吐最优点或明确的平台拐点，同时保留 OOM、吞吐回落
   和容量边界。

## 阶段 1：逐 kernel benchmark

### 预检

```bash
cd ~/mini-train-sys
source .venv/bin/activate
source .minitrain-storage.env
git status --short
git rev-parse HEAD
nvidia-smi
nvidia-smi topo -m
python -m ruff check tests/operator_bench_utils.py \
  tests/moe_operator_bench_runner.py scripts/run_server_notebook.py minitrain/kernels
```

确认四张 GPU 空闲后，安装明确的 notebook kernel，并构建正式模型需要的 sm89、head-dim
64、BF16/FP16 CUDA FlashAttention：

```bash
python -m ipykernel install --user --name mini-train-sys \
  --display-name mini-train-sys
export MINITRAIN_CUDA_BUILD_PROFILE=rtx4090
export MINITRAIN_CUDA_ARCHS=89
export MINITRAIN_CUDA_MAX_JOBS=2
export MINITRAIN_CUDA_VERBOSE=1
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

通过安全执行器运行两个现有 benchmark notebook，不允许 `--inplace`：

```bash
python scripts/run_server_notebook.py \
  tests/operator_bench_linux_server.ipynb \
  --kernel mini-train-sys \
  --output-dir artifacts/notebooks \
  --timeout-seconds -1

python scripts/run_server_notebook.py \
  tests/moe_operator_bench_linux_server.ipynb \
  --kernel mini-train-sys \
  --output-dir artifacts/notebooks \
  --timeout-seconds -1
```

主表覆盖以下八类 kernel，并明确优化实现归属：

| Kernel | 项目中的优化实现 | 主对比 |
|---|---|---|
| RMSNorm | Triton | Torch vs Triton |
| RoPE | Triton | Torch vs Triton |
| SwiGLU | Triton | Torch vs Triton |
| CrossEntropy | Triton | Torch vs Triton |
| FusedLinearCrossEntropy | Triton | Torch vs Triton |
| FlashAttention | Triton + native CUDA | Torch vs Triton vs native CUDA |
| Router postprocess | Triton | Torch vs Triton |
| Fused MoE | Triton | Torch vs Triton |

CUDA backend 目前只有 attention 是本地 CUDA kernel；其他算子会沿
CUDA → Triton → Torch fallback，因此不得把八个算子都描述成手写 CUDA。

项目正式工业 profile 固定来自当前成功训练的 293,494,272 参数 SynBioS MoE：

```text
local batch = 112
sequence length = 512
tokens per rank/layer = 57,344
hidden = 768
intermediate = 1,024
attention heads = 12
head dim = 64
vocab = 50,257
experts = 8
experts per token = 2
dtype = BF16
```

RMSNorm、RoPE、SwiGLU、FlashAttention、Router、Fused MoE 和
FusedLinearCrossEntropy 应优先包含这个项目 profile。普通 CrossEntropy 的显式
`tokens × vocab` logits 可能超过 24 GB；若 Torch reference 无法完成完整 formal
shape，就以相同 tokens/vocab 候选网格寻找 Torch 与优化实现的最大共同安全点，并把
formal-shape OOM 作为 fused loss 空间价值的证据，而不是伪造速度比。

Mixtral-class `H=4096, I=14336, E=8, K=2` 作为第二工业 profile。token 数按 4090
实际容量逐档增加；若最大成功点仍是搜索边界，扩展一档。它是工业维度的容量/性能附录，
不替代项目正式 profile 的简历 headline。

每个 kernel 至少汇报：

- benchmark shape、dtype 和 GPU；
- forward、backward-only、full forward+backward 的 P50/P95；
- 相对 Torch 的 speedup 和 latency reduction percentage；
- peak allocated memory、memory reduction percentage；若工具能提供，同时保留 peak
  reserved memory；
- forward/backward correctness 和任何 unsupported/OOM/timeout。

原始 sweep 全部保留。简历主表每个 kernel 的代表点按以下固定顺序选择，不能事后挑
最好看的点：

1. 项目 formal shape 若所有对比 backend 正确完成，使用 formal shape；
2. 否则使用所有对比 backend 在 92% reserved-VRAM 内共同完成的吞吐最优点；
3. 若吞吐在更大 shape 已进入平台或下降，使用平台拐点，并同时报告更大点；
4. 若选中点位于扫描边界，扩展一档后才能形成 headline。

将两本 notebook 的结果合并到：

```text
artifacts/operator_benchmark/resume_summary/
├── kernel_benchmark_summary.csv
├── kernel_benchmark_summary.json
├── kernel_benchmark_overview.png
└── README.md
```

`README.md` 必须说明比较条件、代表 shape 选择、正确性门槛、失败项和哪些 backend 真正
调用了 native CUDA。阶段完成并导出后，汇报总耗时与结果，再问是否开始阶段 2。

## 阶段 2：正式 FSDP4 端到端 benchmark

服务器标准入口是
`scripts/bash/synbios_backend_benchmark.sh`。它按本节固定参数依次执行 backend
validation、2A、2B、两份 presentation 和结果导出；任何 correctness、进程或容量门槛
失败都会立即停止。下列展开命令仍是审计规范和单阶段重跑入口。

先用阶段 1 已构建的 CUDA extension 做一次正式 D64 attention 正确性检查：

```bash
python scripts/run_dist_bench.py validate-backend \
  --device cuda:0 \
  --batch-size 2 \
  --sequence-length 512 \
  --output artifacts/distributed_benchmark/synbios_backend_torch_vs_cuda/backend_validation.json
```

验证通过后，在完全相同的正式条件下比较三个 backend。端到端必须同时做两种口径，不能
用其中一个冒充另一个：

1. **固定工作量比较**：相同 local/global batch、相同 warmup/measure steps，比较吞吐和
   显存。它回答“处理同样多 token 时节省多少内存、快多少”。
2. **固定显存预算比较**：每个 backend 都受同一个 92% peak-reserved VRAM 上限约束，
   分别选择该上限内吞吐最高的 batch。它回答“同样 24GB 空间能多训练多少 token/s”。

共同条件：

- 293,494,272 参数 SynBioS MoE；
- 12 layers、8 experts、top-2、sequence length 512；
- BF16、FSDP full shard、4 × RTX 4090；
- 相同数据和训练 step 实现。

整体训练 benchmark 只使用这个已经完成正式训练的约 300M MoE，不换成 125M 通用模型
或 Mixtral 参考模型。Mixtral-class 只属于算子 microbenchmark 附录。

### 2A：固定工作量

优先使用正式 local/global batch 112/448，warmup 10 steps、measure 30 steps、每个
backend 3 次：

```bash
python scripts/run_dist_bench.py run \
  --suite backend \
  --strategies fsdp \
  --world-sizes 4 \
  --ops-backends torch triton cuda \
  --local-batch 112 \
  --warmup-steps 10 \
  --measure-steps 30 \
  --repeats 3 \
  --case-config configs/synbios_moe/runs/multi5_permute_fsdp_4gpu.yaml \
  --model-config configs/synbios_moe/model.yaml \
  --output artifacts/distributed_benchmark/synbios_backend_torch_vs_cuda

python scripts/run_dist_bench.py present-backend \
  --input artifacts/distributed_benchmark/synbios_backend_torch_vs_cuda/backend_summary.json \
  --validation artifacts/distributed_benchmark/synbios_backend_torch_vs_cuda/backend_validation.json \
  --output artifacts/distributed_benchmark/synbios_backend_torch_vs_cuda/presentation
```

实际自动流程先做 2B，再从容量结果中确定三个 backend 共同在 92%
reserved-memory 上限内完成的最大 common batch，最后用该 common batch 执行三组各
3 次，作为固定工作量主比较。batch 112 只是工业级候选点，不得预设它对 Torch、
Triton 和 CUDA 都可用；任一 OOM 都必须保留并进入容量边界报告。

固定工作量主表：

| Backend | Throughput tok/s | Step time ms | Peak allocated/GPU | Peak reserved/GPU | Repeats |
|---|---:|---:|---:|---:|---:|
| Torch | mean ± std | mean ± std | mean ± std | mean ± std | 3 |
| Triton | mean ± std | mean ± std | mean ± std | mean ± std | 3 |
| CUDA | mean ± std | mean ± std | mean ± std | mean ± std | 3 |

同时给出 CUDA vs Torch、Triton vs Torch、CUDA vs Triton 的吞吐 speedup 和 step-time
下降比例，以及 allocated/reserved memory reduction percentage。每个 CUDA case必须满足：

```text
backend_dispatch.attention.native_cuda > 0
backend_dispatch.attention.fallback == 0
```

### 2B：固定 92% 显存预算

对三个 backend 使用完全相同的 batch candidates。初始扫描：

```bash
python scripts/run_dist_bench.py run \
  --suite capacity \
  --strategies fsdp \
  --world-sizes 4 \
  --ops-backends torch triton cuda \
  --batch-sizes 1 2 4 8 16 24 32 48 64 80 96 112 120 128 \
  --warmup-steps 5 \
  --measure-steps 20 \
  --repeats 2 \
  --case-config configs/synbios_moe/runs/multi5_permute_fsdp_4gpu.yaml \
  --model-config configs/synbios_moe/model.yaml \
  --output artifacts/distributed_benchmark/synbios_backend_capacity

python scripts/run_dist_bench.py present-capacity \
  --input artifacts/distributed_benchmark/synbios_backend_capacity/capacity_summary.json \
  --memory-limit-percent 92 \
  --min-repeats 2 \
  --output artifacts/distributed_benchmark/synbios_backend_capacity/presentation
```

对每个 backend，只从以下 case 中选择最高吞吐点：

```text
status == completed
peak_memory_reserved_mb / gpu_memory_total_mb <= 0.92
两次重复都完成
```

如果某 backend 的最优点落在扫描最大 batch 边界，向上扩展一档后复测；OOM 和吞吐回落点
都保留。不得按 allocated memory 选择，物理显存预算以 reserved memory 为准。

固定空间主表：

| Backend | Selected local/global batch | Reserved VRAM % | Throughput tok/s | vs Torch |
|---|---:|---:|---:|---:|
| Torch | best under 92% | ≤92% | mean ± std | baseline |
| Triton | best under 92% | ≤92% | mean ± std | speedup |
| CUDA | best under 92% | ≤92% | mean ± std | speedup |

这个表是“同样显存空间下训练吞吐提升”的唯一 headline 来源。固定 batch 的 2A 结果不能
替代这个结论。

内存 headline 必须同时保留“能否装入”的原始 peak allocated/reserved 和扣除静态模型
状态后的激活增量。对 common batch `N` 使用：

```text
activation_allocated_growth(N) = peak_allocated(N) - peak_allocated(batch 1)
activation_reserved_growth(N) = peak_reserved(N) - peak_reserved(batch 1)
```

其中 allocated 增量是激活内存节省的主要口径，能够抵消相同 backend 内共享的模型权重、
梯度、优化器和大部分 FSDP 静态状态；reserved 增量只说明 allocator 行为。另报告
`activation_allocated_growth / (N - 1)`，用于比较每新增 local sample 的内存斜率。
吞吐不做 batch-1 扣减，固定 batch 和固定显存预算下都直接比较实测 tokens/s。

已有 1 卡与 4 卡 FSDP weak-scaling 结果只作为补充引用，不默认重跑：

- 1 GPU：约 93,302 tok/s；
- 4 GPU：约 344,254 tok/s；
- 3.69× scaling，平均效率 92.24%。

## 最终交付

把两阶段结论分别写入两个规范报告，避免算子设计和端到端/FSDP 表格重复：

```text
reports/engineering/kernels.md
reports/engineering/distributed_training.md
```

两份报告合计按以下顺序组织：

1. 硬件、软件版本和 Git commit；
2. 我优化的八个 kernel，以及各自代表 shape、正确性、P50/P95、speedup 和 memory
   reduction；
3. 固定工作量下三 backend 的吞吐、step time、allocated/reserved memory、均值、标准差
   和显存节省比例；
4. 固定 92% reserved-memory 上限下各 backend 的最佳 batch、吞吐和相对 Torch 提升；
5. 已有 1→4 GPU scaling 作为补充；
6. 原始 JSON/CSV、图片、日志、executed notebook 和 `HISTORY.md` 链接；
7. 限制：单机 PCIe、单一模型尺寸、CUDA 原生实现当前只覆盖 attention。

最终给出两条“可写进简历”的候选表述，但必须等真实数字产生后填写，不得预先编造或只挑
最好的一次运行。先让我审核报告，不要自行 commit 或 push。

---
