<div align="center">

<img src="assets/readme-banner.svg" alt="MiniTrainSys banner" width="100%" />

**一个用于学习 LLM 数据、模型、算子和分布式训练的可读小型系统。**

</div>

## 这个项目解决什么问题

MiniTrainSys 不是通用大模型平台，也不是只有一个 `train.py` 的教学 demo。它把一条
完整训练链拆成可单独阅读、测试和替换的模块：

```text
原始文档 → tokenizer → token shards → Dataset/Sampler/DataLoader
        → MiniTransformer(Dense 或 MoE) → Torch/Triton/CUDA 算子
        → single/DDP/FSDP → AdamW/LR/epoch → DCP checkpoint
        → evaluate / P-probe / Q-probe / router analysis
```

当前最完整的端到端实验是 Allen-Zhu bioS 协议的 SynBioS MoE 复现。项目同时包含
Triton/CUDA kernel 学习、单机 1/4/8×RTX 4090 分布式 benchmark，以及带 Adam 状态的
可恢复训练。

## SynBioS MoE 服务器实验报告

> 实验快照：2026-07-21，4 × NVIDIA GeForce RTX 4090 24 GB。以下只报告服务器上已有、
> 可追溯到原始 JSONL/TensorBoard/checkpoint 的结果。`single` 与 `multi5_permute` 均已完成
> 正式训练及全训练语料 progressive-cloze 验证；正式 P/Q probe 尚未运行，
> 不能把运行中指标或结构 smoke test 当作论文结果。

### 摘要

本实验比较同一 293.49M 总参数、约 124M token-active 参数的 top-2 MoE Transformer 在
`single`（每人一篇）与 `multi5_permute`（每人五篇独立措辞及字段排列）语料上的训练。
两者均使用 4 卡 FSDP、BF16、global batch 448，并各处理约 4B scheduled tokens。`single`
总 loss 从 10.9464 降至 0.1932；`multi5_permute` 从 10.9489 降至 0.2962。随后删除每篇
**训练语料原文**中的六个属性 span，按原文顺序 progressive greedy 填回。`single` 达到
600,000/600,000 严格字段匹配；`multi5_permute` 达到 2,999,746/3,000,000（99.9915%），
其中 499,813/500,000 篇六字段全对。两次运行均证明优化与训练分布 recall 有效，但该
训练集评测不证明对未见人物、属性组合或模板的泛化。

### 1. 方法

#### 1.1 数据

两项正式条件均以 seed 1337 生成相同的 100,000 个合成人物。`single` 每人一篇 biography，
共 100,000 篇、7,405,102 tokens；`multi5_permute` 每人五篇独立渲染且随机排列字段，
共 500,000 篇、37,046,556 tokens。每篇文本包含 `birth_date`、`birth_city`、
`university`、`major`、`company`、`company_city` 六个带精确字符 span 的属性。这里的“原始
bio”指训练时实际使用的生成语料原文，不是现实人物数据。两项条件的 `profiles.jsonl`
逐字节相同，因此人物事实保持不变，增强只改变表达模板和字段顺序。

manifest 中的 50%/50% person split 是为 P/Q probe train/validation 预留的元数据；本次
progressive-cloze 结果覆盖训练过的全部 100,000 人，不是 held-out validation。原始大文件
保存在服务器 `/data/mini-train-sys/artifacts/synbios_moe/{single,multi5_permute}/`，Git
中保存 [single manifest](results/datasets/synbios_moe/single/manifest.json) 与
[multi5 manifest](results/datasets/synbios_moe/multi5_permute/manifest.json) 及 SHA256，
不复制大体积数据 payload。

#### 1.2 模型

| 项目 | 实验条件 |
|---|---|
| 架构 | decoder-only Transformer，RoPE，tied input/output embedding |
| 总参数 | 293,494,272 |
| 近似 token-active 参数 | 约 124M；MoE 每 token 激活 top-2 experts |
| 层数 / hidden / heads | 12 / 768 / 12 |
| 序列长度 | 512 |
| FFN | 8 experts，top-2，SwiGLU，每 expert intermediate size 1,024 |
| Router | dropless；aux coefficient 0.01；z-loss coefficient 0.001 |
| Dropout | 0.1 |
| 完整配置 | [`configs/synbios_moe/model.yaml`](configs/synbios_moe/model.yaml) |

#### 1.3 优化与系统条件

| 项目 | 实验条件 |
|---|---|
| 并行 | 4-GPU FSDP full shard；13 个 Transformer wrap units |
| GPU / 软件 | 4 × RTX 4090 24 GB；PyTorch 2.5.1+cu118；CUDA 11.8；Triton 3.1.0 |
| 精度 | BF16，无 GradScaler |
| Batch | local 112/GPU；global 448；229,376 tokens/step；无梯度累计 |
| Optimizer / clipping | AdamW；global grad-norm clip 5.0 |
| LR | peak `1e-3`；1,000-step warmup；cosine decay；floor `1e-4` |
| 训练长度 | `single`: 540 epochs / 17,280 steps / 3,963,617,280 tokens；`multi5`: 108 / 17,388 / 3,988,389,888 |
| 数据顺序 | randomized-document packing；shuffle window 1,024；每 rank 4 workers |
| Checkpoint | 原子保存 DCP+Adam；两项均保留 safety、倒数第二个和 final checkpoint |
| 正式运行配置 | [`single_fsdp_4gpu.yaml`](configs/synbios_moe/runs/single_fsdp_4gpu.yaml)；[`multi5_permute_fsdp_4gpu.yaml`](configs/synbios_moe/runs/multi5_permute_fsdp_4gpu.yaml) |

batch 112 来自完整 forward/loss/backward/AdamW 容量测试，而非只测 forward。batch 120
虽能运行，但显存达到 92.02% 且吞吐下降；batch 112 在 55 个连续完整 step 中保持
86.20% 峰值显存和 368,170 tok/s，因此用于正式训练。

#### 1.4 验证协议

Progressive cloze 直接读取训练 `biographies.jsonl` 中的原始 BPE 序列和属性 span：

1. 删除六个事实各自的连续 token span，保留所有非事实 token；
2. 按字段在原文中的出现顺序 greedy decode，最多 16 tokens/field；
3. 把较早预测插回文本，再预测下一个字段；
4. 以区分大小写的完整字符串相等作为主指标；
5. 另报 case-fold、空格归一化后的 normalized Levenshtein similarity，但模糊指标不参与
   严格正确率或 P/Q 门禁放行。

`single` 100k 与 `multi5` 500k 验证均切为四个连续、不重叠分片；聚合器在汇总前拒绝
overlap/gap。详细协议、逐步填空实例和模糊匹配的高估风险见
[`single 报告`](reports/synbios_single_cloze_100k.md) 与
[`multi5 报告`](reports/synbios_multi5_permute_cloze_500k.md)。

### 2. 结果

#### 2.1 正式训练

| 指标 | 结果 |
|---|---:|
| 完成进度 | 540/540 epochs；17,280/17,280 steps |
| 初始 / 最终 total loss | 10.946440 / 0.193221 |
| 最小记录 total loss | 0.192083 |
| 最终 LM cross-entropy | 0.183091 |
| 最终 MoE regularization | 0.010130 |
| 初始 / 最终 grad norm | 16.9637 / 0.02456 |
| 记录点中发生 clipping | 6 / 4,321 |
| 最终 dead experts / dropped routes | 0 / 0 |
| 最终 expert-load CV | 0.01539 |
| 端到端耗时 | 12,668.67 s（约 3 h 31 min） |
| 端到端平均吞吐 | 312,868 tok/s |
| 记录区间平均 GPU compute utilization | 97.02% |
| 峰值 allocated memory / GPU | 86.20% |
| Final checkpoint | epoch 540 / step 17,280；5,006,353,542 bytes |

原始逐 step 证据同时以
[JSONL](results/formal_runs/synbios_moe/runs/synbios_moe_single_fsdp_4gpu/20260721-045620/events.jsonl)
和 [TensorBoard event](results/formal_runs/synbios_moe/runs/synbios_moe_single_fsdp_4gpu/20260721-045620/)
保存。最终 checkpoint 的 DCP/Adam 和 1.33 GB `model.pt` 留在服务器；Git 仅保存
`COMMITTED`、DCP `.metadata`、runtime、RNG 和配置等小型恢复证据。

`multi5_permute` 在相同硬件、模型、batch 与 LR 配置下完成 108 epochs / 17,388 steps。
其主要训练结果与 token-budget 匹配的 `single` 对照如下：

| 指标 | `single` | `multi5_permute` |
|---|---:|---:|
| Biographies / people | 100,000 / 100,000 | 500,000 / 100,000 |
| Scheduled tokens | 3,963,617,280 | 3,988,389,888 |
| 初始 / 最终 total loss | 10.946440 / 0.193221 | 10.948931 / 0.296150 |
| 最小记录 total loss | 0.192083 | 0.293688 |
| 最终 LM cross-entropy | 0.183091 | 0.285855 |
| 最终 grad norm | 0.02456 | 0.06513 |
| 最终 dead experts / dropped routes | 0 / 0 | 0 / 0 |
| 端到端耗时 | 12,668.67 s | 12,148.31 s |
| 端到端平均吞吐 | 312,868 tok/s | 328,308 tok/s |
| Final checkpoint | epoch 540 / step 17,280 | epoch 108 / step 17,388 |

`multi5` 的 JSONL/TensorBoard 原始证据位于
[`20260721-144408`](results/formal_runs/synbios_moe/runs/synbios_moe_multi5_permute_fsdp_4gpu/20260721-144408/)；
final checkpoint 为 epoch 108 / step 17,388，共 5,006,353,550 bytes。

#### 2.2 原文 progressive-cloze 验证

| 指标 | birth date | birth city | university | major | company | company city | Micro |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strict exact accuracy | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| Mean character similarity | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

| 汇总指标 | 结果 |
|---|---:|
| Biographies / fields | 100,000 / 600,000 |
| Biography 6/6 exact accuracy | 100% |
| Fuzzy accuracy @ 0.50 / 0.80 / 0.90 | 100% / 100% / 100% |
| Unterminated fields | 0 |
| 4-GPU parallel wall time | 419.48 s |
| Aggregate throughput | 238.39 biographies/s |

机器可读汇总在
[`summary.json`](results/formal_runs/synbios_moe/results/single_cloze_eval/full_100k/summary.json)。
由于 strict exact 已经是 100%，这里的 fuzzy 结果没有抬高结论。

`multi5_permute` 的相同协议全量结果为：

| 指标 | birth date | birth city | university | major | company | company city | Micro |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strict exact accuracy | 99.9968% | 99.9966% | 99.9952% | 99.9710% | 99.9946% | 99.9950% | 99.9915% |
| Exact count / 500k | 499,984 | 499,983 | 499,976 | 499,855 | 499,973 | 499,975 | 2,999,746 / 3M |

| `multi5` 汇总指标 | 结果 |
|---|---:|
| Biographies / fields | 500,000 / 3,000,000 |
| Biography 6/6 exact accuracy | 499,813 / 500,000（99.9626%） |
| Mean character similarity | 99.994013% |
| Fuzzy accuracy @ 0.50 / 0.80 / 0.90 | 99.993433% / 99.991733% / 99.991633% |
| Unterminated fields | 0 |
| 4-GPU parallel wall time | 2,450.64 s |
| Aggregate throughput | 204.03 biographies/s |

机器可读汇总见
[`multi5 summary.json`](results/formal_runs/synbios_moe/results/multi5_permute_cloze_eval/full_500k/summary.json)。
0.90 fuzzy 比 strict 多计 3 个字段，所以论文主结果坚持 strict exact；`major` 的 145 个
不完全匹配占全部 254 个错误的 57.1%，是当前最弱字段。

#### 2.3 训练前工程验证与失败对照

| Run | 条件 | 结果 | 结论 |
|---|---|---|---|
| Save/resume validation | 4-GPU FSDP，step 1–3 保存后恢复到 step 4–5 | LR 连续为 `6.542e-5 → 8.723e-5 → 1.090e-4`；模型、Adam、scheduler、计数器和 RNG 均恢复 | checkpoint contract 通过 |
| 64-step preflight | 正式 peak LR `1e-3`，global batch 448 | loss `10.9464 → 3.6881`；final grad norm 1.438；无 NaN/Inf | 正式配置通过 |
| Rejected LR run | 线性 batch scaling，peak LR `4.667e-3` | 30 epochs/960 steps 后停止；loss 最低 1.0981 后升至 5.3797；final grad norm 1,411.65 | 发散，不能作为成功结果 |

失败 run 的事件流仍保存在
[`20260721-042956`](results/formal_runs/synbios_moe/runs/synbios_moe_single_fsdp_4gpu/20260721-042956/)，
用于说明为什么正式运行取消 LR 的线性 batch scaling。

#### 2.4 系统性能结果

| FSDP local batch | Global batch | Peak allocated | Throughput | 状态 |
|---:|---:|---:|---:|---|
| 96 | 384 | 74.56% | 365,034 tok/s | pass |
| 112 | 448 | 86.20% | 370,857 tok/s | pass；正式选择 |
| 120 | 480 | 92.02% | 281,926 tok/s | pass；较慢且余量低 |
| 128 | 512 | — | — | 预期 backward OOM boundary |

FSDP weak scaling 在 local batch 64 下由 1 GPU 的平均 93,302 tok/s 提升到 4 GPU 的
344,254 tok/s，效率 92.24%（两次 4-GPU repeat 为 92.08% 与 92.40%）。完整 raw JSON、
容量 sweep、通用 125M-class single/DDP/FSDP matrix 和图表位于
[`results/benchmarks/`](results/benchmarks/)；简表见
[`results/BENCHMARK_SUMMARY.md`](results/BENCHMARK_SUMMARY.md)。

### 3. 实验状态与结论边界

| 实验 | 当前状态 | 可下结论 |
|---|---|---|
| `single` 4-GPU FSDP pretraining | 已完成 | 优化稳定收敛 |
| `single` 全训练集 progressive cloze | 已完成 | 对原始训练语料精确 recall |
| `multi5_permute` 4-GPU FSDP pretraining | 已完成 | 优化稳定收敛 |
| `multi5_permute` 全训练集 progressive cloze | 已完成 | 99.9915% strict 字段 recall；近乎完全但非 100% |
| 新 progressive-cloze P/Q gate | 已实现；当前 checkpoint 的全量结果已超过 0.90 门槛 | 门禁方法可用 |
| 正式 P/Q probe 与 held-out person validation | **尚未运行** | 暂无机制/泛化结论 |
| Router attribute specialization analysis | **尚未运行** | 暂无 expert 专门化结论 |
| `single` vs. `multi5_permute` held-out 比较 | **尚未运行** | 暂无数据增强泛化收益结论 |

因此，当前最强结论是“两种训练均有效；`single` 达到训练集原文完全 recall，`multi5`
达到 99.9915% strict recall”，不是“对新人物达到约 100%”。`multi5` 在这个训练集指标上
没有超过 `single`；必须完成 person-level held-out P/Q validation 和 router analysis，才能
回答数据增强是否改善泛化或改变内部知识组织。

### 4. 结果、复现与 Git 存档

- 精确实验时间线、命令、失败重试与修复：[`HISTORY.md`](HISTORY.md)
- 训练/验证/benchmark 总结：[`results/BENCHMARK_SUMMARY.md`](results/BENCHMARK_SUMMARY.md)
- Cloze 结论与实例：[`reports/synbios_single_cloze_100k.md`](reports/synbios_single_cloze_100k.md)
- Multi5 cloze 结论与实例：[`reports/synbios_multi5_permute_cloze_500k.md`](reports/synbios_multi5_permute_cloze_500k.md)
- Git-safe 原始证据：[`results/`](results/)
- 内容校验：[`results/MANIFEST.sha256`](results/MANIFEST.sha256)

`scripts/bash/export_test_results.sh` 将服务器上的 JSON/JSONL/CSV、TensorBoard events、
控制台日志、测试/JUnit、数据 manifest 和 checkpoint 小型元数据导出到 `results/` 并生成
SHA256。原始数据、`model.pt`、DCP/Adam shards、cache 和 trash 因体积或敏感性不推送；
它们的位置、manifest/hash、DCP `.metadata` 和 `COMMITTED` 证据保留在仓库中。

## 建议阅读顺序

第一次阅读不要从 kernel 开始。按下面顺序走一遍，能先建立完整心智模型：

1. [项目全流程导读](docs/guides/project_walkthrough.md)：一次 batch 如何穿过整个系统。
2. [代码架构](docs/guides/architecture.md)：每个目录负责什么，模块边界为什么这样划分。
3. [数据流水线](docs/data/data_pipeline.md)：Document、chunk、token shard、Dataset、Sampler。
4. [配置说明](configs/README.md)：YAML `extends`、batch、worker、checkpoint 等参数。
5. [单卡/DDP/FSDP 手册](docs/training/distributed_training.md)：训练与恢复的真实执行方式。
6. [监控与恢复](docs/training/monitoring_and_recovery.md)：ETA、显存、梯度和 checkpoint。
7. [SynBioS 全流程](docs/experiments/synbios_moe_training_flow.md)：数据生成、主训练、probe、结果。
8. [Kernel 与 benchmark 路线](docs/README.md#算子与性能)：进入 Triton/CUDA 深入材料。

所有文档的分类和“当前手册/历史设计记录”标记见 [文档地图](docs/README.md)。

## 安装

需要 Python 3.10+。Linux/CUDA 服务器推荐：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[triton,data,dev]"
```

Windows：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[data,dev]"
```

Linux/NVIDIA 实验服务器可以从干净 checkout 一键创建 `.venv`、安装已验证的
PyTorch/CUDA/Triton 组合并执行环境预检。先把实验产物和编译缓存映射到挂载盘，
再安装环境：

```bash
bash scripts/bash/setup_storage.sh /data
bash scripts/bash/setup_server.sh
```

系统盘/挂载盘分工、已有数据迁移、系统依赖、GPU/NCCL 验收、smoke 和正式实验前清单见
[`docs/guides/server_setup.md`](docs/guides/server_setup.md)。

开发 CUDA C++ 扩展时再安装 `.[cuda]`。训练 notebook 必须使用与命令行相同的
Python 环境，否则容易出现 Jupyter 能 import、`!python` 却找不到依赖的问题。

## 先跑最小闭环

CPU/debug 配置验证模型、优化器、日志和 checkpoint 调用链：

```bash
python scripts/train.py \
  --config configs/train_debug.yaml \
  --model-config configs/model_debug_dense.yaml \
  --device cpu
```

SynBioS 的可执行 notebook 会依次跑数据准备、主训练、评估、P/Q probe 和 router：

```bash
jupyter notebook tests/synbios_moe_end_to_end.ipynb
```

它默认只运行 100 人、2 个主训练 step 和 3 个 probe step，是结构测试，不是论文结果。

## 正式训练入口

模型配置与运行配置分开传入：模型配置决定层数、宽度、Dense/MoE；运行配置决定
数据、优化器、精度、并行方式、epoch 和 checkpoint。

```bash
python scripts/train.py \
  --device cuda \
  --config configs/server/rtx4090_24gb/runs/single_1gpu.yaml \
  --model-config configs/model_125m_moe.yaml
```

单机 4/8 卡服务器：

```bash
NPROC=4 MODEL_CONFIG=configs/model_125m_moe.yaml bash scripts/bash/distributed.sh ddp
NPROC=8 MODEL_CONFIG=configs/model_125m_moe.yaml bash scripts/bash/distributed.sh ddp
NPROC=4 MODEL_CONFIG=configs/model_125m_moe.yaml bash scripts/bash/distributed.sh fsdp
NPROC=8 MODEL_CONFIG=configs/model_125m_moe.yaml bash scripts/bash/distributed.sh fsdp
```

固定拓扑配置会校验 `WORLD_SIZE`，避免 4 卡配置误跑成 8 卡。当前没有梯度累计：

```text
global_batch = train.batch_size（每 rank）× WORLD_SIZE
```

## 数据和 DataLoader

通用预处理入口是 `scripts/prepare_data.py`，SynBioS 使用
`scripts/synbios_moe.py prepare`。训练读取 manifest 和 mmap token shards，不把整个
语料加载进 GPU。

多卡时所有 rank 根据同一确定性顺序取得互不重叠的数据。`data.num_workers: null`
表示按单机 CPU 预算自动分配：默认节点总预算 32、每 rank 最多 4，并为训练进程
保留 CPU 核。详细设计见 [数据流水线](docs/data/data_pipeline.md)。

## Checkpoint 与恢复

每个有效 checkpoint 是带 `COMMITTED` 标记的目录：

```text
checkpoints/<run>/epoch_..._step_.../
├── distributed/       # DCP 模型和 Adam 分片
├── runtime.pt         # scheduler/scaler/计数器/resolved config
├── rng_rank_*.pt      # 每 rank RNG
├── model.pt           # 可选，只供 evaluate/probe
└── COMMITTED
```

恢复完整训练状态：

```bash
python scripts/train.py --device cuda \
  --config <run.yaml> --model-config <model.yaml> --resume latest
```

Probe 只读取 `model.pt`，不会加载主训练 Adam。DCP、跨 world-size 恢复和 epoch 语义见
[分布式训练手册](docs/training/distributed_training.md)。

## 测试与 benchmark

```bash
python -m pytest -q
python -m ruff check minitrain scripts tests
```

主要 notebook：

| Notebook | 用途 |
|---|---|
| `tests/example_training.ipynb` | 模型、LR、checkpoint 小实验 |
| `tests/synbios_moe_end_to_end.ipynb` | SynBioS 端到端结构测试 |
| `tests/operator_bench.ipynb` | 通用算子正确性与性能 |
| `tests/moe_operator_bench.ipynb` | Router 与 fused MoE |
| `tests/distributed_server_benchmark.ipynb` | 1/4/8 卡 DDP/FSDP 弱扩展和显存容量 |

分布式 benchmark 不只报 tokens/s，还记录 step P50/P95、数据等待、单卡和全系统显存、
OOM 边界、硬件拓扑及 Git 状态。协议见 [分布式 benchmark](docs/benchmarks/distributed_benchmark.md)。

## 目录地图

```text
configs/       模型、数据、策略、硬件拓扑和具体 run
docs/
  guides/      项目导读、架构和阅读导航
  data/        数据流水线
  training/    训练、分布式、监控和恢复
  experiments/ SynBioS 流程与 fidelity
  benchmarks/  性能规范和硬件容量
  model/       MoE 等模型专题
  kernels/     CUDA/Triton 深入材料
  design-notes/ 历史设计记录
  references/  外部参考和实现映射
experiments/   SynBioS 的数据、probe、评估与 router 分析
minitrain/
  data/        文档读取、tokenizer、预处理、Dataset/Sampler/DataLoader
  model/       Transformer、Dense/MoE block、router、算子协议
  kernels/     PyTorch 基线、Triton、CUDA C++ 扩展
  distributed/ single、DDP、FSDP 策略
  runtime/     typed config、factory、device、logger、batch scaling
  train/       单 step Trainer、epoch Runner、AdamW、LR、DCP
scripts/       用户入口；Python/Bash/PowerShell 分开
tests/         pytest、教学 notebook 和 benchmark notebook
reports/       选定 benchmark 报告与图表
```

`scripts/eval.py` 和 `scripts/sample.py` 目前仍是通用占位入口；可用的实验评估入口是
`scripts/synbios_moe.py evaluate/probe/analyze`。这一区分很重要，避免把 scaffold 当成
已经完成的功能。

## 清理生成文件

清理脚本只处理声明过的临时/构建产物；checkpoint 和昂贵 benchmark 结果需要确认后
自行归档：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/powershell/clean.ps1
```

```bash
bash scripts/bash/clean.sh
```

训练与 SynBioS 后训练的指标、TensorBoard、checkpoint 内容和跨卡恢复方式见
[`docs/training/monitoring_and_recovery.md`](docs/training/monitoring_and_recovery.md)。

## License

项目代码使用 MIT License；第三方 kernel 代码的许可证随对应包保存。
