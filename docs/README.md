# MiniTrainSys 文档地图

文档按“先理解系统，再深入专题”的顺序组织。标记含义：

服务器首次安装、CUDA/NCCL 预检、smoke、benchmark、正式预训练和 Probe 的完整命令见
[`guides/server_setup.md`](guides/server_setup.md)。

- **当前手册**：描述现有代码，可以直接照着运行。
- **实验说明**：描述某个实验的控制变量、产物和判断方式。
- **深入材料**：阅读 kernel 或性能问题时使用，不是入门前置知识。
- **历史设计记录**：解释实现为什么演变成现在这样；命令和 TODO 可能已经过期。

## 目录结构

```text
docs/
├── guides/         项目导读、架构和分类阅读指南
├── data/           数据预处理与 DataLoader 当前手册
├── training/       训练、分布式、监控、恢复和混合精度
├── experiments/    SynBioS 训练流程与论文 fidelity
├── benchmarks/     算子/分布式 benchmark 规范和硬件容量记录
├── model/          MoE 等模型专题
├── kernels/        CUDA/Triton 算子阅读、实现与排错
├── design-notes/   历史设计方案，不作为当前运行手册
└── references/     外部参考和实现映射
```

想按问题快速查找时，直接打开
[分类与阅读指南](guides/reading_guide.md)。

## 最短学习路线

如果目标是读懂整个项目，依次阅读：

1. [项目全流程导读](guides/project_walkthrough.md) — 从命令行到一次 optimizer update。
2. [架构与代码边界](guides/architecture.md) — 目录、依赖方向和扩展点。
3. [数据流水线](data/data_pipeline.md) — 文档、分词、shard、采样和预取。
4. [配置参考](../configs/README.md) — 配置继承与每组参数的实际语义。
5. [Single/DDP/FSDP 手册](training/distributed_training.md) — 并行、epoch、DCP 恢复。
6. [监控与恢复](training/monitoring_and_recovery.md) — ETA、显存、梯度和 checkpoint。
7. [SynBioS 全流程](experiments/synbios_moe_training_flow.md) — 完整实验实例。

读完前五篇后，再看任意 Python 文件都应能知道它属于哪一层、由谁调用。

按目录浏览代码时，可使用模块内短 README：

- [data](../minitrain/data/README.md)
- [model](../minitrain/model/README.md)
- [kernels](../minitrain/kernels/README.md)
- [distributed](../minitrain/distributed/README.md)
- [runtime](../minitrain/runtime/README.md)
- [train](../minitrain/train/README.md)
- [scripts](../scripts/README.md)
- [tests](../tests/README.md)
- [reports](../reports/README.md)

## 当前手册

| 文档 | 回答的问题 |
|---|---|
| [project_walkthrough.md](guides/project_walkthrough.md) | 一次训练从入口到落盘经过哪些对象？ |
| [architecture.md](guides/architecture.md) | 模块边界、依赖方向、核心抽象是什么？ |
| [data_pipeline.md](data/data_pipeline.md) | `inputs`、Document、chunk、token shard、DataLoader 如何连接？ |
| [distributed_training.md](training/distributed_training.md) | DDP/FSDP 如何分工，怎样保存和恢复？ |
| [monitoring_and_recovery.md](training/monitoring_and_recovery.md) | loss、吞吐、单卡/多卡显存怎样看，怎样完整恢复？ |
| [synbios_moe_probe_pipeline.md](experiments/synbios_moe_probe_pipeline.md) | P/Q probe 怎样分阶段、多卡调度、监控和汇总？ |
| [mixed_precision_plan.md](training/mixed_precision_plan.md) | FP32/BF16/FP16 各自保存和计算什么？ |
| [moe.md](model/moe.md) | Top-k router、expert 和 fused MoE 的数据流是什么？ |
| [distributed_benchmark.md](benchmarks/distributed_benchmark.md) | 如何正确测 1/4/8 卡弱扩展和显存容量？ |
| [cuda_ext_run_commands.md](kernels/cuda_ext_run_commands.md) | CUDA 扩展如何编译、验证和排错？ |

配置字段以 [configs/README.md](../configs/README.md) 为准；notebook 入口以
[tests/README.md](../tests/README.md) 为准。

## SynBioS 实验说明

| 文档 | 定位 |
|---|---|
| [synbios_moe_training_flow.md](experiments/synbios_moe_training_flow.md) | 数据生成到 probe 的当前可执行流程 |
| [synbios_moe_reproduction_plan.md](experiments/synbios_moe_reproduction_plan.md) | 与 Allen-Zhu 论文的 fidelity 差异清单 |
| [synbios_moe_probe_pipeline.md](experiments/synbios_moe_probe_pipeline.md) | Probe 缓存、阶段门控、任意GPU数调度、validation与结果汇总 |
| [experiments/synbios_moe/README.md](../experiments/synbios_moe/README.md) | 最短运行命令和产物路径 |

## 算子与性能

先读 [minitrain/kernels/README.md](../minitrain/kernels/README.md) 理解 backend/fallback，
再按需求选择：

- [benchmark_plan.md](benchmarks/benchmark_plan.md)：所有算子 benchmark 的共同约定。
- [cuda_flash_attention_learning_report.md](kernels/cuda_flash_attention_learning_report.md)：
  本项目 CUDA FlashAttention 的总体设计与验证结果。
- [cuda_flash_attention_code_reading_guide.md](kernels/cuda_flash_attention_code_reading_guide.md)：
  Python → C++ binding → CUDA dispatch 的调用链。
- [flash_fwd_kernel_deep_dive.md](kernels/flash_fwd_kernel_deep_dive.md)：forward kernel 逐层解析。
- [flash_bwd_kernel_deep_dive.md](kernels/flash_bwd_kernel_deep_dive.md)：backward kernel 逐层解析。
- [cuda_flash_attention_sm86_spill_analysis.md](kernels/cuda_flash_attention_sm86_spill_analysis.md)：
  RTX 3050/sm86 上的寄存器 spill 个案。
- [sm86_4gb_benchmark_capacity.md](benchmarks/sm86_4gb_benchmark_capacity.md)：4 GiB 本地开发机容量边界。
- [minitrain/kernels/triton/moe_router_scatter.md](../minitrain/kernels/triton/moe_router_scatter.md)：
  router scatter kernel 可视化说明。

RTX 3050 文档是历史硬件个案，不能替代 24 GB RTX 4090 的服务器 benchmark。

## 历史设计记录

以下文件保留决策过程，但不作为当前运行手册：

- [data_preprocessing_plan.md](design-notes/data_preprocessing_plan.md)
- [flash_attention_pretraining_plan.md](design-notes/flash_attention_pretraining_plan.md)
- [subsession_plan.md](design-notes/subsession_plan.md)

## 参考资料

- [reference_map.md](references/reference_map.md)
- [references.md](references/references.md)

遇到 plan 与代码不一致时，以当前 typed config、测试和“当前手册”为准。
