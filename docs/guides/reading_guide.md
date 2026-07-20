# 文档分类与阅读指南

这份文件只负责给 `docs/` 分类，不替代各专题文档，也不表示必须从头读到尾。

## 一、项目主线：先建立完整心智模型

适合第一次理解项目，建议按顺序阅读：

| 顺序 | 文档 | 读完应该明白什么 |
|---:|---|---|
| 1 | [project_walkthrough.md](project_walkthrough.md) | 一条命令怎样经过配置、数据、模型、Trainer、Runner 和 checkpoint |
| 2 | [architecture.md](architecture.md) | 各目录的职责、依赖方向以及功能应该加在哪一层 |
| 3 | [data_pipeline.md](../data/data_pipeline.md) | Document、chunk、token shard、Dataset、Sampler、DataLoader 的边界 |
| 4 | [distributed_training.md](../training/distributed_training.md) | single、DDP、FSDP 如何执行以及怎样恢复训练 |
| 5 | [monitoring_and_recovery.md](../training/monitoring_and_recovery.md) | 终端/TensorBoard 指标、梯度范数、checkpoint 内容和恢复命令 |

第一遍只需要理解调用关系，不必进入 CUDA kernel 实现。

## 二、数据与预处理

| 文档 | 类型 | 用途 |
|---|---|---|
| [data_pipeline.md](../data/data_pipeline.md) | 当前手册 | 当前代码从输入文件到训练 batch 的完整流程 |
| [data_preprocessing_plan.md](../design-notes/data_preprocessing_plan.md) | 设计记录 | 预处理功能当初的拆分理由和演进过程 |

遇到 `resolve_inputs`、DocumentCleaner、`max_document_chars`、chunk、shard、worker、
pin memory 或 non-blocking 问题，优先读 `data_pipeline.md`。只有想理解历史取舍时才读
`data_preprocessing_plan.md`。

## 三、训练运行时与数值稳定性

| 文档 | 类型 | 用途 |
|---|---|---|
| [monitoring_and_recovery.md](../training/monitoring_and_recovery.md) | 当前手册 | loss、LR、token/s、ETA、显存、梯度范数和恢复训练 |
| [mixed_precision_plan.md](../training/mixed_precision_plan.md) | 当前实现说明 | FP32/BF16/FP16、autocast、GradScaler、FSDP dtype |
| [architecture.md](architecture.md) | 当前手册 | Trainer、TrainingRunner、EventLogger、ProgressReporter 的边界 |
| [subsession_plan.md](../design-notes/subsession_plan.md) | 历史记录 | 早期训练子阶段规划，不作为当前启动说明 |

虽然 `mixed_precision_plan.md` 文件名保留了 `plan`，其中相关章节已经用于解释当前实现；
真正的运行参数仍以 typed config 和当前代码为准。

## 四、分布式训练与服务器性能

建议按照“先正确，再测性能”的顺序：

1. [distributed_training.md](../training/distributed_training.md)：理解 DDP/FSDP、world size、global batch、DCP。
2. [distributed_benchmark.md](../benchmarks/distributed_benchmark.md)：理解 1/4/8 卡 benchmark 的矩阵和验收指标。
3. [benchmark_plan.md](../benchmarks/benchmark_plan.md)：需要比较算子或训练结果时，再看统一测量规范。

硬件个案：

- [sm86_4gb_benchmark_capacity.md](../benchmarks/sm86_4gb_benchmark_capacity.md)：本地 4 GiB RTX 3050 容量边界。
- [cuda_flash_attention_sm86_spill_analysis.md](../kernels/cuda_flash_attention_sm86_spill_analysis.md)：sm86 寄存器 spill 分析。

这两篇不能代替服务器 24 GiB RTX 4090 的实测结果。

## 五、SynBioS / Allen-Zhu 实验

| 顺序 | 文档 | 作用 |
|---:|---|---|
| 1 | [synbios_moe_training_flow.md](../experiments/synbios_moe_training_flow.md) | 当前代码怎样生成数据、训练、evaluate、P/Q probe 和 router 结果 |
| 2 | [synbios_moe_reproduction_plan.md](../experiments/synbios_moe_reproduction_plan.md) | 当前实现与原论文 fidelity 的一致项、差异和预算换算 |
| 3 | [monitoring_and_recovery.md](../training/monitoring_and_recovery.md) | 怎样观察正式实验、保存 checkpoint 和中断恢复 |

阅读时要区分两件事：

- `training_flow` 回答“当前项目实际上怎么运行”；
- `reproduction_plan` 回答“它与论文协议有多一致”。

判断实验结果时按下面顺序：主训练 loss → attribute-token accuracy → Q-probe → P-probe →
router。不要在主模型尚未学会 biography 时直接解释 probe 或 expert specialization。

## 六、MoE 与算子实现

### MoE

- [moe.md](../model/moe.md)：先理解 router、top-k、expert、aux loss 和 fused MoE 数据流。

### CUDA FlashAttention 阅读顺序

1. [cuda_flash_attention_learning_report.md](../kernels/cuda_flash_attention_learning_report.md)：总体设计和验证结论。
2. [cuda_flash_attention_code_reading_guide.md](../kernels/cuda_flash_attention_code_reading_guide.md)：Python 到 CUDA 的调用链。
3. [flash_fwd_kernel_deep_dive.md](../kernels/flash_fwd_kernel_deep_dive.md)：forward kernel。
4. [flash_bwd_kernel_deep_dive.md](../kernels/flash_bwd_kernel_deep_dive.md)：backward kernel。
5. [cuda_ext_run_commands.md](../kernels/cuda_ext_run_commands.md)：实际编译、运行和排错。

[flash_attention_pretraining_plan.md](../design-notes/flash_attention_pretraining_plan.md) 是历史设计记录，不应取代
当前代码阅读指南。

## 七、参考与历史材料

以下文件用于查来源或理解决策演进，不应作为当前运行命令：

- [reference_map.md](../references/reference_map.md)：实现与外部参考的对应关系。
- [references.md](../references/references.md)：参考资料清单。
- [data_preprocessing_plan.md](../design-notes/data_preprocessing_plan.md)：数据预处理历史方案。
- [flash_attention_pretraining_plan.md](../design-notes/flash_attention_pretraining_plan.md)：FlashAttention 接入历史方案。
- [subsession_plan.md](../design-notes/subsession_plan.md)：早期训练阶段规划。

如果历史 plan 与代码不一致，以当前代码、typed config、pytest 和当前手册为准。

## 八、这轮重点更新的文档怎么读

如果你只想理解最近完成的训练架构重构，按下面顺序即可：

1. [architecture.md](architecture.md)：先看 `Trainer/TrainingRunner`、DCP、
   `EventLogger/ProgressReporter`，知道新功能分别落在哪一层。
2. [monitoring_and_recovery.md](../training/monitoring_and_recovery.md)：重点看第 1～4 节，理解终端字段、
   TensorBoard、多卡显存、checkpoint 目录和 `--resume`。
3. [distributed_training.md](../training/distributed_training.md)：重点看 DDP/FSDP 与 world-size 改变后的恢复。
4. [synbios_moe_training_flow.md](../experiments/synbios_moe_training_flow.md)：看实验如何复用新监控与 checkpoint，
   以及 evaluate/probe 为什么只加载 `model.pt`。
5. [distributed_benchmark.md](../benchmarks/distributed_benchmark.md)：最后看服务器 1/4/8 卡怎样验收吞吐和显存。

这条路线不要求先读 kernel 文档。等主训练、恢复和实验流程完全清楚后，再进入 MoE 或
FlashAttention 深入材料。

## 九、按问题快速查找

| 当前问题 | 直接阅读 |
|---|---|
| 看不懂整个训练调用链 | [project_walkthrough.md](project_walkthrough.md) |
| 不知道代码应该改在哪一层 | [architecture.md](architecture.md) |
| 不懂 chunk、shard、Sampler、worker | [data_pipeline.md](../data/data_pipeline.md) |
| 不懂 DDP/FSDP 或多卡数据划分 | [distributed_training.md](../training/distributed_training.md) |
| 不懂 ETA、显存、梯度范数、TensorBoard | [monitoring_and_recovery.md](../training/monitoring_and_recovery.md) |
| 不懂多卡 checkpoint 如何给单卡 probe | [monitoring_and_recovery.md](../training/monitoring_and_recovery.md) |
| 想运行 SynBioS 全流程 | [synbios_moe_training_flow.md](../experiments/synbios_moe_training_flow.md) |
| 想检查 Allen-Zhu fidelity | [synbios_moe_reproduction_plan.md](../experiments/synbios_moe_reproduction_plan.md) |
| 想测 1/4/8 卡扩展效率 | [distributed_benchmark.md](../benchmarks/distributed_benchmark.md) |
| 想读 MoE router/expert | [moe.md](../model/moe.md) |
| 想读 CUDA FlashAttention | [cuda_flash_attention_code_reading_guide.md](../kernels/cuda_flash_attention_code_reading_guide.md) |
