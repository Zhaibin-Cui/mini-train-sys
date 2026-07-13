# 文档索引

## 当前实现

- [`architecture.md`](architecture.md)：MiniTrain 总体模块边界。
- [`benchmark_plan.md`](benchmark_plan.md)：统一算子 benchmark 约定。
- [`cuda_flash_attention_learning_report.md`](cuda_flash_attention_learning_report.md)：CUDA FlashAttention 的设计、实现逻辑与验证结论。
- [`cuda_flash_attention_code_reading_guide.md`](cuda_flash_attention_code_reading_guide.md)：按调用链阅读 Python、C++ 和 CUDA 源码。
- [`flash_fwd_kernel_deep_dive.md`](flash_fwd_kernel_deep_dive.md)：逐段讲解 `flash_fwd_kernel.h` 的结构、在线 softmax、CuTe 数据布局、普通前向与 Split-KV 路径。
- [`flash_bwd_kernel_deep_dive.md`](flash_bwd_kernel_deep_dive.md)：逐段讲解 `flash_bwd_kernel.h` 的梯度公式、概率重算、dQ/dK/dV 分工、seq-k 并行与后处理流水。
- [`cuda_ext_run_commands.md`](cuda_ext_run_commands.md)：Windows 本机与 Linux 服务器的编译、验证和训练命令。
- [`cuda_flash_attention_sm86_spill_analysis.md`](cuda_flash_attention_sm86_spill_analysis.md)：sm86 spill、Nsight 数据和硬件边界。
- [`mixed_precision_plan.md`](mixed_precision_plan.md)：混合精度约定与实施状态。

## 历史计划与参考

- [`flash_attention_pretraining_plan.md`](flash_attention_pretraining_plan.md)：Triton FlashAttention 前期设计记录。
- [`subsession_plan.md`](subsession_plan.md)：早期分阶段实施计划。
- [`reference_map.md`](reference_map.md) 与 [`references.md`](references.md)：参考仓库和资料来源。

当前 CUDA 行为以 `minitrain/kernels/cuda_ext/README.md` 和上述三份 CUDA 专题文档
为准；历史计划用于解释决策背景，不作为现行运行手册。
