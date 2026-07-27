# sm86 FlashAttention spill 分析

## 结论

当前 sm86 构建确实存在生产路径可达的 register spill，但它首先是性能成本，不会
改变数值正确性，也不是 full build 失败的原因。本项目保留 FlashAttention 2.8.4
原有的 tile、寄存器和 shared-memory 权衡，不针对本机 sm86 改写上游 kernel，
也不增加基于 sequence length 的性能 fallback。

需要分开看三件事：

1. 日志中的最大 spill 来自仅供 correctness 使用的 `Return_softmax=true` 调试
   kernel，训练不会调用；
2. D64、D128 等生产 kernel 也有真实 spill，Nsight 已观察到 local-memory 流量；
3. 之前 full build 的致命错误是两个 nvcc 模板编译进程同时耗尽 16 GB 宿主内存。

分析日志位于
`minitrain/kernels/cuda_ext/build/cuda_build_sm86.log`。环境为 RTX 3050 Laptop
GPU（sm86）、CUDA 12.1、PyTorch 2.5.1+cu121、MSVC 19.44。

## 如何理解 ptxas 输出

ptxas 的 stack、spill stores 和 spill loads 是单线程静态代码属性，不是一次
kernel launch 的总流量。实际 local-memory 流量还取决于 grid、循环次数、谓词
分支和 L1/L2 命中率。

因此：

- non-zero spill 说明需要结合运行时数据评估；
- spill 不会导致错误结果，但可能增加延迟并限制 occupancy；
- 不能把 `548-byte spill loads` 直接乘线程数当作准确的 DRAM 流量；
- `--maxrregcount` 可能把更多 live value 压到 local memory，不能仅凭寄存器数字
  盲目添加。

## 静态编译数据

已完成部分包含 556 条 kernel function report，其中 284 条有非零 spill。这些
条目混合了 dtype、causal、dropout、tail 和 debug 模板，绝不代表一次 attention
会执行 284 个 spill kernel。

原始最大值是：

```text
D=32, dropout=true, causal=true, Return_softmax=true
stack frame: 840 B
spill stores: 800 B
spill loads: 1170 B
```

`Return_softmax=true` 只由 `flash_attention_dropout_mask_for_testing()` 使用，用来
提取 CUDA kernel 的真实 dropout keep mask。生产 autograd 固定传
`return_softmax=false`。

有代表性的生产 forward 数据：

| Head bucket | Dropout | Causal | Stack | Stores | Loads |
| ---: | --- | --- | ---: | ---: | ---: |
| 64 | no | yes | 312 B | 464 B | 548 B |
| 128 | no | yes | 288 B | 328 B | 400 B |
| 64 | no | no | 192 B | 260 B | 276 B |
| 32 | yes | yes | 192 B | 224 B | 256 B |
| 256 | yes | yes | 104 B | 104 B | 200 B |

D64 causal/no-dropout 实例达到每线程 255 个寄存器。255 是 CUDA 编译目标的单线程
架构上限，不表示 kernel 非法，也不表示每种 GPU 的总寄存器文件相同。

## Nsight 运行时证据

对 bf16 `(B=1,H=8,S=1024,D=64)`、causal、no-dropout 的一次 forward 捕获：

| 指标 | 数值 |
| --- | ---: |
| Registers/thread | 255 |
| Dynamic shared memory/block | 49.15 KiB |
| Theoretical occupancy | 16.67% |
| Achieved occupancy | 14.78% |
| Requested local loads | 4.79 MB |
| Requested local stores | 4.28 MB |
| Local-load L1 misses | 1.44 MB |
| Local-store L1 misses | 2.67 MB |
| Kernel duration under NCU | 169.66 us |

这些数据证明 spill 指令在生产路径执行，也说明大部分 local 请求会经过缓存层级，
不能只用静态 ptxas 字段判断最终速度。

同一台机器上的局部 benchmark：

| S | Dropout | CUDA fwd | Torch fwd | CUDA bwd | Torch bwd |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.0 | 0.071 ms | 0.041 ms | 0.151 ms | 0.152 ms |
| 512 | 0.0 | 0.074 ms | 0.053 ms | 0.148 ms | 0.189 ms |
| 1024 | 0.0 | 0.135 ms | 0.142 ms | 0.324 ms | 0.537 ms |
| 1024 | 0.1 | 0.138 ms | 0.214 ms | 0.333 ms | 0.853 ms |

这只是单卡单 shape 证据。它说明 spill 是实际成本，但没有抵消 S=1024 时
FlashAttention 的整体收益；不应据此做跨 GPU 的普遍性能结论。

## Full build 为什么失败

失败点是：

```text
flash_bwd_hdim256_bf16_sm80.cu
catastrophic error: out of memory
ninja -j2
```

这是 CUTLASS/CUTE 大模板实例化耗尽宿主内存，不是 GPU register spill。16 GB
Windows 本机应使用 `MINITRAIN_CUDA_MAX_JOBS=1`；相同矩阵重试时 ninja 会复用
已完成 object。服务器的并行度应按单个 D256 backward 编译进程的峰值内存设置，
不能只按 CPU 核数设置。

## D256 dropout 的独立边界

上游 D256 backward 在共享内存不足的 sm86/sm89 分支只实例化 no-dropout。
MiniTrain 在两层保护这个边界：

- Python 对 sm86/sm89 的 `D > 192 + dropout` 返回 unsupported；
- C++ 对直接 pybind 调用查询 opt-in shared memory，低于 144 KiB 时明确报错。

D256 no-dropout 在 sm86 仍可使用；有足够共享内存的服务器 GPU 保留上游 dropout
路径。这个限制与 register spill 无关。

## 当前工程决策

- 保留 `--ptxas-options=-v`，让新架构构建仍能看到资源数据；
- 接受上游 kernel 的 spill/tile 权衡，不添加寄存器上限；
- 不为短 sequence 增加性能分支或 fallback；
- correctness 和性能都在 `tests/operator_bench.ipynb` 中按明确 shape 验证；
- 将来升级 FlashAttention 版本时重新对比文件 hash、ptxas 数据和 notebook 结果。
