# `flash_bwd_kernel.h` 结构与实现详解

本文专门讲解
[`flash_bwd_kernel.h`](../minitrain/kernels/cuda_ext/csrc/third_party/flash_attn/src/flash_bwd_kernel.h)，
并结合它依赖的
[`flash_bwd_preprocess_kernel.h`](../minitrain/kernels/cuda_ext/csrc/third_party/flash_attn/src/flash_bwd_preprocess_kernel.h)、
[`flash_bwd_launch_template.h`](../minitrain/kernels/cuda_ext/csrc/third_party/flash_attn/src/flash_bwd_launch_template.h)
解释完整 backward 流程。

对应的 forward 讲解见
[`flash_fwd_kernel_deep_dive.md`](flash_fwd_kernel_deep_dive.md)。建议先理解 forward 的 tile、CuTe
命名和 online softmax，再阅读本文。

本文重点回答：

- backward 为什么不需要保存完整 attention probability；
- forward 保存的 O、LSE 和 Philox RNG state 分别有什么作用；
- 为什么 backward 固定一个 K/V 列块，再倒序扫描 Q 行块；
- dQ、dK、dV 分别由哪些矩阵乘得到；
- 为什么 dQ 需要 FP32 中间缓冲和原子累加，而 dK/dV 可以由 CTA 独占；
- dropout mask 如何从 forward RNG state 精确重放；
- `dot(dO, O)` 为什么能替代显式计算 softmax Jacobian 的行归约；
- seq-k parallel、deterministic、double buffer 和 shared-memory 复用如何工作；
- 哪些同步、缩放和边界逻辑不能随意修改。

## 1. Backward 的数学基础

### 1.1 不带 dropout 的公式

对单个 batch、单个 head，forward 为：

\[
S=QK^T,
\]

\[
Z=\alpha S+B,
\]

\[
P=\operatorname{softmax}(Z),
\]

\[
O=PV.
\]

其中 \(\alpha\) 是 softmax scale，\(B\) 包含 causal/local mask 和 ALiBi 等项。给定上游梯度
\(dO\)，各梯度为：

\[
dV=P^TdO,
\]

\[
dP=dOV^T,
\]

softmax 的逐行导数可写成：

\[
D_i=\sum_j P_{ij}dP_{ij},
\]

\[
dZ_{ij}=P_{ij}(dP_{ij}-D_i).
\]

最后：

\[
dQ=\alpha dZK,
\]

\[
dK=\alpha dZ^TQ.
\]

这个写法避免显式构造 softmax Jacobian。每个 Query 行只需要一个标量 \(D_i\)。

### 1.2 为什么 `D = dot(dO, O)`

因为：

\[
O_i=\sum_jP_{ij}V_j,
\]

所以：

\[
dO_i\cdot O_i
=dO_i\cdot\sum_jP_{ij}V_j
=\sum_jP_{ij}(dO_i\cdot V_j)
=\sum_jP_{ij}dP_{ij}
=D_i.
\]

因此 backward 不必先物化 `dP` 再单独沿 K 方向归约，只需预先计算每个 Query 行的：

```text
softmax_d[i] = dot(dO[i], O[i])
```

源码中这个数组叫 `dsoftmax_sum`，局部变量通常叫 `dP_sum`。

### 1.3 带 dropout 时的公式

设 keep mask 为 \(M\in\{0,1\}\)，keep probability 为 \(p\)，`rp_dropout=1/p`。forward 输出：

\[
O=\frac{1}{p}(M\odot P)V.
\]

于是：

\[
dV=\frac{1}{p}(M\odot P)^TdO,
\]

而进入 softmax 导数的概率梯度只在保留位置有效。实现没有把 `1/p` 乘进每个中间 `dP`，而是延迟到最终
dQ、dK、dV 缩放。因此预处理中的 `dot(dO,O)` 会乘 keep probability，使 `dP` 与 `dP_sum`
保持同一缩放基准。

注意参数名容易误解：上游结构中的 `p_dropout` 在这些设备端公式里表示 keep probability；
`rp_dropout` 是它的倒数。

## 2. Backward 为什么可以重算 P

forward 不保存完整 \(P\)，只保存：

- Q、K、V；
- forward output O；
- 每个 Query 行的 LSE；
- dropout 启用时的 Philox seed/offset。

LSE 为：

\[
L_i=\log\sum_j e^{Z_{ij}}.
\]

backward 对当前 Q/K tile 重新计算 score 后，可直接恢复：

\[
P_{ij}=e^{Z_{ij}-L_i}.
\]

这不需要重新执行跨所有 K block 的 online softmax，也不需要再次求整行 max/sum，因为 forward 的 LSE 已经
包含完整行的归一化信息。

dropout mask 同样不保存。backward 使用 forward 保存的 seed/offset，加上相同的 batch、head、warp lane
和 attention 子块坐标，重建完全相同的 keep mask。

这是 FlashAttention backward 的核心空间换时间策略：

```text
不保存 O(S_q * S_k) 的 P/mask
             |
             v
backward 分块重算 QK^T、P 和 dropout mask
```

## 3. 完整 backward 调用链

当前 MiniTrain 集成中的主要调用链是：

```text
Python autograd backward
  -> C++ adapter 填充 Flash_bwd_params
  -> run_mha_bwd_*()
  -> run_flash_bwd()
  -> run_flash_bwd_seqk_parallel()
       |
       +-- flash_bwd_dot_do_o_kernel
       |     -> compute_dot_do_o()
       |
       +-- flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel
       |     -> compute_dq_dk_dv_seqk_parallel()
       |          -> compute_dq_dk_dv_1colblock()
       |
       +-- flash_bwd_convert_dq_kernel
             -> convert_dQ()
```

当前 `run_flash_bwd()` 直接调用 `run_flash_bwd_seqk_parallel()`。文件中还保留了非 seq-k 的：

```text
flash_bwd_dq_dk_dv_loop_kernel
  -> compute_dq_dk_dv()
```

但它不是当前 launcher 的主路径。阅读时可用它理解 `1colblock` 的另一种累计设计，但不应在未检查模板
实参和实例化矩阵前假定该保留路径当前可直接启用。

## 4. 三个文件如何分工

| 文件 | 作用 |
| --- | --- |
| `flash_bwd_launch_template.h` | 选择 traits/模板实例，启动预处理、主 kernel 和 dQ 转换 kernel |
| `flash_bwd_preprocess_kernel.h` | 计算 `dot(dO,O)`、清零累加区、转换 FP32 梯度缓冲 |
| `flash_bwd_kernel.h` | 重算 P/dS，执行 dQ/dK/dV 的矩阵乘和主流水 |

与 forward 相比，backward 不只是一个 kernel。seq-k 主路径至少包含：

1. 预处理 `softmax_d`；
2. 主 dQ/dK/dV kernel；
3. dQ FP32 累加结果的归并、缩放和类型转换。

## 5. `flash_bwd_kernel.h` 自身结构

该文件约 841 行，主要由五部分组成：

| 行号 | 函数 | 作用 |
| ---: | --- | --- |
| 29 | `make_tiled_copy_B_warpcontiguousN()` | 为 B operand 构造 N 方向 warp-contiguous copy |
| 57 | `make_tiled_copy_C_warpcontiguousN()` | 为 C/P/dS 构造 N 方向 warp-contiguous copy |
| 80 | `compute_dq_dk_dv_1colblock()` | 固定一个 K/V 列块，完成 backward 主计算 |
| 799 | `compute_dq_dk_dv()` | 非 seq-k 调度包装，顺序处理多个 K block |
| 826 | `compute_dq_dk_dv_seqk_parallel()` | 当前主调度包装，沿 K sequence 并行 CTA |

真正的算法主体几乎全部集中在 `compute_dq_dk_dv_1colblock()`。

## 6. Forward 与 backward 的 CTA 分工差异

### 6.1 Forward 固定 Q 行块

forward 中一个 CTA 固定：

```text
一个 Q tile [kBlockM, d]
```

然后扫描所有相关 K/V tile。这样一个 CTA 可以完整产生该 Q tile 的 O。

### 6.2 Backward 固定 K/V 列块

backward 的 `1colblock` 固定：

```text
一个 K/V tile [kBlockN, d]
```

然后倒序扫描所有与它相连的 Q/dO tile：

```text
固定 K[n_block], V[n_block]
    |
    +-- Q[m_block_max - 1], dO[m_block_max - 1]
    +-- Q[m_block_max - 2], dO[m_block_max - 2]
    +-- ...
    +-- Q[m_block_min],     dO[m_block_min]
```

这样做的直接好处是：

- 当前 CTA 独占一个 dK tile；
- 当前 CTA 独占一个 dV tile；
- dK/dV 可以在 FP32 寄存器中跨所有 Q block 累积，最后只写一次。

代价是每个 Q tile 会同时接收多个不同 K block 的 dQ 贡献，所以 dQ 需要跨 CTA 归并。

## 7. Seq-K parallel 的网格映射

主 kernel 的 grid 为：

```cpp
grid_n = dim3(gridDimx, batch, query_heads);
```

设备端解释为：

```cpp
bidb = blockIdx.y;
bidh = blockIdx.z;
```

K block 使用：

```cpp
for (n_block = blockIdx.x;
     n_block < total_n_blocks;
     n_block += gridDim.x)
```

非 deterministic 模式下通常：

```text
gridDim.x = total_n_blocks
```

即一个 CTA 对应一个 `n_block`。

deterministic 模式下，`gridDim.x` 根据 SM 数量限制。每个 `blockIdx.x` 以固定 stride 处理多个 K block，
并写入自己独立的 dQ accumulation split。最后 `convert_dQ()` 按固定顺序合并这些 split。

当前 MiniTrain C++ adapter 将 `params.deterministic=false`，但上游设备代码保留 deterministic 支持。

## 8. `Flash_bwd_params` 中的重要输入与 workspace

`Flash_bwd_params` 继承 forward 参数，并增加：

| 字段 | 作用 |
| --- | --- |
| `do_ptr` | 上游梯度 dO |
| `dq_ptr/dk_ptr/dv_ptr` | 最终输出梯度 |
| `dq_accum_ptr` | dQ 的 FP32 跨 CTA 累加缓冲 |
| `dk_accum_ptr/dv_accum_ptr` | 其他并行策略可用的 FP32 dK/dV 缓冲 |
| `dsoftmax_sum` | 每个 Q 行的 `dot(dO,O)` |
| `softmax_lse_ptr` | forward 保存的 LSE |
| `rng_state` | forward 保存的 Philox seed/offset |
| `deterministic` | 是否使用分离的 dQ split 缓冲 |
| `dq_accum_split_stride` | deterministic split 之间的跨度 |

另外仍会使用 forward 参数中的 scale、dropout、mask、ALiBi、softcap、sequence length 和 stride。

## 9. 预处理 kernel：`compute_dot_do_o()`

### 9.1 Grid 分工

预处理 grid 是：

```text
[num_q_blocks, batch, query_heads]
```

每个 CTA 读取同一个 Q row tile 对应的 O 和 dO。

### 9.2 计算 softmax 行归约

`dot_do_o()` 把 O/dO 转成 FP32，对 head dimension 做：

\[
D_i=dO_i\cdot O_i.
\]

线程先计算自己的局部点积，再通过 `Allreduce<THREADS_PER_ROW>` 做行内归约，最后每行由一个线程写入
`dsoftmax_sum`。

dropout 开启时，这里按实现的延迟缩放约定乘 keep probability。

### 9.3 清零 dQaccum

非 deterministic 路径中，多个 K-block CTA 随后会 `atomicAdd` 到同一 dQaccum，因此预处理同时把实际会
被原子写入的区域清零。

deterministic 路径传入 `Clear_dQaccum=false`，因为它使用按 CTA 分离的 split buffer，其初始化和合并
约定不同。

## 10. `Kernel_traits`：Backward 有三套 MMA

forward 主要有 `QK^T` 和 `PV` 两种 MMA。backward 需要更多矩阵乘，因此 traits 中定义：

```cpp
TiledMmaSdP // score S 和 dP
TiledMmadKV // dK 和 dV
TiledMmadQ  // dQ
```

对应五次逻辑 GEMM：

| GEMM | 数学含义 | 输出 shape |
| --- | --- | --- |
| `Q K^T` | 重算 score | `[kBlockM, kBlockN]` |
| `dO V^T` | 计算 dP | `[kBlockM, kBlockN]` |
| `P^T dO` | 累积 dV | `[kBlockN, d]` |
| `dS K` | 计算当前块对 dQ 的贡献 | `[kBlockM, d]` |
| `dS^T Q` | 累积 dK | `[kBlockN, d]` |

traits 还决定：

- `Is_V_in_regs`：把 V 常驻寄存器，减少 shared memory、增加寄存器压力；
- `No_double_buffer`：禁用 Q 双缓冲，减少 shared memory、牺牲流水重叠；
- `kSmemSize1colblock`：主 kernel 动态 shared memory；
- Q/dO、K/V、P/dS、dQ/dKV 的 shared-memory layouts；
- global-memory copy 和 atomic-add copy 的线程映射。

## 11. 两个 warp-contiguous copy helper

文件开头两个 helper 针对 backward 的 N 方向线程布局构造 tiled copy：

```cpp
make_tiled_copy_B_warpcontiguousN()
make_tiled_copy_C_warpcontiguousN()
```

backward 的 score/P/dS tile 同时服务于：

- `QK^T` / `dOV^T` 的输出；
- `P^T dO` / `dS^T Q` 的转置输入。

helper 让同一 warp 在 N 方向取得连续、适合后续转置 MMA 的元素映射。源码注释中的 “This gives the
correct layout, idk why” 也说明这里是高度依赖 CuTe/MMA lane layout 的经验性配置；不能只按二维矩阵直觉
重写。

## 12. `1colblock` 的地址与张量视图

函数先固定：

```text
batch = bidb
query head = bidh
K/V block = n_block
```

然后构造 global views：

```cpp
gQ, gK, gV
gdO, gO
gdQ, gdQaccum
gLSE, gdPsum
```

Q、dO、O 的初始指针指向 `m_block_max - 1`，随后循环向前移动。K/V 地址在整个 `1colblock` 调用中
固定。

读取 K/V 时使用：

```cpp
bidh / params.h_h_k_ratio
```

处理 GQA/MQA 的 Q-head 到 KV-head 映射。最终 dK/dV 如何跨共享同一 KV head 的多个 Q head 归并，还
取决于上层参数与 workspace 策略；MiniTrain 当前 adapter 要求 Q/K/V head shape 一致，常规路径即
`h_h_k_ratio=1`。

## 13. Shared-memory 全局布局

主函数在同一片动态 shared memory 上建立：

```text
sQ / sQt             Q 及其转置 view
sdO / sdOt           dO 及其转置 view
sK / sKt             K 及其转置 view
sV                    V
sdS / sdSt           dS 及其转置 view
sP / sPt             P 及其转置 view
sdQ                   dQ staging，和 sP 复用地址
```

这里的 `t` 后缀通常是同一数据地址的转置逻辑 view，不一定发生物理 transpose。

重要复用关系包括：

- `sP` 与 `sdQ` 共享地址；
- epilogue 的 `sdK/sdV` 会复用原来的 `sK/sV` 区域；
- Q 可选双缓冲，在两个 `sQ` page 之间交替；
- `Is_V_in_regs=true` 时，V 读进寄存器后可释放部分 shared-memory 空间。

这些复用是大量 `__syncthreads()` 的根本原因。

## 14. 计算 Q-block 范围

固定 `n_block` 后，函数计算它可能连接的：

```text
[m_block_min, m_block_max)
```

普通非 causal/local attention 中，所有有效 Q block 都参与。

causal/local 路径使用 Q/K 实际长度差和 window size 裁剪范围。表达式中的：

```cpp
actual_seqlen_q - actual_seqlen_k
```

用于 Q/K 长度不等时的位置右对齐。

local attention 下可能出现某个 K tile 不被任何 Q 行访问。此时 kernel 必须显式把对应 dK/dV 写 0 后
返回，否则既可能读取越界 Q/dO，也可能留下未初始化梯度。

## 15. Prologue：加载固定和首轮数据

prologue 主要完成：

1. 为 head-dim tail 构造 Q/K/V predicate；
2. 根据 `m_block` 奇偶选择 Q double-buffer page；
3. 加载首个 Q 和 dO/O tile；
4. 加载固定的 K/V tile；
5. 加载当前 Q 行的 LSE；
6. 首次需要时计算/读取 `dP_sum`；
7. 可选把 V 从 shared memory 搬入寄存器；
8. 创建 dropout 与 ALiBi 状态；
9. 清零跨 Q-block 累积的 `acc_dk/acc_dv`。

如果是非 seq-k 的多 `n_block` 调度，`Is_first/Is_last` 还决定 dQaccum 是初始化、继续读取还是最终直接
写 dQ。当前 seq-k 主路径固定以 `Seq_parallel=true, Is_first=false, Is_last=false` 调用，dQ 始终走
原子累加和后处理。

## 16. 主循环第一步：重算 score 和 P

每轮首先执行：

\[
S_b=Q_bK_{fixed}^T.
\]

然后按 forward 相同顺序重放：

1. softcap；
2. 计算 softcap 的导数因子 `dtanh`；
3. ALiBi；
4. causal/local/tail mask；
5. 使用 LSE 恢复 P。

概率恢复对应：

\[
P_{ij}=e^{Z_{ij}-L_i}.
\]

源码通过 `scale_apply_exp2(..., lse, scale_softmax_log2)` 使用以 2 为底的指数实现。

对于越界 Q 行，LSE 人工设为正无穷，从而恢复出的 P 恒为 0。只把 Q/K padding 清零还不够：ALiBi
可能改变零 score，极端情况下还会引入 NaN，所以必须从概率层面屏蔽。

## 17. Mask 为什么仍然必须重做

一种看似合理但错误的想法是：越界 K 已被加载成 0，最后乘 K/V 时自然没有贡献，因此可以跳过 score
mask。

源码明确解释了反例：越界位置的 score 可能在后续运算和 FP16/BF16 转换时溢出成 Inf，使 dS 或 dQ
出现 NaN。causal/local/尾部 mask 因此不仅是数学语义，也承担数值稳定性保护。

mask 只在 tile 可能穿过边界时执行，内部完整 tile 会跳过复杂判断。

## 18. Dropout mask 重放与符号位编码

backward 的 `Dropout` 使用：

```cpp
params.rng_state[0]
params.rng_state[1]
batch/head/thread/tile coordinates
```

重建 forward mask。

调用 `apply_dropout<encode_dropout_in_sign_bit=true>()` 后，score/P fragment 使用符号位记录 keep/drop：

- 正值：该位置被保留；
- 负值：该位置被 dropout。

生成供 dV 使用的 `rP` 时，`convert_type_relu()` 把负值裁成 0，因此：

\[
dV\mathrel{+}=(M\odot P)^TdO.
\]

但原始带符号 `scores` 仍保留在 FP32 fragment 中，后面计算 dS 时利用符号区分：

```cpp
kept:    P * (dP - D)
dropped: P * (0  - D)
```

源码对 dropped 分支借助负号编码实现等价结果，避免额外保存布尔 mask fragment。

## 19. 第二步：计算 dP

当前 Q/dO tile 与固定 V tile 做：

\[
dP_b=dO_bV_{fixed}^T.
\]

对应 `acc_dp`，shape 是：

```text
[kBlockM, kBlockN]
```

如果 `Is_V_in_regs=true`，V 已常驻寄存器；否则从 shared memory 读取。这个选择在 shared-memory 使用量
与寄存器压力之间权衡。

## 20. 第三步：softmax backward 得到 dS

代码把 `acc_dp` 按 score 的 row/column layout 重解释，然后逐元素执行：

\[
dZ_{ij}=P_{ij}(dP_{ij}-D_i).
\]

其中 `D_i` 来自预处理的 `dsoftmax_sum`。

如果启用 softcap，还要乘 logits 变换的导数。假设 softcap 形式是缩放后的 tanh，则这里的 `dtanh`
承担链式法则对应项。

得到的 FP32 dS 随后转换为 FP16/BF16 并写入 `sdS`，供后续三个 Tensor Core GEMM 复用。

scale 和 dropout reciprocal 没有在每个 dS 元素上立即乘，而是延迟到最终 dQ/dK 输出前统一应用，以减少
循环中的逐元素操作。

## 21. 第四步：累积 dV

使用转置 P view 和转置 dO view：

\[
acc\_dv\mathrel{+}=P_b^TdO_b.
\]

`acc_dv` 对固定 K/V tile 跨所有 Q block 累积，始终保持 FP32。

dropout 的 dropped P 已在转换时变为 0。循环结束后，若启用 dropout，再统一乘：

\[
1/p.
\]

## 22. 第五步：计算并归并 dQ

当前 K tile 对 Q tile 的贡献：

\[
dQ_b^{(n)}=dS_bK_{fixed}.
\]

一个 Q tile 会接收所有相关 `n_block` 的贡献：

\[
dQ_b=\sum_n dQ_b^{(n)}.
\]

seq-k 并行时，不同 K-block CTA 同时产生同一 dQ tile 的部分和，所以代码对 FP32 `dq_accum_ptr`
执行逐元素 `atomicAdd`。

为什么不直接原子加到 FP16/BF16 `dq_ptr`：

- FP32 累加误差更小；
- 原子支持和性能更合适；
- 最终 scale、dropout reciprocal 和类型转换只做一次。

## 23. 第六步：累积 dK

使用转置 dS 和 Q：

\[
acc\_dk\mathrel{+}=dS_b^TQ_b.
\]

因为 CTA 固定一个 K tile，它可以在寄存器中遍历所有 Q block 后得到完整 dK tile，无需与其他
`n_block` CTA 合并。

循环结束后 dK 统一乘：

```cpp
params.scale_softmax_rp_dropout
```

即 softmax scale 与 dropout reciprocal 的组合。

## 24. Q/dO 双缓冲流水

K/V 在 `1colblock` 中固定，变化的是 Q、dO、O 和 LSE。主循环因此围绕 Q 方向构造流水：

```text
计算当前 Q block 的 score/P/dP/dS
       |
       +-- 异步预取下一个 Q block 到另一 sQ page
       +-- 预取下一个 dO block
       +-- 更新下一个 LSE/dP_sum 指针
       |
       +-- 用当前 dS 完成 dV/dQ/dK GEMM
       v
切换 shared-memory page，进入下一轮
```

当 `No_double_buffer=false` 时，两个 sQ page 按 `m_block` 奇偶切换。`tdKsQt` 等 transpose view 的
data pointer 也必须同步切换，否则 dK 会读取错误 page。

禁用双缓冲能降低 shared-memory 占用，但需要在覆盖 sQ 前等待当前计算完成，流水重叠减少。

## 25. 为什么 Q-block 也倒序遍历

函数从：

```cpp
m_block = m_block_max - 1
```

向 `m_block_min` 递减。

这与 forward 倒序扫描 K 的思路相似：末端 Q tile 最可能有 sequence tail，也更靠近 causal/local 的复杂
边界。先处理尾部后，后续地址只需固定步长递减；同时可能节省一个寄存器中的上界状态。

## 26. Epilogue：dK/dV 写回

循环结束后：

1. 对 `acc_dv` 应用 dropout reciprocal；
2. 对 `acc_dk` 应用 softmax scale 和 dropout reciprocal；
3. 将 FP32 accumulator 转为 FP16/BF16；
4. 写入复用的 `sdK/sdV` shared-memory staging；
5. 重新排列成适合 global store 的线程布局；
6. 使用 M/N/head-dim predicate 写入最终 dK/dV。

写 `sdK/sdV` 前的同步非常关键。它们复用了此前的 `sK/sV` 地址，如果还有线程在 dQ/dK GEMM 中读取
旧 K/V，提前覆盖就会造成 race。

## 27. 后处理 kernel：`convert_dQ()`

主 kernel 结束后，`dq_accum_ptr` 中保存 FP32 dQ 部分和。

`convert_dQ()` 使用 forward 类似的 Q-tile grid：

1. 读取一个或多个 dQ accumulation split；
2. 按固定循环累加 split；
3. 乘 `scale_softmax_rp_dropout`；
4. 转换为 FP16/BF16；
5. 经 shared memory 重排；
6. 用 Q-row/head-dim predicate 写入 `dq_ptr`。

非 deterministic 模式通常 `nsplits=1`，因为所有 K CTA 已原子加进同一缓冲。deterministic 模式下
`nsplits=gridDim.x`，每个 split 独立，最后以固定顺序求和。

### 27.1 `clear_dKVaccum()` 与 `convert_dKV()`

`flash_bwd_preprocess_kernel.h` 还提供：

```cpp
clear_dKVaccum()
convert_dKV()
```

它们服务于“沿 Q sequence 拆分 backward、多个 CTA 共同产生同一 dK/dV”一类调度：先清零 FP32
`dk_accum/dv_accum`，主计算原子累加 partial，最后分别乘正确 scale 并转换到最终 dK/dV。

当前 `run_flash_bwd_seqk_parallel()` 固定 K/V tile 所有权，`compute_dq_dk_dv_1colblock()` 已直接在
寄存器中完成 dK/dV 并写回，因此当前启动链没有调用这两个 kernel。它们反映的是上游文件保留的另一套
并行方向和 workspace 支持，不应与当前 seq-k 主路径混在一起。

## 28. Deterministic 与非 deterministic dQ

### 28.1 非 deterministic

```text
每个 K block 一个 CTA
  -> 所有 CTA atomicAdd 到同一 dq_accum
  -> convert_dQ 读取一个 buffer
```

浮点原子加法的执行顺序由调度决定，因此末位结果可能在不同运行间略有差异。

### 28.2 Deterministic

```text
每个 blockIdx.x 拥有独立 dq_accum split
  -> 该 CTA 按 stride 顺序处理若干 K block
  -> convert_dQ 按 split 0..N-1 固定顺序求和
```

代价是更多 workspace、可能更少的 K 方向并行度和额外归并开销。

这里的 deterministic 主要解决跨 CTA dQ 累加顺序；它不意味着任意 GPU、编译器或 dtype 之间逐 bit
完全一致。

## 29. 非 seq-k 包装中的 `Is_first/Is_last`

保留的 `compute_dq_dk_dv()` 设计为在一个更外层的调用中倒序处理多个 `n_block`，并通过模板标记：

```text
只有一个 block: Is_first=true,  Is_last=true
第一个调用:     Is_first=true,  Is_last=false
中间调用:       Is_first=false, Is_last=false
最后调用:       Is_first=false, Is_last=true
```

这些标志控制：

- dQ accumulator 是清零、从 global 读回，还是最终写入 dQ；
- `dot(dO,O)` 是现场计算还是复用；
- 某些同步能否省略；
- dO/O 使用寄存器还是 shared-memory copy。

seq-k 主路径不依赖这一串跨 `n_block` 的 CTA 内状态，而是让每个 K block 独立产生 dQ partial 并原子归并。
当前 launcher 没有实例化这条保留包装；若要重新启用，需要先审计它与现有 `1colblock` 模板签名和 traits
组合是否仍一致。

## 30. 普通主路径的数据流

```text
预处理：
O + dO
  -> dot(dO, O)
  -> dsoftmax_sum[D]
  -> clear dq_accum

主 kernel（固定 K/V tile）：
Q + K
  -> QK^T
  -> softcap / ALiBi / mask
  -> exp(score - LSE)
  -> replay dropout
  -> P

dO + V
  -> dP = dO V^T

P + dP + dsoftmax_sum
  -> dS = P * (dP - D)
  -> optional softcap derivative

P^T + dO   -> accumulate dV
dS + K      -> partial dQ -> atomic dq_accum
dS^T + Q    -> accumulate dK

epilogue：
acc_dK/acc_dV
  -> scale / convert / store dK,dV

后处理：
dq_accum split(s)
  -> sum / scale / convert / store dQ
```

## 31. CuTe 命名阅读法

命名规则与 forward 相同：

| 前缀/片段 | 含义 |
| --- | --- |
| `g` | global-memory tensor |
| `s` | shared-memory tensor |
| `r` | register fragment |
| `acc` | FP32 accumulator |
| `t` | 当前 thread slice 的 partition |
| `c` | identity coordinate tensor |
| `p` | boundary predicate |
| `d` | 梯度量，如 dO/dQ/dS |
| `t` 后缀 | transpose layout view，如 `sQt/sKt/sPt` |

例如：

```cpp
tdOgdO       // dO copy 中当前线程的 global source
tdOsdO       // dO copy 中当前线程的 shared destination
tdQgdQaccum  // 当前线程负责的 global dQ accumulation 区域
tdKrdSt      // dK MMA 使用的寄存器 dS-transpose fragment
tdKrQt       // dK MMA 使用的 Q-transpose operand fragment
```

阅读一个复杂名称时，先找最后两个大写张量名，再判断中间的 `g/s/r` 存储位置，不要试图一次理解完整
CuTe layout 类型。

## 32. 边界 predicate 的三个维度

backward 同时处理三种尾部：

1. **M tail**：实际 Q 长度不是 `kBlockM` 整数倍；
2. **N tail**：实际 K 长度不是 `kBlockN` 整数倍；
3. **K/head-dim tail**：实际 `params.d` 小于编译期 `kHeadDim` bucket。

`Is_even_MN` 和 `Is_even_K` 为 true 时，编译器可移除对应 predicate；为 false 时通过 identity tensor 的
逻辑坐标判断每个线程元素是否有效。

注意这里 `Is_even_K` 的 K 指 GEMM K/head dimension，不是 Key sequence length。

## 33. Causal、local 与 ALiBi 的 backward

这些功能不需要新的梯度矩阵公式，但必须在重算 P 时与 forward 完全一致：

- causal/local：相同位置必须恢复为 0 概率；
- ALiBi：相同 bias 必须加回 score；
- softcap：既要重放 forward 变换，也要在 dS 乘其导数。

当前代码只使用 ALiBi slope 重算概率，没有在这里输出 slope gradient。mask 自身同样没有梯度。

Q/K 长度不等时，causal/local 坐标使用右对齐偏移；修改 forward mask 坐标后必须同步修改 backward，否则
概率重算与 forward 不一致。

## 34. Shared-memory 同步的关键危险点

### 34.1 `cp_async_wait` 不能替代 `__syncthreads`

前者等待异步 copy group，后者确保 CTA 内所有线程对 shared memory 的写入/读取阶段一致。多数 load-to-MMA
边界需要二者配合。

### 34.2 `sP` 与 `sdQ` 地址复用

在把 dQ accumulator 写入 `sdQ` 前，必须保证没有线程继续从 `sP` 读取概率。

### 34.3 `sdK/sdV` 覆盖 `sK/sV`

epilogue 写梯度 staging 前必须保证最后一次 dQ/dK GEMM 已读完 K/V。源码对此有明确 race 注释。

### 34.4 非双缓冲路径覆盖 sQ

下一块 Q load 前需要同步，避免当前 dK GEMM 尚未读取完 Q，shared memory 已被下一块覆盖。

### 34.5 dO 的同址复用

循环中 dV GEMM 仍会读取当前 sdO；预取下一块 dO 前必须等待，源码第 640 行附近的同步就是为此服务。

## 35. 缩放因子最容易改错

实现有意延迟若干 scale：

- P 恢复时应用 `scale_softmax_log2`；
- dQ/dK 最终乘 `scale_softmax_rp_dropout`；
- dV 只乘 `rp_dropout`；
- `dsoftmax_sum` 在 dropout 情况按 keep probability 调整；
- softcap derivative 在 dS 上单独乘。

如果把 `1/p` 提前乘进 dP，却没有同步调整 `dP_sum`，softmax 导数中的：

\[
dP-D
\]

就会处于不同缩放尺度，得到系统性错误。修改缩放时必须沿 `dsoftmax_sum -> dS -> dQ/dK/dV`
整条链一起推导。

## 36. 精度策略

主要输入和中间 staging 通常是 FP16/BF16：

- Q、K、V、O、dO；
- 重算后的 P；
- 写入 shared memory 的 dS；
- 最终 dQ/dK/dV。

主要累加状态是 FP32：

- score 与 dP accumulator；
- dQ/dK/dV accumulator；
- LSE 与 dsoftmax_sum；
- deterministic split 合并。

dS 在进入 Tensor Core GEMM 前转换为 Element 是性能/精度权衡。最终跨多个 block 的 dQ accumulation
保留 FP32，避免直接在低精度输出上原子累加。

## 37. 资源压力为什么比 forward 更大

backward 同时持有或流水使用：

- Q、K、V、dO；
- P、dS；
- score、dP；
- acc_dQ、acc_dK、acc_dV；
- 多套 MMA fragment 和 copy partition。

因此它通常比 forward 消耗更多寄存器和 shared memory。`Is_V_in_regs`、Q double buffer、tile 大小和
warp layout 都是在：

```text
寄存器压力 / spill
shared-memory 大小
occupancy
异步流水重叠
```

之间权衡。

这也是高 head-dim、dropout backward 对硬件 opt-in shared memory 更敏感的原因。不要只根据单个
register 数或 occupancy 数字判断整体性能。

## 38. 容易误读的地方

### 38.1 `dP_sum` 不是对当前 tile 的临时归约

它是完整 Query 行的 softmax backward 标量，由 O 和 dO 预处理得到，对所有 K tile 复用。

### 38.2 backward 重算的是 P，不是 online softmax 状态

forward 已保存 LSE，所以 backward 对每个 tile 直接计算 `exp(score-LSE)`，不需要再次遍历整行更新
row max/sum。

### 38.3 `1colblock` 的 “col” 指 K/V 列块

一个调用固定 `n_block`，循环的是 Q 方向 `m_block`。

### 38.4 dK/dV 不需要 seq-k 原子归并

当前 CTA 独占对应 K tile 并扫描全部相关 Q tile；需要跨 K CTA 合并的是 dQ。

### 38.5 dropout 符号位不是负概率

负号是临时 mask 编码。供 dV 使用时会 ReLU 为 0；供 dS 使用时负号帮助实现 dropped 位置的
`P*(0-D)`。

### 38.6 transpose view 不一定物理转置

`sQt/sKt/sPt/sdSt` 多数是同一地址的不同 layout 解释。改变底层 layout 会同时影响多个 MMA。

### 38.7 `Is_first/Is_last` 不是 Q 循环首尾

它们描述非 seq-k 外层多个 `n_block` 调用对 dQaccum 的阶段，而 Q 循环首尾由 `m_block` 判断。

## 39. 推荐源码阅读顺序

建议按以下顺序精读：

1. forward 深度导读的 online softmax、CuTe 命名和数据流；
2. `flash_bwd_launch_template.h` 第 29–132 行，确认实际启动的三个 kernel；
3. `flash_bwd_preprocess_kernel.h` 第 24–140 行，理解 `dot(dO,O)`；
4. `flash_bwd_kernel.h` 第 799–838 行，理解两种调度包装；
5. 第 80–178 行，查看地址、global/shared views 和内存复用；
6. 第 216–309 行，查看三套 MMA 与 tiled-copy partition；
7. 第 311–456 行，查看 Q 范围、early exit 和 prologue；
8. 第 457–596 行，查看 P 重算、dropout replay、dP 和 dS；
9. 第 598–724 行，查看 dQ/dK/dV 流水与 dQ accumulation；
10. 第 726–795 行，查看 dK/dV epilogue；
11. `flash_bwd_preprocess_kernel.h` 的 `convert_dQ()`，完成主路径闭环；
12. 最后再展开 `Flash_bwd_kernel_traits` 的具体 lane/layout。

## 40. 调试与验证建议

### 40.1 数学正确性

小 shape 下分别比较：

- dQ、dK、dV；
- causal/non-causal；
- dropout/no-dropout；
- M/N/head-dim tail；
- softcap、ALiBi、local；
- varlen 与定长。

不要只检查最终 dQ。dV 正确而 dQ/dK 错误通常指向 dS/scale；dQ 正确而 dK/dV 错误更可能指向转置
layout、epilogue 或 K-tile 所有权。

### 40.2 分阶段观察

推荐按以下顺序定位：

```text
dsoftmax_sum
  -> recomputed score
  -> P before/after dropout replay
  -> dP
  -> dS
  -> partial dQ / acc_dK / acc_dV
  -> final converted gradients
```

### 40.3 同步与越界

修改 shared-memory pipeline 后，使用 CUDA sanitizer 检查：

- out-of-bounds global/shared access；
- racecheck；
- 非整齐 M/N/D；
- local attention 的空 K tile；
- varlen 的空 Q/K 序列。

### 40.4 性能与资源

查看：

- ptxas registers、stack、spill loads/stores；
- dynamic shared memory；
- theoretical/achieved occupancy；
- local-memory 和 DRAM 流量；
- 预处理、主 kernel、convert-dQ 三段各自耗时；
- atomic contention 随 sequence length/head 数的变化。

不能只优化主 kernel 而忽略 `dot_do_o` 和 `convert_dQ`，短序列时固定开销可能占比较高。

## 41. 总结

`flash_bwd_kernel.h` 的核心可以概括为：

1. 利用 forward LSE 分块重算 P，而不读取完整 attention matrix；
2. 利用 forward Philox state 重放 dropout mask，而不保存二次方 mask；
3. 一个 CTA 固定一个 K/V tile，倒序扫描相关 Q/dO tile，从而在寄存器中独占累计 dK/dV；
4. 每轮依次完成 `QK^T`、`dOV^T`、softmax backward、`P^TdO`、`dSK` 和 `dS^TQ`；
5. 不同 K CTA 对同一 dQ 的贡献先原子累加到 FP32 workspace，再由独立 kernel 缩放并转换；
6. double buffer、异步 copy、转置 view 和 shared-memory 复用共同降低数据搬运成本。

从数学上看，最关键的恒等式是：

\[
D_i=dO_i\cdot O_i=\sum_jP_{ij}dP_{ij}.
\]

从并行结构上看，最关键的选择是“固定 K/V 列块而不是 Q 行块”。前者让 dK/dV 无需跨 CTA 归并，
把冲突集中到 dQ，再通过 FP32 accumulation workspace 解决。理解这两个核心后，文件中复杂的 CuTe
partition、同步和模板分支就都有了明确目的。
