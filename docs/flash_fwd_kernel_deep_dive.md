# `flash_fwd_kernel.h` 结构与实现详解

本文专门讲解
[`flash_fwd_kernel.h`](../minitrain/kernels/cuda_ext/csrc/third_party/flash_attn/src/flash_fwd_kernel.h)。
它是 MiniTrain 所集成 FlashAttention 前向实现的设备端算法主体，包含普通前向、Split-KV
前向和 Split-KV 结果合并三部分。

本文关注以下问题：

- 这个文件在完整调用链中的位置；
- 一个 CUDA CTA 负责什么工作；
- Q、K、V 如何在 global memory、shared memory 和寄存器之间流动；
- 分块计算为什么仍然能得到精确 softmax；
- causal、local、ALiBi、dropout、GQA/MQA 和变长序列如何接入主循环；
- Split-KV、paged KV cache、append KV 和 rotary embedding 如何工作；
- CuTe 的张量、layout、partition 和命名应该怎样阅读；
- 哪些同步语句不能随意移动。

完整的 Python 到 CUDA 调用链见
[`cuda_flash_attention_code_reading_guide.md`](cuda_flash_attention_code_reading_guide.md)。本文从已经进入上游
CUDA forward launcher 的位置开始。

## 1. 先建立整体认识

### 1.1 标准 Attention

对单个 batch、单个 head，标准 scaled dot-product attention 为：

\[
S = QK^T,
\]

\[
P = \operatorname{softmax}(S\cdot \alpha + B),
\]

\[
O = PV.
\]

其中：

- \(Q\in\mathbb{R}^{M\times d}\)；
- \(K,V\in\mathbb{R}^{N\times d}\)；
- \(\alpha\) 是 softmax scale，通常为 \(1/\sqrt d\)；
- \(B\) 表示 causal/local mask、ALiBi 等附加项；
- \(S,P\in\mathbb{R}^{M\times N}\)。

普通实现会物化完整的 `S` 或 `P`。当序列长度很大时，这个二次方规模的中间矩阵会带来很高的显存占用和显存读写开销。

### 1.2 这个 kernel 的基本策略

FlashAttention 将 Q、K、V 分块：

```text
Q tile:  [kBlockM, head_dim]
K tile:  [kBlockN, head_dim]
V tile:  [kBlockN, head_dim]
S tile:  [kBlockM, kBlockN]
O tile:  [kBlockM, head_dim]
```

一个 CTA 固定处理一个 Q tile，然后逐块扫描它允许访问的 K/V：

```text
固定 Q[m_block]
    |
    +-- K/V[n_block_max - 1]
    +-- K/V[n_block_max - 2]
    +-- ...
    +-- K/V[n_block_min]
```

每处理一个 K/V tile，CTA 都执行：

```text
Q x K^T
  -> mask / bias / softcap
  -> online softmax 更新
  -> dropout（可选）
  -> P x V
  -> 累积到寄存器中的 O tile
```

完整的 `M x N` attention matrix 从不进入生产路径的 global memory。计算过程中只保留当前
`kBlockM x kBlockN` score tile、每行 softmax 状态和 `kBlockM x head_dim` 输出累加器。

## 2. 文件在调用链中的位置

与本文最相关的调用链是：

```text
显式实例化 .cu
  -> run_mha_fwd_*()
  -> run_flash_fwd() / run_flash_splitkv_fwd()
  -> flash_fwd_kernel / flash_fwd_splitkv_kernel
  -> compute_attn() / compute_attn_splitkv()
  -> compute_attn_1rowblock() / compute_attn_1rowblock_splitkv()
```

三份文件的职责不要混淆：

| 文件 | 主要职责 |
| --- | --- |
| `flash_fwd_launch_template.h` | 选择模板布尔值、设置 grid/shared memory、启动 `__global__` kernel |
| `kernel_traits.h` | 定义 tile、warp 数、MMA、拷贝方式、shared-memory layout 和资源大小 |
| `flash_fwd_kernel.h` | 执行 CTA 内部的数据搬运、mask、online softmax、GEMM 和写回 |

真正的 `__global__` 包装位于
[`flash_fwd_launch_template.h`](../minitrain/kernels/cuda_ext/csrc/third_party/flash_attn/src/flash_fwd_launch_template.h)：

```cpp
flash_fwd_kernel(...) {
    compute_attn<...>(params);
}

flash_fwd_splitkv_kernel(...) {
    compute_attn_splitkv<...>(params);
}

flash_fwd_splitkv_combine_kernel(...) {
    combine_attn_seqk_parallel<...>(params);
}
```

因此，文件名虽然叫 `flash_fwd_kernel.h`，其主体实际上是被全局 kernel 调用的内联
`__device__` 算法。

## 3. 文件自身的结构

[`flash_fwd_kernel.h`](../minitrain/kernels/cuda_ext/csrc/third_party/flash_attn/src/flash_fwd_kernel.h)
共约 1294 行，可分成以下六块：

| 行号 | 函数 | 作用 |
| ---: | --- | --- |
| 30 | `get_lse_tile()` | 构造当前 Q tile 对应的 LSE 输出视图 |
| 51 | `compute_attn_1rowblock()` | 普通 forward 的 CTA 内部主体 |
| 498 | `compute_attn_1rowblock_splitkv()` | Split-KV、KV cache 和 append-KV 主体 |
| 1075 | `compute_attn()` | 普通 forward 的 block-index 包装 |
| 1096 | `compute_attn_splitkv()` | Split-KV 的 block-index 包装 |
| 1110 | `combine_attn_seqk_parallel()` | 按 LSE 权重合并多个 KV split |

其中最值得先精读的是 `compute_attn_1rowblock()`。Split-KV 的核心矩阵计算与它相同，只是在
K/V 地址计算、split 范围、KV cache 和输出格式上增加了逻辑。

## 4. 编译期配置：`Kernel_traits` 与布尔模板

### 4.1 `Kernel_traits` 决定什么

`Flash_fwd_kernel_traits` 定义在
[`kernel_traits.h`](../minitrain/kernels/cuda_ext/csrc/third_party/flash_attn/src/kernel_traits.h)。

主要字段包括：

```cpp
kBlockM       // 每个 CTA 处理多少个 Q token
kBlockN       // 每次扫描多少个 K/V token
kHeadDim      // 编译期 head-dim bucket
kNWarps       // CTA warp 数量
kNThreads     // kNWarps * 32
TiledMma      // Tensor Core MMA 的线程与 tile 映射
SmemLayoutQ   // Q 在 shared memory 中的布局
SmemLayoutKV  // K/V 在 shared memory 中的布局
GmemTiledCopyQKV
GmemTiledCopyO
kSmemSize
```

`kHeadDim` 是编译期 bucket，不一定等于实际 `params.d`。例如实际 `d=80` 可以落到更大的已编译
bucket，此时 `Is_even_K=false`，所有涉及列维度的 load/store 都必须使用 predicate 防止越界。

### 4.2 主路径的模板布尔值

`compute_attn_1rowblock` 的模板参数为：

```cpp
Kernel_traits,
Is_dropout,
Is_causal,
Is_local,
Has_alibi,
Is_even_MN,
Is_even_K,
Is_softcap,
Return_softmax
```

| 参数 | 含义 |
| --- | --- |
| `Is_dropout` | 是否生成并应用 attention dropout |
| `Is_causal` | 是否只能看当前位置及之前的 K |
| `Is_local` | 是否启用滑动窗口 attention |
| `Has_alibi` | 是否向 logits 添加 ALiBi bias |
| `Is_even_MN` | Q/K 长度是否完整对齐 tile，且无需 varlen 边界路径 |
| `Is_even_K` | 实际 `d` 是否正好等于编译期 `kHeadDim` |
| `Is_softcap` | 是否对 logits 执行 softcap |
| `Return_softmax` | 是否输出测试用的 softmax/dropout 中间矩阵 |

这些值在 launcher 中通过 `BOOL_SWITCH` 等宏变成编译期常量。核心循环中的无用分支因而可以被编译器删除。

注意 launcher 为控制模板实例数量，会把若干条件合并。例如 local、ALiBi、return-softmax 或较大
head-dim 可能强制走 `Is_even_MN=false` 的通用版本。这意味着 `Is_even_MN=false` 不只表示物理长度
不整齐，也可能是 launcher 主动选择较少实例的结果。

## 5. CUDA 网格：一个 CTA 到底负责什么

普通 forward 的 launcher 使用：

```cpp
num_m_block = ceil_div(seqlen_q, kBlockM);
grid = dim3(num_m_block, batch, num_query_heads);
```

`compute_attn()` 将 block index 解读为：

```cpp
m_block = blockIdx.x;
bidb    = blockIdx.y;
bidh    = blockIdx.z;
```

所以一个 CTA 负责：

```text
一个 batch
  x 一个 query head
  x 连续 kBlockM 个 query token
```

它会独立生成这一块 Q 行的 output 和 LSE。不同 Q tile、batch、head 之间不需要 CTA 间同步。

对于 GQA/MQA，Q head 到 KV head 的映射是：

```cpp
kv_head = bidh / params.h_h_k_ratio;
```

例如 `h=32, h_k=8` 时，`h_h_k_ratio=4`，连续四个 Q head 共用一个 K/V head。

## 6. `get_lse_tile()`：为什么 LSE 布局这么复杂

LSE 是每个 Query 行的：

\[
\operatorname{LSE}_i = \log\sum_j \exp(z_{ij}),
\]

其中 \(z\) 是应用 scale、bias 和 mask 后的 logits。forward 保存 LSE，backward 用它重建概率。

`get_lse_tile()` 需要兼容三种逻辑布局：

1. 普通定长：`[batch, head, seqlen_q]`；
2. unpadded varlen：所有序列的有效 Q 行压成 `total_q`；
3. `seqlenq_ngroups_swapped`：为 Q 长度与 head/group 交换优化而改变 stride。

函数先根据 `BlockInfo` 计算当前 batch 的 Q 起点，再用 CuTe `make_layout()` 和
`local_tile()` 返回当前 `m_block` 的一维 LSE tile。

## 7. `BlockInfo`：统一定长、变长和 KV cache 长度

`BlockInfo` 位于
[`block_info.h`](../minitrain/kernels/cuda_ext/csrc/third_party/flash_attn/src/block_info.h)。

它向 kernel 提供：

```cpp
actual_seqlen_q
actual_seqlen_k
q_offset(...)
k_offset(...)
seqlen_k_cache
```

普通定长路径中，offset 主要由 `batch_stride` 决定；varlen 路径则使用 `cu_seqlens_q/k` 中的累计
长度。Split-KV/缓存路径还会把已有 cache 长度、`seqlen_knew`、left padding 和 `seqused_k`
纳入实际 K 长度。

主 kernel 因此不需要在每个地址表达式中重复区分定长和变长。

## 8. 普通 forward：逐阶段阅读

### 8.1 类型、shared memory 和线程号

函数开头提取：

```cpp
Element       // 通常为 fp16 或 bf16
ElementAccum  // float
index_t       // int64_t
```

动态 shared memory 由 launcher 传入：

```cpp
extern __shared__ char smem_[];
```

随后 `sQ/sK/sV/sO` 都是在这片内存上构造的不同 CuTe view。它们不一定同时存活，epilogue 会复用前面
Q/K/V 已不再需要的 shared memory。

### 8.2 Philox dropout 状态

kernel 从 `params.philox_args` 解包 seed 和 offset，并创建 `Dropout` 对象。

RNG 映射不是依赖“线程执行到第几个随机数”，而是编码了：

- batch；
- head；
- warp lane；
- attention matrix 中的 `16 x 32` 子块位置。

这样 forward 和 backward 即使使用不同线程布局或遍历顺序，仍能重建相同 dropout mask。

第一个 grid block 的线程 0 会在任何 early return 之前保存 RNG seed/offset。否则，如果第一个 Q tile
因空序列提前返回，backward 可能拿不到状态。

### 8.3 计算可访问的 K/V tile 范围

当前 Q tile 的 K/V 范围为：

```text
[n_block_min, n_block_max)
```

普通非 local attention 的 `n_block_min=0`。`n_block_max` 起始值是：

```cpp
ceil_div(actual_seqlen_k, kBlockN)
```

causal/local 路径会进一步裁剪上界，local 路径也会抬高下界。表达式中的：

```cpp
actual_seqlen_k - actual_seqlen_q
```

用于在 Q/K 长度不同的场景中将位置按右侧对齐，这在 KV cache 解码中尤其重要。

如果 `n_block_max <= n_block_min`，当前 Q tile 没有任何合法 K：

- 将有效 `O` 行写为 0；
- 将对应 LSE 写为实现约定的无有效行标志；
- 避免读取越界 K/V；
- 直接返回。

### 8.4 为什么倒序遍历 K/V

主循环从：

```cpp
n_block = n_block_max - 1;
```

开始递减。

原因是最末端的 K tile 最可能包含：

- `seqlen_k` 不是 `kBlockN` 整数倍产生的尾部；
- causal 对角线边界；
- local window 右边界。

先处理末端复杂 tile，之后可以进入条件更少的内部快速循环。倒序还使代码只维护一个 `n_block`
变量，不必同时保留额外的循环上界状态。

## 9. CuTe Tensor 应该怎样读

### 9.1 命名前缀

这份代码常用以下命名规则：

| 前缀 | 含义 |
| --- | --- |
| `m` | 完整逻辑矩阵 view，如 `mQ` |
| `g` | global-memory tile，如 `gQ` |
| `s` | shared-memory tensor，如 `sQ` |
| `r` | register fragment，如 `rP` |
| `acc` | FP32 accumulator，如 `acc_s`、`acc_o` |
| `t` | 某个 thread slice 看到的 partition |
| `c` | identity coordinate tensor |
| `p` | predicate tensor |

名称通常同时编码 source 和 destination。例如：

```cpp
tQgQ  // Q copy 中，当前线程负责的 global Q source
tQsQ  // Q copy 中，当前线程负责的 shared Q destination
tKgK  // K copy 中的 global K source
tKsK  // K copy 中的 shared K destination
tOgO  // O copy 中的 global O destination
```

### 9.2 `make_tensor` 与 `local_tile`

`make_tensor(pointer, layout)` 只创建 view，不搬运数据。

例如 `mQ` 的逻辑 shape 为：

```text
[actual_seqlen_q, num_heads, head_dim]
```

stride 来自 `params.q_row_stride/q_head_stride`。因此 MiniTrain 可以把外层不连续但最后一维连续的
Q/K/V 直接描述给 kernel，无需先 transpose/copy。

`local_tile()` 再从逻辑矩阵中取得当前 CTA 的：

```text
gQ: [kBlockM, kHeadDim]
gK: [kBlockN, kHeadDim, nblocksN]
gV: [kBlockN, kHeadDim, nblocksN]
```

### 9.3 `partition_S` 与 `partition_D`

`partition_S` 表示按 tiled-copy/MMA 规则切分 source，`partition_D` 表示按相同规则切分 destination。
它们回答的是：

> 当前线程负责 source/destination 的哪些逻辑坐标？

它们本身不执行 load/store。真正的数据移动发生在后面的 `copy(...)`。

### 9.4 identity tensor 与 predicate

`cQ/cKV/cO` 是 identity coordinate tensor：每个元素保存自己的逻辑 `(row, col)` 坐标。

对它执行与数据完全相同的 partition 后，kernel 就能知道当前线程持有元素的真实列号，从而构造：

```cpp
col < params.d
```

这样的 predicate。该方法比手工推导每个 lane 的坐标更稳健，也能适应不同 tiled-copy layout。

## 10. Shared-memory 布局与复用

普通路径构造：

```cpp
sQ  // [kBlockM, kHeadDim]
sK  // [kBlockN, kHeadDim]
sV  // [kBlockN, kHeadDim]
sVt // 与 sV 同一地址的转置逻辑 view
```

`sVt` 不代表执行了一次物理 transpose；它是对同一 shared-memory 数据的另一种 layout 解释，用于满足
`P x V` MMA 的 operand 布局。

shared-memory layout 带有 swizzle，目的是降低 Tensor Core `ldmatrix` 访问时的 bank conflict。

当 `Share_Q_K_smem=true` 时，Q 和 K 复用地址。此时必须先：

1. 等待 Q load 完成；
2. 将 Q 从 shared memory 搬到寄存器 fragment；
3. `__syncthreads()`；
4. 才允许 K 覆盖原来的 Q 空间。

如果移动或删除这些同步点，就可能出现尚未读完 Q、K 已经覆盖它的 race。

## 11. Prologue 与异步流水

在 SM80+，Q/K/V 的 global-to-shared copy 通常由 `cp.async` 实现。

几个同步原语的含义是：

```cpp
cp_async_fence()  // 提交当前异步 copy group
cp_async_wait<N>()// 等到未完成 group 数量不超过 N
__syncthreads()   // CTA 内线程对 shared memory 达成一致
```

prologue 先启动 Q load，再启动第一个 K tile load。若 `Is_Q_in_regs=true`，Q 会在合适的等待点被进一步
搬进寄存器，从而在扫描所有 K tile 时复用。

主循环则大体形成：

```text
等待当前 K
  -> 启动当前 V load
  -> 计算 QK
  -> 等待当前 V
  -> 启动下一块 K load
  -> online softmax
  -> 计算 PV
  -> 下一轮
```

加载和矩阵计算因此可以部分重叠。

特别需要注意，源文件明确要求“下一块 K 的 `cp_async_fence()` 必须位于条件内部”。最后一轮没有下一块
K，若仍无条件提交空 group，会改变后续 `wait<N>` 对 group 数的假设，造成同步错误甚至 race。

## 12. 第一次 GEMM：`QK^T`

每轮创建 FP32 score accumulator：

```cpp
acc_s: [kBlockM, kBlockN]
```

随后执行：

```cpp
gemm(acc_s, Q, K, ...);
```

输入 Q/K 通常是 FP16/BF16，Tensor Core 使用 FP32 累积。逻辑结果是：

\[
S_b = Q_{tile}K_b^T.
\]

如果启用 softcap，会在 mask 和 softmax 前修改 logits。随后 `Mask` 统一处理：

- causal；
- local window；
- 非整齐 N 尾部；
- ALiBi bias。

无效位置最终以负无穷进入 softmax，使其指数权重为 0。

## 13. Online softmax：FlashAttention 的数学核心

### 13.1 维护的状态

对 Q tile 的每一行，`Softmax` 保存：

```cpp
row_max
row_sum
```

同时 `acc_o` 保存尚未做最终除法的输出分子。

假设已处理部分 K block，对某行有：

\[
m_{old}=\max z_{old},
\]

\[
l_{old}=\sum_j e^{z_j-m_{old}},
\]

\[
O_{old}=\sum_j e^{z_j-m_{old}}V_j.
\]

### 13.2 加入新 block

新 block 的最大值为 \(m_b\)，合并后的最大值：

\[
m_{new}=\max(m_{old},m_b).
\]

为了换到新的数值基准，旧状态必须乘：

\[
r=e^{m_{old}-m_{new}}.
\]

于是：

\[
l_{new}=r\,l_{old}+\sum_{j\in b}e^{z_j-m_{new}},
\]

\[
O_{new}=r\,O_{old}+\sum_{j\in b}e^{z_j-m_{new}}V_j.
\]

这正是 `softmax_rescale_o()` 的工作：更新最大值、缩放已有 `row_sum` 和 `acc_o`，再把当前
`acc_s` 原地转换成指数权重。

### 13.3 为什么使用 `scale_softmax_log2`

实现使用 `exp2`，因此把自然指数转换为：

\[
e^x=2^{x\log_2e}.
\]

`params.scale_softmax_log2` 等价于：

\[
scale\_softmax\cdot\log_2e.
\]

`row_max` 保留的是 scale 之前的 score 最大值；最终 LSE 使用：

\[
LSE = row\_max\cdot scale\_softmax + \log(row\_sum).
\]

### 13.4 最终归一化

扫描结束后，`normalize_softmax_lse()` 完成：

\[
O\leftarrow O/row\_sum,
\]

并产生每行 LSE。若启用 dropout，输出还要乘保留概率的倒数 `rp_dropout`。

因此 kernel 在整个扫描过程中无需存储完整概率矩阵，最终结果仍是精确 softmax，而不是分块近似。

## 14. Masking loop 与 fast loop 为什么分开

普通主路径有两段循环。

第一段是固定、编译期展开的 `n_masking_steps`，处理最靠近末端的复杂 tile：

- K 尾部不足 `kBlockN`；
- causal 对角线穿过 tile；
- local window 边界穿过 tile。

第二段处理剩余内部 tile。它们不需要 causal/tail 的完整判断，因此走更简单的路径。

这种拆分让大量规则 tile 避免承担少数边界 tile 的控制流成本。模板参数又会把不适用的 mask 分支进一步
编译删除。

## 15. Dropout 与 `Return_softmax`

在线 softmax 后，`acc_s` 中已经是当前 block 的指数权重。代码先将其转换为 `Element` 类型，再可选地应用
dropout。

dropout 只作用于进入 `P x V` 的权重；最终 `normalize_softmax_lse()` 使用未 dropout 的 `row_sum`
做 softmax 归一化，再通过 `rp_dropout=1/(1-p)` 保持期望不变。

`Return_softmax` 是 correctness/debug 路径。它会额外向 `params.p_ptr` 写出中间矩阵，并把 dropout
keep/drop 信息编码在符号位中，供测试恢复真实 mask。该路径：

- 会物化二次方规模数据；
- 会增加寄存器和显存压力；
- 不是正常训练 forward 的输出接口；
- 不应被当作通用的最终归一化 attention probability API。

## 16. 第二次 GEMM：`P x V`

softmax 权重转换为 FP16/BF16 后，代码把 accumulator layout 重新解释为适合 MMA A operand 的布局：

```cpp
tOrP = convert_layout_acc_Aregs(...);
```

然后执行：

```cpp
gemm_rs(acc_o, P, V, ...);
```

`rs` 可理解为 register-shared：

- P 在寄存器；
- V 从 shared memory 读取；
- O 在 FP32 寄存器中累积。

逻辑上是：

\[
acc\_o \mathrel{+}=P_bV_b.
\]

前一步 online softmax 已在必要时重缩放旧 `acc_o`，所以所有 K/V block 的贡献始终处于同一个指数基准。

## 17. Epilogue：为什么先写 shared memory

主循环结束后：

1. 完成 softmax 归一化并计算 LSE；
2. 将 FP32 `acc_o` 转为 FP16/BF16；
3. 将 Tensor Core accumulator 的线程分散布局写入 `sO`；
4. 从 `sO` 以更适合 global store 的 tiled-copy 布局读出；
5. 写入 `params.o_ptr`；
6. 写入 `params.softmax_lse_ptr`。

中间经过 shared memory 的原因是 Tensor Core C accumulator 在各 lane 中的布局并不是最终输出矩阵的连续
行布局。shared memory 在这里承担一次 CTA 内重排，使 global store 可以更连续、向量化。

写回阶段仍使用两类 predicate：

- 行 predicate：最后一个 Q tile 不能超过 `actual_seqlen_q`；
- 列 predicate：实际 `params.d` 小于 `kHeadDim` bucket 时不能写 padding 列。

注释中的 `Clear_OOB=false` 很重要：写回时越界位置必须“不写”，而不是向相邻或 padding 区域写 0。

## 18. 普通路径完整数据流

```text
                         global memory
                Q             K             V
                |             |             |
                +------ cp.async / tiled copy ------+
                              |
                         shared memory
                     sQ      sK      sV/sVt
                      |       |        |
                      +-- QK^T MMA ----+
                              |
                     acc_s (FP32 registers)
                              |
                  softcap / mask / ALiBi
                              |
                       online softmax
                    row_max + row_sum
                              |
                    rP (FP16/BF16 regs)
                              |         sV
                              +-- PV MMA --+
                                           |
                                 acc_o (FP32 regs)
                                           |
                              rescale across K blocks
                                           |
                              final normalize + LSE
                                           |
                              sO reorder in shared mem
                                           |
                                  O + LSE to global
```

## 19. Split-KV 路径

### 19.1 为什么需要 Split-KV

普通路径中，一个 Q tile 的全部 K/V block 由一个 CTA 扫描。解码时 Q 很短而 KV cache 很长，grid 中
Q tile 数量可能太少，GPU 并行度不足。

Split-KV 将 K 方向切成多个区间：

```text
同一个 Q tile
  +-- CTA split 0: K blocks [0, a)
  +-- CTA split 1: K blocks [a, b)
  +-- CTA split 2: K blocks [b, c)
```

每个 split 独立执行 online softmax，并输出局部 `Oaccum` 和 `LSEaccum`。第二个 kernel 再将它们合并。

### 19.2 Split-KV 的 grid 映射

`compute_attn_splitkv()` 中：

```cpp
m_block    = blockIdx.x;
n_split_idx= blockIdx.y;
bidb/bidh  = blockIdx.z 中解码；
```

当 `Split=false` 时，它也可作为不拆分但支持 KV cache/append-KV 的路径使用。

### 19.3 每个 split 的 K 范围

先计算：

```cpp
n_blocks_per_split = ceil_div(total_k_blocks, num_n_splits);
```

第 `s` 个 split 的基础范围是：

```text
[s * n_blocks_per_split,
 (s + 1) * n_blocks_per_split)
```

之后再与实际 K 长度、causal/local 可见范围取交集。

空 split 必须写：

- `Oaccum = 0`；
- `LSEaccum = -infinity`。

否则 combine kernel 会把未初始化数据当成有效局部结果。

## 20. KV cache、paged cache 与 append-KV

### 20.1 连续 KV cache

当 `block_table == nullptr` 时，K/V block 在逻辑和物理上连续。倒序遍历下一块只需将数据指针减少：

```text
kBlockN * row_stride
```

### 20.2 Paged KV cache

当 `block_table != nullptr` 时，逻辑 token 页映射到不连续的物理 cache page。

kernel 根据：

```cpp
logical_position / page_block_size
logical_position % page_block_size
```

得到逻辑页号和页内偏移，再用 `block_table` 找到物理页。由于循环倒序扫描，代码在相邻 K/V block
之间计算物理 page 与 offset 的差值，并直接更新 CuTe tensor 的底层 data pointer。

此处不能假设相邻逻辑 block 在显存中相邻。

### 20.3 Append KV

当 `Append_KV=true` 时，新产生的 `knew/vnew` 会追加到已有 cache。kernel 需要：

1. 找到 cache 的逻辑追加位置；
2. 处理连续或 paged cache 的物理地址；
3. 可选地对新 K 应用 rotary embedding；
4. 将新 K/V 写入 cache；
5. 同步后再让 attention 主循环读取更新后的值。

写 cache 后的 `__syncthreads()` 是可见性要求，不只是性能上的流水选择。

## 21. Rotary embedding 路径

Split-KV/append-KV 路径支持两种 rotary 布局：

- interleaved：偶数/奇数通道成对旋转；
- contiguous：前半维和后半维成对旋转。

当 `rotary_dim > 0` 时，Q 和新 K 会根据 cache 中的位置读取 cos/sin。

causal/local 场景中，不同 Q 行使用递增位置；非 causal 场景可以通过将 cos/sin 的 row stride 设为 0，
让一个 Q tile 中所有行复用同一位置。

rotary 只作用于前 `rotary_dim` 个通道，其余 head dimension 原样复制。

## 22. Split 结果为什么不能直接相加

第 `s` 个 split 有自己的局部归一化：

\[
L_s=\log\sum_{j\in s}e^{z_j},
\]

\[
O_s=\frac{\sum_{j\in s}e^{z_j}V_j}{e^{L_s}}.
\]

全局 LSE 为：

\[
L=\log\sum_s e^{L_s}.
\]

因此全局输出是：

\[
O=\sum_s e^{L_s-L}O_s.
\]

`combine_attn_seqk_parallel()` 正是按这个公式工作：

1. 把各 split 的 LSE 读入 shared memory；
2. 在 split 维做稳定 logsumexp；
3. 计算每个 split 的权重 `exp(L_s - L)`；
4. 读取每个 `Oaccum_s`；
5. 加权累积；
6. 转换到最终 Element 类型并写回 O。

当所有局部 LSE 都无效时，代码有专门的 infinity/NaN 处理，避免出现 `-inf - (-inf)` 导致 NaN。

## 23. 三条设备端路径的关系

```text
普通训练/批量 attention
  flash_fwd_kernel
    -> compute_attn
      -> compute_attn_1rowblock
        -> 直接写 O、LSE

Split-KV / cache attention
  flash_fwd_splitkv_kernel
    -> compute_attn_splitkv
      -> compute_attn_1rowblock_splitkv
        -> 写局部 Oaccum、LSEaccum
           |
           v
  flash_fwd_splitkv_combine_kernel
    -> combine_attn_seqk_parallel
      -> 写最终 O、LSE
```

普通路径支持 dropout 和测试用 `Return_softmax`；Split-KV 路径主要面向推理/cache，不包含普通训练路径的
dropout 逻辑。

## 24. 容易误读或误改的地方

### 24.1 `acc_s` 不一直是原始 score

`QK^T` 后它是 FP32 logits；`softmax_rescale_o()` 后同一块存储被原地改为指数权重。看到后面的
`convert_type<Element>(acc_s)` 时，转换的是 P 的分块表示，不是原始 score。

### 24.2 `sVt` 不等于真的转置了一份 V

它是 `sV` 同一地址的另一种 layout view。修改它的 layout 或底层 swizzle 必须同时理解 MMA operand
的访问方式。

### 24.3 `Is_even_K` 说的是 bucket 尾部

它不是数学矩阵 K，也不是 key length；这里的 K 维是 GEMM/head dimension。`Is_even_MN` 才与
Q/K sequence tile 边界相关。

### 24.4 mask 的坐标包含 warp/lane 映射

`m_block * kBlockM + ...` 中后半部分来自 MMA accumulator 在 warp/lane 内的行坐标。不能简单用
`threadIdx.x` 当作 Query 行号。

### 24.5 `__syncthreads()` 与 `cp_async_wait()` 不是互相替代

`cp_async_wait()` 等待异步 copy group，`__syncthreads()` 保证 CTA 内线程都到达并能一致访问 shared
memory。通常两者需要配合，删除任意一个都可能留下数据依赖问题。

### 24.6 空行的 LSE 约定因路径而异

普通最终输出和 Split 局部输出对“无有效 key”的 LSE 使用不同 infinity 约定，以保证后续 backward
或 combine 不产生 NaN。修改时应连同 `softmax.h` 的 `normalize_softmax_lse<Split>()` 一起检查。

### 24.7 模板常量影响正确性路径，不只是性能

`Is_even_MN/Is_even_K` 决定是否生成边界 predicate；错误地把它们设为 true 会导致越界访问，而不只是
得到一个更快或更慢的 kernel。

## 25. 推荐的源码精读顺序

第一次阅读建议按以下顺序：

1. `flash_fwd_launch_template.h` 第 28–99 行：看 kernel 包装、grid 和模板开关；
2. `kernel_traits.h` 的 `Flash_fwd_kernel_traits`：看 tile、warp、MMA 与 shared memory；
3. `flash_fwd_kernel.h` 第 1075–1092 行：确认 grid index 映射；
4. 第 51–288 行：看边界、tensor view、partition 和 prologue；
5. 第 290–429 行：看两段 K/V 循环；
6. 第 431–494 行：看归一化、LSE 和输出重排；
7. `softmax.h` 的 `Softmax`：核对 online softmax 状态更新；
8. `mask.h`、`dropout.h`：分别理解坐标 mask 与 Philox 映射；
9. 第 498–1071 行：在普通路径基础上看 Split-KV/cache 差异；
10. 第 1110–1292 行：看 split 的 LSE 加权合并。

不要一开始就尝试展开每一个 CuTe layout。先明确“这个 view 位于哪里、shape 是什么、作为 source 还是
destination”，再追踪 lane 级 layout，阅读成本会低很多。

## 26. 调试时建议观察什么

如果要验证或修改这份 kernel，建议分层检查：

1. **逻辑范围**：打印 `m_block/n_block_min/n_block_max/actual_seqlen_q/k`；
2. **地址映射**：检查 batch/head/GQA、varlen offset 和 paged block-table；
3. **边界 predicate**：分别覆盖 M tail、N tail、head-dim tail；
4. **数值状态**：小 shape 下观察 `row_max/row_sum/LSE`；
5. **功能组合**：causal、local、ALiBi、dropout、softcap 不要只测单一路径；
6. **同步正确性**：修改 pipeline 后使用 `compute-sanitizer` 检查 race/OOB；
7. **资源变化**：查看 ptxas registers、spill 和 dynamic shared memory；
8. **性能**：最后再比较 kernel time、occupancy 和显存流量。

`Return_softmax` 适合小 shape correctness，不适合性能结论。模板或 shared-memory 改动还应分别检查
SM 架构，因为寄存器压力、可用 shared memory 和 occupancy 会变化。

## 27. 总结

这份文件的核心可以浓缩为四句话：

1. 一个 CTA 固定负责一个 batch、一个 Q head 和一个 `kBlockM` Query tile；
2. CTA 倒序流式读取 `kBlockN` 大小的 K/V tile，不物化完整 attention matrix；
3. `row_max + row_sum + acc_o` 构成 online softmax 状态，使不同 K block 能稳定、精确地合并；
4. Split-KV 将 K 方向扩展到多个 CTA，再用局部 LSE 对局部输出做数学上正确的加权合并。

从工程角度看，它的性能来自四部分共同作用：Tensor Core MMA、异步 global-to-shared 流水、
shared-memory swizzle/复用，以及用模板常量移除运行时分支。也正因为如此，修改任何 tile、layout、同步或
边界模板时，都必须同时考虑数值公式、地址映射和 GPU pipeline。
