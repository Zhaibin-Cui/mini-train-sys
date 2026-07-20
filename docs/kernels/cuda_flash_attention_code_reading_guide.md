# CUDA FlashAttention 代码阅读指南

这份指南只回答三个问题：一次调用经过哪些层、每层负责什么、修改时应该停在哪个
边界。编译和运行命令集中在
[`cuda_ext/README.md`](../../minitrain/kernels/cuda_ext/README.md)，spill 数据集中在
[`cuda_flash_attention_sm86_spill_analysis.md`](cuda_flash_attention_sm86_spill_analysis.md)。

## 1. 先看调用链

从训练代码到 CUDA kernel 的路径是：

```text
model
  -> get_ops_backend("cuda")
  -> CudaOpsBackend.attention()
  -> flash_attention()
  -> MiniTrainCudaFlashAttentionFunction.apply()
  -> _C.forward() / _C.backward()
  -> flash_api_upstream.cpp
  -> run_mha_fwd() / run_mha_bwd()
  -> 显式实例化的 FlashAttention kernel
```

建议按下面的顺序读，不要一开始钻进 CUTLASS 模板：

1. `minitrain/model/ops.py`
2. `minitrain/kernels/cuda_ext/__init__.py`
3. `minitrain/kernels/cuda_ext/flash_attention.py`
4. `minitrain/kernels/cuda_ext/build.py`
5. `minitrain/kernels/cuda_ext/csrc/flash_api_upstream.cpp`
6. 任意一个 `csrc/instantiations/*.cu`
7. `csrc/third_party/flash_attn/src/flash_*_launch_template.h`
8. `flash_fwd_kernel.h`、`flash_bwd_kernel.h`、`softmax.h`、`dropout.h`

前五步能解释 MiniTrain 的全部封装。后面三步才是上游设备端算法。

## 2. Python 层

### Backend 与 fallback

`CudaOpsBackend` 继承已有 backend，并只覆盖 attention。支持的输入进入 CUDA
extension；不支持的输入沿继承关系继续走 Triton，最后才是 PyTorch。这里没有按
sequence length 或性能做动态 fallback。

`is_flash_attention_supported()` 在加载扩展前检查：

- tensor 必须在 CUDA 上，head-dim 最后一维必须连续；外层 stride 可以不同；
- Q/K/V shape、dtype 和 device 必须一致；
- batch、head 和 sequence 维度必须为正，GPU 必须是 sm80 或更新架构；
- dtype 和 head bucket 必须包含在当前 build profile；
- head dim 必须是 8 的倍数且不超过 256；
- `D > 192 + dropout` 需要至少 144 KiB opt-in block shared memory；sm86/sm89
  因此不进入 CUDA backward。

最后一条是共享内存硬件约束。它不是对 spill 或 tile 性能的判断。
Python predicate 和 C++ pybind 边界都执行这些结构性检查：前者保护
`CUDA -> Triton -> PyTorch` fallback，后者保护直接 extension 调用。
Python 优先读取 PyTorch 设备属性；旧版 PyTorch 缺少该字段时使用已审计架构表，
未知架构在验证前按 unsupported 处理。C++ 始终读取 CUDA runtime 的真实属性。

Dropout 还有一层数值规范化。Python 参数最初是 double，但上游参数结构和 kernel
使用 float32；支持判断、autograd context 和 C++ RNG 分支都使用同一个 float32
值。下溢到 0 的概率等价于 no-dropout，舍入到 1.0 的概率直接拒绝。

### Autograd 保存什么

`MiniTrainCudaFlashAttentionFunction.forward()` 调用 C++，并保存 backward 所需的
Q、K、V、输出、softmax LSE 和 Philox RNG state。生产路径不会保存完整的
`S x S` attention matrix 或 dropout mask。

`backward()` 把保存的状态交回 C++。上游 kernel 根据 forward 的 Philox
seed/offset 重建同一个 dropout mask，因此不需要额外显存保存 mask。

`flash_attention_dropout_mask_for_testing()` 是 correctness 专用入口。它打开上游
`return_softmax` 调试分支，从返回矩阵的符号位恢复 keep mask。该入口会物化
`S x S` 数据，只应在小 shape 测试中使用。

## 3. 编译层

### Profile 如何变成源文件

`build.py` 先解析 profile：

```text
minimal     -> fp16 x D32
workstation -> fp16/bf16 x D32/D64/D128
full        -> fp16/bf16 x D32/D64/D96/D128/D192/D256
```

每个 dtype/head bucket 还要乘以：

```text
forward/backward x causal/non-causal
```

因此 full profile 一共有 `2 x 6 x 2 x 2 = 48` 个 `.cu`。dropout 不再增加文件
数量，因为每个文件内部通过上游 `DROPOUT_SWITCH` 同时实例化 dropout 和
no-dropout 路径。

已编译 object 中可以同时看到 `Is_dropout=false/true` 的独立符号。对相同 D128
fp16 tile 导出 SASS 后，no-dropout forward/backward 都完全没有 Philox4x32 固定
常量，而 dropout 对照组明确包含这些常量。这证明随机数逻辑是编译期消除，不是
hot loop 中的运行时 `if`。两者总 `REG` 仍可能同为 255，因为 attention 主计算本身
占用大量寄存器。

`generate_kernels.py` 只生成薄模板实例化文件。以 fp16、D=128、causal forward
为例，它做三件事：定义 dtype 和 causal 常量、包含上游 launch template、显式
实例化目标函数。kernel 算法仍在 `third_party/flash_attn`。

### JIT cache 为什么不会串配置

profile、SM 架构、head bucket、dtype 和上游版本共同生成 cache key。PyTorch
extension 名称和 build 目录都带这个 hash，因此 minimal、workstation、full 或
不同架构不会加载到错误的 DLL。

`MINITRAIN_CUDA_ARCHS` 是架构的唯一入口。loader 在调用 PyTorch JIT 前用它生成并
覆盖 `TORCH_CUDA_ARCH_LIST`，因此 shell 中残留的 PyTorch 变量不会让同一个 cache
名称实际编译出另一套 cubin。

所选架构中最后一个同时保留 PTX。例如 `80;86;90` 会生成对应 cubin，并为 90
保留 PTX，供兼容的新架构做 forward JIT。这里的 forward 指 PTX 前向兼容，不是
attention forward。

### Wheel 为什么仍然需要 nvcc

`pyproject.toml` 把 `cuda_ext/csrc/**/*` 作为 package data。Wheel 携带生成 `.cu`、
adapter、FlashAttention/CUTLASS header 和 license，但明确不携带本机 build cache。
安装后 `build.py` 从安装目录定位这些源文件，再在目标机器首次调用时 JIT 编译。
因此 wheel 的 `py3-none-any` 标签不代表 CUDA kernel 已经跨平台预编译。

源码 checkout 的 build cache 留在 `cuda_ext/build`。Wheel 安装则写入用户 Torch
cache，并用 Python/PyTorch/CUDA/platform ABI hash 分目录，避免只读
`site-packages` 和跨环境 `.pyd` 冲突。`MINITRAIN_CUDA_BUILD_ROOT` 可覆盖根目录。

## 4. C++ 适配层

`flash_api_upstream.cpp` 是最值得精读的文件。它不是另写一套 FlashAttention，
而是把 PyTorch tensor 和上游 `Flash_fwd_params` / `Flash_bwd_params` 接起来。

### 输入与 stride

MiniTrain 的张量布局是 `(B,H,S,D)`，上游参数结构允许显式传 stride，因此不需要
transpose 或复制成另一种布局。适配层把 batch/head/row stride 逐项写入参数，
并检查 contiguous、dtype、device、shape 和 head bucket。

### Forward

forward 的主要步骤是：

1. 分配输出和 LSE；
2. 填充 Q/K/V 指针、shape、stride 和 scale；
3. dropout 时从当前 CUDA generator 取得 Philox seed/offset；
4. 根据 dtype 和 head bucket dispatch 到已编译实例；
5. 返回 output、LSE 和 backward 所需 RNG state。

causal、dropout、even-N 等条件在上游 launch template 中继续变成编译期分支。

### Backward

backward 接收 `dout` 和 forward 保存的状态，分配 `dq/dk/dv` 及上游 workspace，
然后进入同一套 dtype/head bucket dispatch。`D > 192 + dropout` 会先查询设备的
`cudaDevAttrMaxSharedMemoryPerBlockOptin`，不足 144 KiB 时直接报错，防止绕过
Python 支持检查的 pybind 调用得到未初始化梯度。

Autograd 的 `output.sum().backward()` 可能传入 expanded/stride-0 `dout`。C++
adapter 会只在这种情况下生成 contiguous 副本；普通最后一维连续的梯度保持
零拷贝。C++ 还独立检查 Q/K/V、saved output、LSE 和 dropout RNG state 的
device、dtype、shape 与布局，避免 direct pybind 绕过 Python 后传入异设备指针。

## 5. 上游 kernel 怎么读

先读 launch template，再读 device kernel。

`flash_fwd_launch_template.h` 和 `flash_bwd_launch_template.h` 决定：

- head bucket 对应的 block tile；
- warp 数量；
- shared memory 大小；
- causal、dropout、even-tail 等模板布尔值；
- 最终 launch 哪个 kernel 实例。

设备端主循环可以概括为：

```text
Q tile 载入 shared memory
  -> QK^T tensor-core matmul
  -> online softmax 更新 max/sum
  -> causal 或边界 mask
  -> 可选 Philox dropout
  -> probability x V matmul
  -> 写回 output 与 LSE
```

backward 不恢复完整 attention matrix，而是利用 Q/K/V、output、LSE、dout 和 RNG
state 分块重算需要的概率，再累计 dQ、dK、dV。这是 FlashAttention 用计算换显存
的核心。

MiniTrain 保留了上游 tile 和 register/shared-memory 权衡。看到 ptxas spill 时，
先确认模板完整名称和运行时数据；不要直接加 `--maxrregcount`，也不要只为 sm86
改写全架构共用的上游 traits。

## 6. 用一个调用串起来

假设输入是 fp16 `(B,H,S,D=128)`、causal、dropout=0：

1. backend 选择 CUDA；
2. Python 支持检查找到 D128 bucket；
3. autograd forward 加载当前 profile 对应的 extension；
4. C++ 设置 `(B,H,S,D)` stride，dropout=0 时不申请 Philox state；
5. dispatch 进入 fp16、D128、causal forward 实例；
6. 上游 no-dropout 编译分支执行；
7. backward 使用保存的 Q/K/V、output 和 LSE 进入匹配的 D128 实例。

如果改成 dropout=0.25，第 4 步会取得 Philox seed/offset，forward 与 backward
各自在 kernel 内生成同一 mask。如果改成 sm86、D=256、dropout=0.25，支持检查
会在加载 extension 前拒绝 CUDA 路径。

## 7. 验证入口

`tests/operator_bench.ipynb` 是 CUDA correctness 与性能的统一入口。

notebook 的 CUDA correctness 固定使用 fp16，并遍历：

```text
D = 32, 40, 64, 80, 96, 128, 160, 192, 200, 256
causal = false, true
dropout = 0, 0.25
```

其中 6 个 bucket 边界检查主模板，D40/D80/D160/D200 检查向上选择 bucket 后的
uneven-K masked load/store。

每个支持的分支都用显式 fp32 attention 公式比较 forward、dq、dk、dv。dropout
参考结果使用 CUDA helper 读出的真实 keep mask，不假设 PyTorch SDPA 与上游
Philox 消耗方式相同；同时检查 no-dropout 前后 RNG state 不变、dropout 会推进
generator。另一张 correctness 表使用外层非连续、最后一维连续的 `(B,H,S,D)`
输入，并在显式非默认 CUDA stream 中执行 forward、backward 和 mask helper，检查
stride 透传与当前 stream 接线。性能部分继续使用 notebook 原有的统一 benchmark
封装，fp16 输入下比较 Torch、Triton 和 CUDA 的延迟、峰值显存与 speedup。

## 8. 修改边界

常规集成修改应落在以下位置：

- 增减编译矩阵：`build.py` 和 `generate_kernels.py`；
- 改框架支持条件：`flash_attention.py`；
- 改张量/API 适配：`flash_api_upstream.cpp`；
- 加测试或 benchmark：`tests/operator_bench.ipynb`。

除非明确进行独立的上游版本升级或架构调优，不要修改
`csrc/third_party/flash_attn` 和 `csrc/third_party/cutlass`。这样可以继续用文件
hash 对照上游，定位问题时也能清楚区分“kernel 本身”和“MiniTrain 封装”。当前
vendored 的 19 个 FlashAttention header 已逐个验证为 19/19 SHA256 完全一致。
