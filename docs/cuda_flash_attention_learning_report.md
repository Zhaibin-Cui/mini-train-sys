# MiniTrain CUDA FlashAttention 学习报告

## 1. 这次迁移做了什么

MiniTrain 新增了一个 CUDA-only FlashAttention 后端。核心 forward/backward
算法直接采用 FlashAttention 2.8.4 的 Ampere 实现，MiniTrain 负责以下工程层：

- 接入现有 OpsBackend，并保持 `CUDA -> Triton -> PyTorch` 能力 fallback；
- 把 `(B,H,S,D)` PyTorch tensor 映射到上游参数结构；
- 按 dtype、head bucket、causal 和 forward/backward 生成显式 `.cu`；
- 提供适合本机与服务器的编译 profile、架构选择和 JIT cache；
- 在 notebook 中统一做 fp16 correctness 与性能验证。

迁移没有重新设计 CUDA kernel，也没有为了 sm86 的 spill 改写上游 tile。当前
vendored 的 19 个 FlashAttention kernel、launcher 与 helper header 已逐个和源仓库
比较，19/19 的 SHA256 完全一致。MiniTrain 的改动集中在外围适配层，这使以后升级
上游版本时仍然可以直接比较差异。

## 2. 为什么使用显式实例化矩阵

FlashAttention 的性能依赖大量编译期条件。把所有条件塞进一个运行时 kernel 会
失去上游针对 tile、warp、causal 和 dropout 的专门化。因此迁移沿用源仓库思路，
生成以下矩阵：

```text
direction = forward, backward
dtype     = fp16, bf16
head dim  = 32, 64, 96, 128, 192, 256
causal    = false, true
```

总数是 `2 x 2 x 6 x 2 = 48` 个 `.cu`。这些文件只包含模板参数和显式实例化，
真正的设备端算法仍在 vendored FlashAttention header 中。

dropout 没有把文件数再乘二，因为上游 `DROPOUT_SWITCH` 在每个 translation unit
内部生成 dropout/no-dropout 两棵独立模板。`dropout_p=0` 会进入编译期
`Is_dropout=false` 路径，不读取 Philox state，也不生成随机数或保留 dropout
mask。

这一点还做了二进制级核对。对 D128 fp16、相同 tile/even 条件的成对 kernel 导出
SASS：forward no-dropout 为 2568 条指令，Philox4x32 的四个固定常量出现 0 次；
dropout 为 2944 条，常量共出现 54 次。Backward 对应为 1752/1960 条，no-dropout
仍是 0 次，dropout 为 18 次。因此随机数路径确实被编译消除，而不是运行时跳过。
两类 kernel 的 ptxas `REG` 都可能达到 255，因为主体 QK/softmax/PV 计算本身仍有
很高寄存器压力；不能把“没有 dropout 专属寄存器状态”误读成总寄存器数必然下降。

## 3. 一次调用如何运行

训练侧选择 `backend.ops=cuda` 后，调用依次经过：

```text
CudaOpsBackend.attention
  -> Python 支持检查
  -> autograd Function
  -> PyTorch C++/CUDA extension
  -> dtype/head bucket dispatch
  -> upstream launch template
  -> CUDA kernel
```

Python 层先判断 device、shape、dtype、head dim、当前编译矩阵和硬件能力。条件不
满足时不会加载 CUDA extension，而是继续调用继承的 Triton/PyTorch 实现。这里
只做能力 fallback，不按 sequence length 或 benchmark 结果做性能 fallback。

C++ 适配层利用上游参数结构的显式 stride 支持，直接描述 MiniTrain 的
`(B,H,S,D)` 布局，不需要 transpose 或 Q/K/V 数据复制。head dim 可以是 8 的
倍数，运行时映射到不小于它的最小已编译 bucket。

## 4. Forward、dropout 与 backward

Forward 按 tile 读取 Q/K/V，计算分块 `QK^T`，通过 online softmax 维护每行 max
与归一化和，再计算 probability 与 V 的乘积。causal mask 和越界 tail 都在上游
模板中专门化，最终只写 output 与每行 softmax LSE，不物化完整 attention matrix。

dropout 开启时，C++ 从当前 CUDA generator 获取 Philox seed/offset。Forward 和
backward 保存/复用同一 RNG state，backward 在 kernel 内重新生成 mask。生产
路径因此不保存 `S x S` mask。

Backward 使用 Q/K/V、forward output、LSE、dout 和 RNG state 分块重算概率，
然后累计 dQ、dK、dV。这是 FlashAttention 以有限重计算换取线性显存占用的核心。

测试 helper 会临时打开上游 `return_softmax` 调试分支，通过返回矩阵符号位提取
真实 keep mask。它只用于小 shape correctness，不进入训练路径。

## 5. 硬件与 size 分支

运行时没有手写一串 shape 特判。head dim 首先映射到已编译 bucket，再由上游
launch template 根据架构资源和模板布尔值选择 tile、warp、shared memory、
even-tail、causal 和 dropout 实例。

需要单独保护的边界是 D256 dropout backward。上游该路径至少需要 144 KiB
opt-in shared memory；sm86/sm89 的低共享内存分支只提供 no-dropout kernel。
MiniTrain 在 Python 支持检查和 C++ pybind 边界各检查一次，防止直接调用得到未
初始化梯度。Python 优先使用 PyTorch 提供的 opt-in shared-memory 字节数；当前
PyTorch 2.5.1 未暴露该字段时，回退到已审计的 sm80/sm86/sm87/sm89/sm90 表，
未知架构保守返回 unsupported。C++ 始终通过 CUDA runtime 查询真实属性。D256
no-dropout 在 sm86 仍然可用。

当前 source 可以为 sm80/sm86/sm89/sm90 编译，但仍属于 Ampere 风格 kernel。
sm90 cubin 不是 FlashAttention-3 的 Hopper WGMMA/TMA 专用实现。

## 6. 编译系统

三个 profile 对应不同机器：

| Profile | 内容 | 用途 |
| --- | --- | --- |
| `minimal` | fp16, D32 | 工具链 smoke build |
| `workstation` | fp16/bf16, D32/D64/D128 | 16 GB 本机日常开发 |
| `full` | 两种 dtype、六个 bucket | 大内存服务器 |

profile、架构、dtype、head bucket 和上游版本共同组成扩展 cache key。不同矩阵的
DLL 和 build 目录彼此隔离。最后一个目标架构同时保留 PTX，给兼容的新架构提供
forward JIT 余地。`MINITRAIN_CUDA_ARCHS` 是唯一架构入口，loader 会用它覆盖生成
`TORCH_CUDA_ARCH_LIST`，防止外部残留环境变量让 cache key 与实际 fatbin 不一致。

Windows sm86 full build 的实际瓶颈是宿主编译内存。两个 D256 backward nvcc
进程重叠时，16 GB 主机出现 cudafe out-of-memory；改为一个 worker 后可以复用
已有 object 继续编译。该错误与 GPU runtime register spill 无关。

安装交付采用 source-in-wheel + target JIT，而不是发布本机 `.pyd`。实际构建 wheel
后，将其中 `csrc` 文件集合与仓库逐路径比较：875/875 完全一致，包含 48 个 `.cu`、
19 个 FlashAttention header、CUTLASS/CUTE 和第三方 license，且没有 `.obj/.pyd`
或 build cache。隔离安装后，full profile 能从安装目录重新发现全部 48 个实例化
源文件。

安装态不能假定 `site-packages` 可写，因此 build root 按运行形态分流：源码
checkout 保留仓库内 cache；wheel 使用 `TORCH_EXTENSIONS_DIR` 或 PyTorch 用户
cache，并附加 Python/PyTorch/CUDA/platform ABI hash。隔离 wheel 实测确认 build
root 位于用户 cache，而不是安装包目录；`MINITRAIN_CUDA_BUILD_ROOT` 可显式覆盖。

完整命令见 [`cuda_ext_run_commands.md`](cuda_ext_run_commands.md)。

## 7. Correctness 与 benchmark

所有 CUDA 主验证都集中在 `tests/operator_bench.ipynb`。CUDA correctness 固定为
fp16，遍历：

```text
10 head dims x 2 causal modes x 2 dropout modes = 40 branches
```

10 个维度由 6 个 bucket 边界和 D40/D80/D160/D200 四个 masked-tail case 组成；
后者验证向上选择模板 bucket 后的 uneven-K load/store。

每个硬件支持的分支都与显式 fp32 PyTorch attention 公式比较 forward、dq、dk、
dv。dropout 参考使用 CUDA helper 提取的真实 keep mask，避免假设 PyTorch SDPA
与上游 kernel 以相同顺序消耗随机数。测试还比较调用前后的 CUDA RNG state：
no-dropout 必须保持完全不变，dropout 必须推进 generator。
每个分支还执行 `output.sum().backward()`，用 fp32 参考检查 expanded/stride-0
`dout` 经 C++ adapter 连续化后的 dq/dk/dv。

另有一组 D128 布局与 stream 测试：先创建物理布局 `(B,S,H,D)`，再通过
`transpose(1, 2)` 得到外层非连续但 `stride(-1)==1` 的 `(B,H,S,D)` 输入；随后把
forward、backward 和 dropout mask helper 全部放到显式非默认 CUDA stream 上。
该测试覆盖 causal/non-causal 与 dropout/no-dropout，并继续比较 forward、dq、dk、
dv，确认 adapter 传递的是实际 stride，launcher 使用的是 PyTorch 当前 stream。

capability 表只调用 Python predicate，不触发 JIT。它覆盖有效 D128 control、空
B/H/S、fp32、shape 不匹配、非法 dropout、非 8 倍数或超过 256 的 D、最后一维
stride=2、dropout 的 float32 上溢/下溢，以及硬件相关的 D200 dropout control。
13 个本机 case 全部符合预期；额外模拟 sm75 时，有效 tensor 也会在加载 extension
前被拒绝。C++ 边界保留同样的 sm80 与正维度检查，防止直接 pybind 调用绕过
Python。

直接 pybind 审计进一步证明 float32 语义一致：`dropout_p=1e-50` 下溢为 0，返回
空 RNG state，forward/backward 与显式 dropout=0 bitwise 相同且 CUDA RNG state
不变；`1-1e-12` 舍入为 float32 `1.0`，在 forward 参数构造前明确报错。

性能验证分为两层：原 D128 sequence sweep 观察长度扩展；完整矩阵固定 S=1024，
比较 Torch、Triton、CUDA 的 forward/backward p50/p95、峰值显存和 speedup。
所有显式标记为 CUDA 或 Triton 的 benchmark 都在计时前再次做对应 native
kernel 的支持判断。未编译或硬件不支持的组合显示 unsupported/unavailable，
不会把生产路径的下一级 fallback 时间误记为当前 provider 的 kernel 时间。

## 8. 当前证据与限制

本机已分别编译并运行过 fp16 D32/D64/D128 与 bf16 D32/D64/D128 shard，覆盖
causal/non-causal、forward/backward 和 dropout RNG replay。D128 fp16 的一个
`(B=1,H=4,S=1024,D=128)` 样例中，CUDA backward 比 PyTorch SDPA 更快且峰值
显存更低；该结果只代表本机单 shape，不是跨 GPU 性能结论。

随后使用 full cache 中已经完成的 24 个 fp16 object 与最新 C++ adapter 做了一次
只链接、不重编译 CUDA 的本机审计，并直接执行 notebook 中的 correctness
函数。sm86 可运行的 36 个分支全部通过 forward、dq/dk/dv、Philox RNG 行为和
expanded `dout` 检查；D200/D256 dropout 的 4 个分支按设计报告 unsupported。

同一批 fp16 对象还执行了 notebook 的布局与 stream 函数。4 个
`causal x dropout` 组合全部通过；kernel 输入 stride 保持
`(9472, 128, 256, 1)`，每次记录的 stream ID 都不是默认 stream 0，forward 与三组
梯度均通过 fp32 参考比较。

同一个临时链接还实际执行了 notebook 的性能函数。审计 shape 为
`(B=1,H=4,S=1024)`、causal，使用缩短的 5 ms warmup/20 ms forward repetition，
因此下表只证明计时流程与 provider 标记正确，不替代 notebook 正式 sweep：

| D | Dropout | Torch fwd/bwd | Triton fwd/bwd | CUDA fwd/bwd |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0 | 0.108 / 0.477 ms | 0.038 / 0.435 ms | 0.050 / 0.255 ms |
| 32 | 0.25 | 0.137 / 0.540 ms | 0.151 / 0.567 ms | 0.058 / 0.267 ms |
| 128 | 0 | 0.179 / 1.300 ms | 0.143 / 0.667 ms | 0.124 / 0.455 ms |
| 128 | 0.25 | 0.222 / 1.330 ms | 0.228 / 0.933 ms | 0.173 / 0.417 ms |
| 192 | 0 | 0.390 / 3.641 ms | unsupported | 0.215 / 0.720 ms |
| 192 | 0.25 | 0.382 / 3.694 ms | unsupported | 0.220 / 0.741 ms |
| 256 | 0 | 0.406 / 4.608 ms | unsupported | 0.231 / 0.748 ms |
| 256 | 0.25 | 0.412 / 4.659 ms | unsupported | unsupported |

Triton 当前只支持 head dim 不超过 128；sm86 CUDA 不支持 D256 dropout backward。
Notebook 在计时前检查这两种 native 能力，所以上述 unsupported 行没有执行下一级
fallback。

full sm86 日志已有 46/48 个 CUDA object，最新 C++ adapter 也已成功编译；本机
没有继续等待最后两个重型 D256 bf16 backward object，也没有把未链接的 full
mixed-dtype profile 声称为完整验证。服务器仍需完成 bf16 D256 object、链接 full
extension，并执行服务器目标架构上的 notebook。

ptxas 与 Nsight 证明部分生产 kernel 有真实 local-memory spill。项目接受上游
tile/spill 权衡，不添加 `--maxrregcount`、短序列阈值或性能 fallback。详细数据见
[`cuda_flash_attention_sm86_spill_analysis.md`](cuda_flash_attention_sm86_spill_analysis.md)。

逐文件阅读顺序见
[`cuda_flash_attention_code_reading_guide.md`](cuda_flash_attention_code_reading_guide.md)。
