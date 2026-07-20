# CUDA FlashAttention 扩展

这里是 MiniTrain 的 CUDA FlashAttention 后端。设备端算法来自
FlashAttention 2.8.4；MiniTrain 只维护框架接入、张量适配、显式实例化矩阵和
JIT 编译配置。上游 kernel 代码保存在 `csrc/third_party`，迁移时没有改写其
tile、寄存器或共享内存策略。

## 支持范围

- CUDA sm80 及以上，本机已验证 sm86；
- Q/K/V 布局为 `(batch, heads, sequence, head_dim)`；
- fp16 和 bf16；
- head dim 不超过 256，并映射到 32、64、96、128、192、256 六个编译桶；
- causal / non-causal；
- dropout / no-dropout 的 forward 和 backward。

当前不支持 GQA/MQA、varlen、KV cache、local window、ALiBi 和 softcap。调用不在
CUDA 扩展支持范围内时，MiniTrain 按 `CUDA -> Triton -> PyTorch` 继续查找可用
实现。

sm86/sm89 上 `D > 192` 的 dropout backward 超过设备可用的 opt-in shared
memory。Python 支持检查会提前拒绝该组合，C++ 入口也会再次检查；D=256 的
no-dropout 路径不受影响。Python 优先读取 PyTorch 暴露的真实设备属性，旧版
PyTorch 则使用已审计架构表；未知架构保守地等待验证。这是上游实现的硬件边界，
不是性能 fallback。

## 目录职责

| 路径 | 职责 |
| --- | --- |
| `__init__.py` | 注册 `CudaOpsBackend`，继承 Triton/PyTorch fallback |
| `flash_attention.py` | 支持判断、autograd 封装、测试用 dropout mask helper |
| `build.py` | profile、架构、编译参数和 JIT cache key |
| `generate_kernels.py` | 生成 48 个薄实例化 `.cu` 文件 |
| `csrc/flash_api_upstream.cpp` | PyTorch tensor 到上游参数结构的 C++ 适配 |
| `csrc/instantiations` | dtype、head bucket、causal 的显式模板实例化 |
| `csrc/third_party` | FlashAttention 2.8.4 与 CUTLASS/CUTE 原始代码 |

第一次阅读请从
[`docs/kernels/cuda_flash_attention_code_reading_guide.md`](../../../docs/kernels/cuda_flash_attention_code_reading_guide.md)
开始。

## 编译配置

| Profile | dtype | head-dim bucket | `.cu` 数量 |
| --- | --- | --- | ---: |
| `minimal` | fp16 | 32 | 4 |
| `workstation`（默认） | fp16、bf16 | 64 | 8 |
| `full` | fp16、bf16 | 32、64、96、128、192、256 | 48 |

本机是 Windows sm86 机器，项目模型默认使用 D=64，因此 `workstation` 只构建
D=64 的 fp16/bf16 路径并使用一个 nvcc worker。原来的 32/64/128 矩阵保留在
`build.py` 注释中，服务器应显式选择 `full`，并根据单个 D=256 backward 编译
进程的峰值内存决定并行度。

本机先编译最小 profile：

```powershell
$env:MINITRAIN_CUDA_BUILD_PROFILE="minimal"
$env:MINITRAIN_CUDA_ARCHS="86"
$env:MINITRAIN_CUDA_MAX_JOBS="1"
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

再编译日常使用的 profile：

```powershell
$env:MINITRAIN_CUDA_BUILD_PROFILE="workstation"
$env:MINITRAIN_CUDA_ARCHS="86"
$env:MINITRAIN_CUDA_MAX_JOBS="1"
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

服务器完整构建示例：

```bash
export MINITRAIN_CUDA_BUILD_PROFILE=full
export MINITRAIN_CUDA_ARCHS="80;86;89;90"
export MINITRAIN_CUDA_MAX_JOBS=8
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

可用 `MINITRAIN_CUDA_HEAD_DIMS="64;128"` 和
`MINITRAIN_CUDA_DTYPES="fp16;bf16"` 覆盖 profile。架构、dtype 和 head-dim
矩阵都会进入扩展名称的 hash，不同配置不会误用同一个 `.pyd`。
`MINITRAIN_CUDA_ARCHS` 会在加载前统一转换为 `TORCH_CUDA_ARCH_LIST`；不要直接设置
后者，因为 MiniTrain 的架构列表必须同时决定 cache key 和实际 fatbin 目标。

Windows 编译默认包含以下兼容参数：

```text
--ptxas-options=-v
-allow-unsupported-compiler
-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
```

`MAX_JOBS` 只影响并行编译，不影响 cache key。编译中断后用相同矩阵重试，ninja
会复用已经生成的 object。

## 验证与运行

检查生成文件没有漂移：

```powershell
python minitrain/kernels/cuda_ext/generate_kernels.py --check
```

`tests/operator_bench.ipynb` 是 CUDA correctness 和性能的统一验证入口。attention
单元统一使用 fp16，包含：

- Torch、Triton、CUDA 的 forward/backward 延迟和显存 benchmark；
- 六个 head bucket 加四个 masked-tail head dim、两种 causal 模式和
  dropout/no-dropout 分支的 CUDA correctness 表；
- 从 CUDA 调试 helper 取回上游 Philox keep mask，再用显式 fp32 PyTorch 公式
  比较 forward、dq、dk、dv。
- 验证 no-dropout 不推进 CUDA RNG state，而 dropout 会推进 generator。
- 用 `output.sum().backward()` 覆盖 expanded/stride-0 `dout` 的 backward 适配。
- 用外层非连续、最后一维连续的输入，在非默认 CUDA stream 上覆盖 forward、
  backward 和 dropout mask helper。
- 用 capability 边界表验证空 B/H/S、错误 dtype/shape/dropout、非法 D 和最后一维
  非连续输入会在 JIT 加载前被拒绝，并覆盖 dropout 的 float32 上溢/下溢边界。
- 固定 S=1024 遍历同一分支矩阵的 Torch/Triton/CUDA 性能表。

Notebook 的 CUDA/Triton benchmark 会在计时前检查对应 native kernel 的支持
条件。训练仍可正常 fallback，但 benchmark 不会把 fallback 路径标成上一级
provider 的结果。执行 `Run All` 时，任一受支持分支的数值误差、布局/stream
回归或 capability 判断错误都会触发断言并停止，避免继续产出无效性能数据。

本机 sm86 已运行 notebook correctness 的全部 40 个 fp16 组合：36 个硬件支持
分支全部通过；D200/D256 dropout 的 4 个分支按上游限制显示 unsupported。
D40/D80/D160/D200 验证了向上 bucket 分派和 uneven-K masked load/store。完整
fp16/bf16 `full` 二进制仍应在大内存服务器上构建和验证。D128 的 4 个
布局/stream 组合也已实际运行通过，输入外层 stride 未被强制连续化，kernel 均
运行在显式非默认 stream。13 个 capability 边界 case 也全部符合预期；模拟 sm75
会在 Python predicate 中直接返回 unsupported。

Dropout 概率在 Python 和 C++ 边界统一规范化为 kernel 实际使用的 float32。极小
正数若下溢为 0，会严格进入 no-dropout 分支且不推进 RNG；小于 1 但舍入为
float32 `1.0` 的值会被拒绝，避免 forward/backward 选择不同模板。

D128 fp16 object 的成对 SASS 审计也确认：no-dropout forward/backward 中四个
Philox4x32 固定常量均为 0 次，dropout 对照组分别出现 54 次和 18 次。随机数路径
确实由 `Is_dropout=false` 编译消除；ptxas 总 `REG` 仍可能因 attention 主体达到 255。

训练配置使用：

```yaml
backend:
  ops: cuda
```

首次遇到受支持的 attention 输入时会加载对应 JIT 扩展。运行 notebook 前应先
检查它打印的 build config，避免把未编译 bucket 的 fallback 结果误当作 CUDA
kernel benchmark。

Wheel 和 editable install 都携带完整 JIT 源码，不携带本机 `.obj/.pyd` cache。
实际 wheel 已核对为 875 个 `csrc` 文件逐路径完整匹配源码目录，其中包括 48 个
实例化 `.cu`、adapter、上游 header、CUTLASS/CUTE 和两份第三方 license。安装到
新环境后首次调用仍需要对应的 CUDA Toolkit、host compiler 和 Ninja。

源码 checkout 默认把 Ninja cache 放在本目录的 `build/`，便于查看日志和断点续编。
Wheel 安装不会写可能只读的 `site-packages`，而是使用 `TORCH_EXTENSIONS_DIR` 或
PyTorch 用户 cache，并按 Python/PyTorch/CUDA/platform ABI 隔离。可用
`MINITRAIN_CUDA_BUILD_ROOT` 指定其他可写根目录。

sm86 的 spill 结论和测量数据见
[`docs/kernels/cuda_flash_attention_sm86_spill_analysis.md`](../../../docs/kernels/cuda_flash_attention_sm86_spill_analysis.md)。
