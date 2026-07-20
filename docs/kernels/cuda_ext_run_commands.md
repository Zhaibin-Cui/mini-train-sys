# CUDA FlashAttention 编译与运行手册

所有命令都从 `mini-train-sys` 仓库根目录执行。代码结构与设计说明见
[`cuda_flash_attention_code_reading_guide.md`](cuda_flash_attention_code_reading_guide.md)。

安装包携带 CUDA JIT 源码而不是预编译 `.pyd`。可使用：

```bash
pip install ".[cuda]"
```

构建出的 wheel 标记为 `py3-none-any`，这里只表示 Python 包本身没有绑定某个预编译
平台二进制；首次调用 CUDA attention 时仍会在目标机器上用本节工具链编译对应
profile。已验证 wheel 中 875 个 `csrc` 文件与源码目录逐路径完全一致，并且没有
打包本机 build cache。

源码 checkout 默认使用 `minitrain/kernels/cuda_ext/build`。Wheel 安装默认使用
`TORCH_EXTENSIONS_DIR` 或 PyTorch 用户 cache，避免向可能只读的 `site-packages`
写入，并附加 Python/PyTorch/CUDA/platform ABI tag。需要把编译放到高速本地盘时：

```bash
export MINITRAIN_CUDA_BUILD_ROOT=/local_nvme/minitrain_cuda_build
```

Windows 使用 `$env:MINITRAIN_CUDA_BUILD_ROOT="D:\minitrain_cuda_build"`。

## 1. 检查工具链

Windows PowerShell：

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))"
nvcc --version
where.exe cl
where.exe ninja
```

Linux：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))"
nvcc --version
which c++
which ninja
```

PyTorch 的 CUDA runtime、CUDA Toolkit 和 host compiler 必须彼此兼容。Windows
构建已在 `build.py` 中固定加入：

```text
--ptxas-options=-v
-allow-unsupported-compiler
-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
```

## 2. 检查生成矩阵

仓库提交了 48 个薄 `.cu` 实例化文件。日常只检查，不应每次重写：

```powershell
python minitrain/kernels/cuda_ext/generate_kernels.py --check
```

只有修改 dtype/head-dim/causal 实例化矩阵后才运行：

```powershell
python minitrain/kernels/cuda_ext/generate_kernels.py
```

## 3. 本机 sm86 构建

先用 fp16/D32 验证工具链：

```powershell
$env:MINITRAIN_CUDA_BUILD_PROFILE="minimal"
$env:MINITRAIN_CUDA_ARCHS="86"
$env:MINITRAIN_CUDA_MAX_JOBS="1"
$env:MINITRAIN_CUDA_VERBOSE="1"
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

再编译本机常用矩阵（fp16/bf16，D32/D64/D128）：

```powershell
$env:MINITRAIN_CUDA_BUILD_PROFILE="workstation"
$env:MINITRAIN_CUDA_ARCHS="86"
$env:MINITRAIN_CUDA_MAX_JOBS="1"
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

16 GB 内存机器不要并行编译多个 D256 backward translation unit。编译被中断后，
保持完全相同的 profile、arch、dtype 和 head dims 再执行一次，ninja 会继续使用
已完成的 object。

需要保存 ptxas 日志时：

```powershell
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())" 2>&1 | Tee-Object -FilePath ".\minitrain\kernels\cuda_ext\build\cuda_build_sm86.log"
```

## 4. 服务器完整构建

以下示例生成 sm80/sm86/sm89/sm90 cubin，并为最后一个架构保留 PTX：

```bash
export MINITRAIN_CUDA_BUILD_PROFILE=full
export MINITRAIN_CUDA_ARCHS="80;86;89;90"
export MINITRAIN_CUDA_MAX_JOBS=8
export MINITRAIN_CUDA_VERBOSE=1
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

`MINITRAIN_CUDA_ARCHS` 是架构的唯一配置入口。构建器会在加载前据此覆盖并生成
`TORCH_CUDA_ARCH_LIST`，从而保证扩展 cache key 与实际 cubin/PTX 目标一致；不要再
手工设置后者。

`MAX_JOBS=8` 只是大内存服务器示例。先观察一个 D256 backward nvcc 进程的峰值
宿主内存，再决定并行度。当前移植的是 FlashAttention 2.8.4 的 Ampere 风格
kernel；编译为 sm90 并不等于使用 Hopper 专用 WGMMA/TMA kernel。

只部署单一架构时应缩小 arch 列表，例如：

```bash
export MINITRAIN_CUDA_BUILD_PROFILE=full
export MINITRAIN_CUDA_ARCHS=90
export MINITRAIN_CUDA_MAX_JOBS=8
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

## 5. 自定义构建矩阵

环境变量会覆盖 profile 的 dtype 和 head-dim 默认值：

```powershell
$env:MINITRAIN_CUDA_BUILD_PROFILE="minimal"
$env:MINITRAIN_CUDA_HEAD_DIMS="64;128"
$env:MINITRAIN_CUDA_DTYPES="fp16"
$env:MINITRAIN_CUDA_ARCHS="86"
$env:MINITRAIN_CUDA_MAX_JOBS="1"
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

允许的 bucket 是 `32;64;96;128;192;256`，dtype 是 `fp16;bf16`。配置矩阵进入
JIT 扩展 cache key，因此切换配置不会误加载旧 DLL。

## 6. Notebook 全量验证

先编译 notebook 需要的 profile，再启动：

```powershell
jupyter lab tests/operator_bench.ipynb
```

attention 区域完成四类 fp16 验证：

1. correctness 遍历六个 head bucket 与 D40/D80/D160/D200 masked-tail 维度，再
   组合 causal/non-causal、dropout/no-dropout，使用 CUDA helper 导出的真实
   Philox keep mask 构造 fp32 PyTorch 参考；
2. layout/stream correctness 使用外层非连续、最后一维连续的输入，在显式非默认
   CUDA stream 上检查 forward、backward、dropout mask 与 stride 透传；
3. capability matrix 验证无效 shape、dtype、dropout、head dim 和 stride 会在
   extension JIT 加载前返回 unsupported，并检查 dropout 转换为 float32 后的
   上溢/下溢边界；
4. performance 使用统一 `benchmark_step` 封装比较 Torch、Triton、CUDA 的
   forward/backward 延迟、峰值显存和 speedup。

前三组 correctness 检查带有硬断言。执行 `Run All` 时，只要受支持分支出现数值
错误、布局/stream 回归或 capability 判断不符，notebook 就会在性能测试前停止。
标记为 `unsupported` 的已知 build/hardware 边界不会被当作失败。

notebook 会打印当前 build config。未编译的 bucket 和硬件不支持的
sm86/sm89 `D > 192 + dropout` 显示为 `unsupported`，不会把 fallback 耗时当作 CUDA
kernel 结果。D128 sequence sweep 也执行相同检查；profile 缺少 D128 时 CUDA 行
显示 `unavailable`，而不是静默测量 Triton。Triton 行也执行自身支持检查，例如
D192/D256 会显示 `unsupported`，不会静默测量 PyTorch。

这个特殊 backward 分支按设备的 opt-in shared-memory 容量判断，而不是只看型号。
PyTorch 未暴露真实字节数时使用已审计架构表；未知架构在完成验证前显示
`unsupported`，直接 pybind 调用仍由 C++ 的 CUDA runtime 属性检查保护。

## 7. 训练运行

配置 MiniTrain：

```yaml
backend:
  ops: cuda
```

第一次遇到受支持的 attention 输入时加载 JIT extension。不支持的能力组合按
`CUDA -> Triton -> PyTorch` 处理；没有基于 sequence length 或实测速度的性能
fallback。
