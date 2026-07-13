# CUDA FlashAttention Build And Run Commands

本文档提供 MiniTrain CUDA FlashAttention 扩展在 Windows PowerShell 和 Linux Bash 下的完整运行示例。

所有命令都应在 `mini-train-sys` 项目根目录执行。首次构建建议先使用 `minimal` 配置验证编译工具链，再根据机器条件切换到 `workstation` 或 `full`。

## Windows PowerShell

### 1. 进入项目并激活 Python 环境

```powershell
cd "C:\Users\Zhai-Bin Cui\Desktop\GPTScratch\mini-train-sys"

# 根据实际虚拟环境路径调整。
.\.venv\Scripts\Activate.ps1
```

### 2. 检查 PyTorch、CUDA Toolkit 和编译工具

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
nvcc --version
where.exe cl
where.exe ninja
```

如果 `nvcc` 不在 `PATH` 中，根据实际安装版本设置 CUDA Toolkit：

```powershell
$env:CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$env:PATH="$env:CUDA_HOME\bin;$env:PATH"
```

### 3. 生成并检查 CUDA 模板实例化文件

```powershell
# 根据 dtype、head-dim、forward/backward 和 causal 配置生成 .cu 文件。
python minitrain/kernels/cuda_ext/generate_kernels.py

# 只检查仓库中的生成文件是否为最新版本，不修改文件。
python minitrain/kernels/cuda_ext/generate_kernels.py --check
```

### 4. 使用 minimal 配置验证编译工具链

以下配置面向当前 SM86 Windows 工作站，只编译 fp16、head-dim 32 的四个前向/反向实例化文件。

```powershell
$env:MINITRAIN_CUDA_BUILD_PROFILE="minimal"
$env:MINITRAIN_CUDA_ARCHS="86"
$env:MINITRAIN_CUDA_MAX_JOBS="1"
$env:MINITRAIN_CUDA_VERBOSE="1"

python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; ext = load_cuda_extension(); print('Loaded:', ext)"
```

Windows 构建会自动使用以下兼容参数，无需在命令行重复设置：

```text
--ptxas-options=-v
-allow-unsupported-compiler
-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
```

### 5. 编译工作站常用配置

`workstation` 包含 fp16/bf16、head-dim 32/64/128、forward/backward 和 causal/non-causal。

```powershell
$env:MINITRAIN_CUDA_BUILD_PROFILE="workstation"
$env:MINITRAIN_CUDA_ARCHS="86"
$env:MINITRAIN_CUDA_MAX_JOBS="1"
$env:MINITRAIN_CUDA_VERBOSE="1"

python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; ext = load_cuda_extension(); print('Loaded:', ext)"
```

### 6. 编译自定义矩阵

下面的例子只编译 fp16 的 head-dim 64 和 128：

```powershell
$env:MINITRAIN_CUDA_BUILD_PROFILE="minimal"
$env:MINITRAIN_CUDA_HEAD_DIMS="64;128"
$env:MINITRAIN_CUDA_DTYPES="fp16"
$env:MINITRAIN_CUDA_ARCHS="86"
$env:MINITRAIN_CUDA_MAX_JOBS="1"
$env:MINITRAIN_CUDA_VERBOSE="1"

python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

显式的 `MINITRAIN_CUDA_HEAD_DIMS` 和 `MINITRAIN_CUDA_DTYPES` 会覆盖 profile 中的默认矩阵。

保存编译日志
```powershell
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())" 2>&1 | Tee-Object -FilePath ".\minitrain\kernels\cuda_ext\build\cuda_build_sm86.log"
```

查看编译错误
```powershell
Select-String -Path ".\minitrain\kernels\cuda_ext\build\cuda_build_sm86.log" -Pattern "Used .* registers|stack frame|spill stores|spill loads"
```

### 7. 运行测试

```powershell
pytest tests/test_cuda_build_config.py -v
pytest tests/test_cuda_flash_attention.py -v
pytest tests/test_cuda_backend_fallback.py -v
```

### 8. 清理当前 PowerShell 会话中的配置

```powershell
Remove-Item Env:MINITRAIN_CUDA_BUILD_PROFILE -ErrorAction SilentlyContinue
Remove-Item Env:MINITRAIN_CUDA_HEAD_DIMS -ErrorAction SilentlyContinue
Remove-Item Env:MINITRAIN_CUDA_DTYPES -ErrorAction SilentlyContinue
Remove-Item Env:MINITRAIN_CUDA_ARCHS -ErrorAction SilentlyContinue
Remove-Item Env:MINITRAIN_CUDA_MAX_JOBS -ErrorAction SilentlyContinue
Remove-Item Env:MINITRAIN_CUDA_VERBOSE -ErrorAction SilentlyContinue
```

## Linux Bash

### 1. 进入项目并激活 Python 环境

```bash
cd ~/GPTScratch/mini-train-sys

# 根据实际虚拟环境路径调整。
source .venv/bin/activate
```

### 2. 检查 PyTorch、CUDA Toolkit 和编译工具

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
nvcc --version
which c++
which ninja
```

如果 CUDA Toolkit 不在默认搜索路径中：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

### 3. 生成并检查 CUDA 模板实例化文件

```bash
python minitrain/kernels/cuda_ext/generate_kernels.py
python minitrain/kernels/cuda_ext/generate_kernels.py --check
```

### 4. 使用 minimal 配置验证 A100/SM80 工具链

```bash
export MINITRAIN_CUDA_BUILD_PROFILE=minimal
export MINITRAIN_CUDA_ARCHS=80
export MINITRAIN_CUDA_MAX_JOBS=2
export MINITRAIN_CUDA_VERBOSE=1

python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; ext = load_cuda_extension(); print('Loaded:', ext)"
```

如果 Linux 机器使用 SM86、SM89 或 SM90 GPU，将 `MINITRAIN_CUDA_ARCHS` 改为对应值即可。

### 5. 编译服务器完整配置

下面的配置会编译全部 dtype 和 head-dim，并生成 SM80、SM86、SM89 和 SM90 目标代码：

```bash
export MINITRAIN_CUDA_BUILD_PROFILE=full
export MINITRAIN_CUDA_ARCHS="80;86;89;90"
export MINITRAIN_CUDA_MAX_JOBS=8
export MINITRAIN_CUDA_VERBOSE=1

python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; ext = load_cuda_extension(); print('Loaded:', ext)"
```

完整配置的模板编译会占用较多内存。出现内存压力时降低并行任务数：

```bash
export MINITRAIN_CUDA_MAX_JOBS=2
```

如果扩展只部署在一台 H100/SM90 服务器上，可以只生成 SM90 目标代码：

```bash
export MINITRAIN_CUDA_BUILD_PROFILE=full
export MINITRAIN_CUDA_ARCHS=90
export MINITRAIN_CUDA_MAX_JOBS=8
export MINITRAIN_CUDA_VERBOSE=1

python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

当前移植仍使用 SM80 风格的 FlashAttention kernel。将它编译为 SM90 机器码并不等于使用了 Hopper 专用的 WGMMA/TMA kernel。

### 6. 编译自定义矩阵

下面的例子只编译 bf16 的 head-dim 64 和 128：

```bash
export MINITRAIN_CUDA_BUILD_PROFILE=minimal
export MINITRAIN_CUDA_HEAD_DIMS="64;128"
export MINITRAIN_CUDA_DTYPES=bf16
export MINITRAIN_CUDA_ARCHS=80
export MINITRAIN_CUDA_MAX_JOBS=4
export MINITRAIN_CUDA_VERBOSE=1

python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

### 7. 运行测试

```bash
pytest tests/test_cuda_build_config.py -v
pytest tests/test_cuda_flash_attention.py -v
pytest tests/test_cuda_backend_fallback.py -v
```

### 8. 清理当前 Bash 会话中的配置

```bash
unset MINITRAIN_CUDA_BUILD_PROFILE
unset MINITRAIN_CUDA_HEAD_DIMS
unset MINITRAIN_CUDA_DTYPES
unset MINITRAIN_CUDA_ARCHS
unset MINITRAIN_CUDA_MAX_JOBS
unset MINITRAIN_CUDA_VERBOSE
```

## 推荐构建顺序

1. 用 `generate_kernels.py --check` 检查生成文件。
2. 用 `minimal` 验证 CUDA、宿主编译器、Ninja 和 PyTorch ABI。
3. 在本机使用 `workstation`，在内存充足的构建服务器上使用 `full`。
4. `MINITRAIN_CUDA_ARCHS` 只填写实际需要部署的 GPU 架构，避免无意义地增加编译时间和扩展体积。
5. 编译成功后运行 CUDA 正确性、反向传播和 fallback 测试。
