# Linux/CUDA 服务器环境准备与实验验收

本文给出从一台已安装 NVIDIA 驱动的 Linux 服务器，到 MiniTrainSys 预训练、Probe 和
分布式 benchmark 可运行状态的完整流程。目标环境是单机 4/8×RTX 4090；其他 NVIDIA
GPU 可以使用同一流程，但需要重新确认显存、BF16 和算子支持范围。

当前锁定并验证的 Python 栈：

```text
Python       3.10 / 3.11 / 3.12（推荐 3.11）
PyTorch      2.5.1
CUDA wheel   cu124（默认服务器安装脚本）
Triton       3.1.0
```

PyTorch wheel 自带运行时所需的 CUDA 用户态库，但不包含编译 CUDA C++ 扩展所需的
`nvcc`。默认 Triton/DDP/FSDP 实验不要求系统 CUDA Toolkit；只有使用 `cuda_ext`
FlashAttention 时才需要额外安装 Toolkit。

## 1. 主机前置检查

```bash
uname -a
cat /etc/os-release
nvidia-smi
nvidia-smi -L
nvidia-smi topo -m
lscpu
free -h
df -h
```

要求：`nvidia-smi` 显示全部计划使用的 GPU；驱动支持 PyTorch CUDA 12.4 wheel；项目、
数据和 checkpoint 磁盘建议至少预留 100 GiB；容器的 `/dev/shm` 建议至少 8 GiB。
不要在已有云镜像上盲目重装驱动，驱动不满足要求时应按云厂商或 NVIDIA 文档升级。

## 2. 系统工具与代码

Ubuntu/Debian：

```bash
sudo apt-get update
sudo apt-get install -y git build-essential python3 python3-venv python3-dev
python3 --version
```

Python 必须是 3.10、3.11 或 3.12。若系统 Python 不在该范围，使用服务器已有的 Conda、
模块系统或 pyenv 提供 Python 3.11。

```bash
git clone <mini-train-sys-repository-url> mini-train-sys
cd mini-train-sys
git status --short
git rev-parse HEAD
```

如果代码由 `rsync`、共享盘或调度系统同步，只需进入仓库根目录，并记录 commit 和未提交
改动。正式对照实验必须保存这两项 provenance。

## 3. 一键安装

下面一条命令会创建 `.venv`、安装官方 PyTorch 2.5.1 CUDA 12.4 wheel、安装
`.[server]`、运行 `pip check`，最后自动执行服务器预检：

```bash
bash scripts/bash/setup_server.sh
source .venv/bin/activate
```

常用覆盖：

```bash
PYTHON_BIN=python3.11 bash scripts/bash/setup_server.sh
VENV_DIR=/data/venvs/mini-train-sys bash scripts/bash/setup_server.sh
EXPECTED_GPUS=4 bash scripts/bash/setup_server.sh
PYTORCH_CUDA_INDEX=cu121 bash scripts/bash/setup_server.sh
REQUIRE_NVCC=1 bash scripts/bash/setup_server.sh
```

安装脚本可以安全重跑：已有 virtualenv 会被复用，pip 只更新不满足约束的包。

### 手工等价命令

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[server]"
python -m pip check
minitrain-check-server --expected-gpus 8 \
  --output artifacts/server_environment.json
```

标准 `pyproject.toml` 可以声明版本和 extras，但不能声明某个依赖必须来自特定包索引。
因此安装脚本先从 PyTorch 官方索引安装明确的 CUDA wheel，再安装项目 server extra。

## 4. `server` extra 内容

```bash
python -m pip install -e ".[server]"
```

包含：

- 核心训练：NumPy、PyYAML、TensorBoard、PyTorch；
- 数据：PyArrow、tiktoken、Hugging Face tokenizers；
- GPU kernel：Triton 3.1；
- Notebook/图表：JupyterLab、IPykernel、pandas、matplotlib、nbconvert；
- 测试：pytest、Ruff、cloudpickle；
- CUDA 扩展构建辅助：Ninja、packaging。

`experiments.*` 已加入 setuptools 包发现范围，因此 editable install 后 SynBioS 脚本可以
稳定导入实验模块。

## 5. 环境预检

修改驱动、`CUDA_VISIBLE_DEVICES`、Python 环境或节点后重新运行：

```bash
source .venv/bin/activate
minitrain-check-server --expected-gpus 8 \
  --min-free-disk-gb 100 \
  --output artifacts/server_environment.json
```

它检查 Linux/Python、server 模块、PyTorch CUDA、NCCL、BF16、GPU 数和型号、拓扑、
checkout 完整性、磁盘、`/dev/shm` 和可选 `nvcc`。退出码为 0 才表示硬性条件通过。

单独核对版本：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
python -c "import torch.distributed as dist; print(dist.is_nccl_available())"
python -c "import triton; print(triton.__version__)"
```

## 6. 可选 CUDA C++ 扩展

默认服务器配置使用 Triton，不需要 `nvcc`。只有运行 `minitrain.kernels.cuda_ext` 时才
安装与 PyTorch wheel ABI 兼容的 CUDA Toolkit，并执行：

```bash
nvcc --version
export CUDA_HOME=/usr/local/cuda-12.4
export MINITRAIN_CUDA_ARCHS=89
export MINITRAIN_CUDA_BUILD_PROFILE=workstation
export MINITRAIN_CUDA_MAX_JOBS=2
minitrain-check-server --expected-gpus 8 --require-nvcc
```

RTX 4090 是 SM89。`workstation` 当前编译模型需要的 head-dim 64、FP16/BF16 路径。
只有确实需要全部 bucket 时才使用 `MINITRAIN_CUDA_BUILD_PROFILE=full`；完整构建会明显
增加编译时间、主机内存和磁盘占用。

## 7. 代码、CPU 与 CUDA smoke

```bash
source .venv/bin/activate
ruff check minitrain experiments scripts tests
python -m pytest -q
```

最小 CPU 训练：

```bash
python scripts/train.py \
  --config configs/train_debug.yaml \
  --model-config configs/model_debug_dense.yaml \
  --device cpu
```

单 GPU CUDA：

```bash
python scripts/train.py \
  --device cuda --smoke-steps 2 \
  --config configs/server/rtx4090_24gb/runs/single_1gpu.yaml \
  --model-config configs/model_debug_dense.yaml
```

4 卡 DDP/FSDP：

```bash
torchrun --standalone --nproc_per_node 4 scripts/train.py \
  --device cuda --smoke-steps 2 \
  --config configs/server/rtx4090_24gb/runs/ddp_4gpu.yaml \
  --model-config configs/model_debug_dense.yaml

torchrun --standalone --nproc_per_node 4 scripts/train.py \
  --device cuda --smoke-steps 2 \
  --config configs/server/rtx4090_24gb/runs/fsdp_4gpu.yaml \
  --model-config configs/model_debug_dense.yaml
```

8 卡服务器把进程数和配置名中的 `4gpu` 同时改为 `8gpu`。配置会再次校验
`WORLD_SIZE`，不一致时主动失败。

## 8. 分布式 benchmark 验收

```bash
python -m ipykernel install --user \
  --name mini-train-sys \
  --display-name "MiniTrainSys (.venv)"
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

本地建立隧道：

```bash
ssh -L 8888:127.0.0.1:8888 <user>@<server>
```

打开 `tests/distributed_server_benchmark.ipynb`，选择 `MiniTrainSys (.venv)`，依次运行
预检、weak scaling 和 capacity。结果、CSV、失败明细和 PNG 写入
`artifacts/distributed_benchmark/`。原理见
[`distributed_benchmark.md`](../benchmarks/distributed_benchmark.md)。

## 9. SynBioS 预训练

长实验建议在 `tmux` 中启动：

```bash
tmux new -s minitrain
source .venv/bin/activate
```

脚本会在数据缺失时自动生成 100,000 人的正式语料和 token shards：

```bash
# 单GPU
bash scripts/bash/synbios_moe.sh single single
bash scripts/bash/synbios_moe.sh multi5_permute single

# 4卡DDP
NPROC=4 bash scripts/bash/synbios_moe.sh single ddp
NPROC=4 bash scripts/bash/synbios_moe.sh multi5_permute ddp

# 8卡FSDP
NPROC=8 bash scripts/bash/synbios_moe.sh single fsdp
NPROC=8 bash scripts/bash/synbios_moe.sh multi5_permute fsdp
```

发现 committed checkpoint 时默认恢复 latest。安全恢复：

```bash
RESUME=safety NPROC=8 bash scripts/bash/synbios_moe.sh single ddp
```

## 10. Probe 与 Router

必须按 smoke → pilot → formal：

```bash
STAGE=smoke NPROC=8 \
  bash scripts/bash/synbios_probes.sh single ddp latest
STAGE=pilot NPROC=8 \
  bash scripts/bash/synbios_probes.sh single ddp latest
STAGE=formal NPROC=8 \
  bash scripts/bash/synbios_probes.sh single ddp latest
```

主训练使用 8 卡、Probe 只用指定 GPU：

```bash
STAGE=pilot NPROC=8 PROBE_DEVICES=cuda:1,cuda:3 \
  bash scripts/bash/synbios_probes.sh single ddp latest
```

完整矩阵见 [`scripts/synbios_moe_runbook.md`](../../scripts/synbios_moe_runbook.md)。

## 11. 一键正式实验

确认配置、磁盘和恢复策略后：

```bash
CONFIRM_FULL_EXPERIMENT=1 NPROC=8 \
  bash scripts/bash/synbios_full_experiment.sh ddp
```

它串行执行两个语料条件的预训练、smoke/pilot/formal Probe、Router analysis 和结果比较。
中断后重跑会恢复 committed 主训练 checkpoint，并复用 identity 一致的 Probe 产物。

## 12. 实验前最终清单

```text
[ ] nvidia-smi 显示全部 GPU，拓扑已保存
[ ] Git commit/dirty 状态已记录
[ ] source .venv/bin/activate
[ ] pip check 通过
[ ] minitrain-check-server 返回 ok=true
[ ] ruff 与 pytest 通过
[ ] 单 GPU smoke 通过
[ ] 目标卡数 DDP/FSDP smoke 通过
[ ] distributed benchmark weak/capacity 已验收
[ ] 数据、checkpoint、runs 所在磁盘充足
[ ] tmux/调度日志与恢复命令已确认
[ ] 正式 run YAML 和 model YAML 已归档
```

## 参考

- [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)
- [PyTorch 历史版本 CUDA wheel](https://pytorch.org/get-started/previous-versions/)
- [Triton 官方安装](https://triton-lang.org/main/getting-started/installation.html)
- [PyTorch C++/CUDA extension](https://docs.pytorch.org/docs/main/cpp_extension.html)
