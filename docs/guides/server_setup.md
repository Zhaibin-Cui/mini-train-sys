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

要求：`nvidia-smi` 显示全部计划使用的 GPU；驱动支持 PyTorch CUDA 12.4 wheel；当前
两条件正式实验建议使用 80 GB 系统盘和 100 GB 挂载盘；容器的 `/dev/shm` 建议至少
8 GiB。
不要在已有云镜像上盲目重装驱动，驱动不满足要求时应按云厂商或 NVIDIA 文档升级。

## 2. 系统盘与挂载盘分工

不要把训练产物长期写入系统盘。推荐布局：

| 位置 | 保存内容 | 原因 |
|---|---|---|
| 系统盘 | 操作系统、NVIDIA 驱动、Git 仓库、源码、配置、脚本、文档、`.venv` | 体积较小，代码升级和环境维护集中 |
| 挂载 SSD | 数据、token shards、Probe cache、checkpoint、TensorBoard/JSONL 日志、结果、benchmark、CUDA/Triton/Torch/Pip 缓存、临时文件 | 容量和写入量大，可独立扩容、快照和保留 |

当前 SynBioS 模型有 293,494,272 个参数。FP32 模型约 1.09 GiB；包含模型和两份 Adam
状态的 DCP checkpoint 约 3.28 GiB。默认每个 run 保留 2 个最新 checkpoint、1 个
safety checkpoint，并只给最新 checkpoint 保留一份约 1.09 GiB 的 `model.pt`，因此：

```text
单个 run 稳态 checkpoint       约 10.9 GiB
两个语料条件稳态 checkpoint    约 21.8 GiB
原子保存和清理前的瞬时峰值      约 26.2 GiB
正式数据、token/probe cache     小于 2 GiB
CUDA/Triton/Pip cache 与结果    约 5～15 GiB
```

所以一种训练策略下的两个正式条件通常占 35～45 GiB，保存新 checkpoint 时按 55 GiB
峰值准备即可。100 GB 挂载盘仍有约 45 GB 安全余量。只有同时长期保留 single、DDP、
FSDP 三套正式训练时，才建议把挂载盘增加到 150～200 GB。

代码仓库建议放在 `$HOME/src/mini-train-sys`，挂载盘示例为 `/data`。先确认 `/data`
确实是独立磁盘，而不是系统根分区中的普通目录：

```bash
lsblk -f
findmnt -T /data
df -h / /data
```

进入仓库后初始化存储布局：

```bash
cd "$HOME/src/mini-train-sys"
bash scripts/bash/setup_storage.sh /data
source .minitrain-storage.env
readlink -f artifacts
df -h . artifacts
```

脚本会创建以下结构，并让仓库中的 `artifacts` 成为指向挂载盘的符号链接：

```text
系统盘/$HOME/src/mini-train-sys/
├── .venv/
├── configs/ minitrain/ experiments/ scripts/ tests/ docs/
├── .minitrain-storage.env
└── artifacts -> /data/mini-train-sys/artifacts

/data/mini-train-sys/
├── artifacts/                 数据、checkpoint、日志、结果和 benchmark
├── cache/
│   ├── cuda_ext/              CUDA C++ 扩展构建
│   ├── triton/                Triton JIT cache
│   ├── torch_extensions/
│   ├── torch/ pip/ xdg/
│   └── huggingface/
└── tmp/                       大型临时编译文件
```

`.minitrain-storage.env` 是机器本地配置，已被 Git 忽略。所有 `scripts/bash/*.sh` 启动
入口会自动加载它；直接运行 Python、Jupyter 或手工编译 CUDA 前应先执行：

```bash
source .minitrain-storage.env
```

如果仓库已经有真实的 `artifacts/` 目录，初始化脚本会拒绝覆盖。先安全复制并保留备份：

```bash
mkdir -p /data/mini-train-sys/artifacts
rsync -aH --info=progress2 artifacts/ /data/mini-train-sys/artifacts/
mv artifacts artifacts.system-disk-backup
bash scripts/bash/setup_storage.sh /data
```

确认新位置的数据、checkpoint 和恢复流程正常后，再人工处理
`artifacts.system-disk-backup`。初始化脚本不会替你删除已有实验数据。

## 3. 系统工具与代码

Ubuntu/Debian：

```bash
sudo apt-get update
sudo apt-get install -y git rsync util-linux build-essential python3 python3-venv python3-dev
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

## 4. 一键安装

完成第 2 节的挂载盘初始化后，下面一条命令会在系统盘源码目录创建 `.venv`、安装
官方 PyTorch 2.5.1 CUDA 12.4 wheel、安装
`.[server]`、运行 `pip check`，最后自动执行服务器预检：

```bash
bash scripts/bash/setup_server.sh
source .venv/bin/activate
```

常用覆盖：

```bash
PYTHON_BIN=python3.11 bash scripts/bash/setup_server.sh
EXPECTED_GPUS=4 bash scripts/bash/setup_server.sh
PYTORCH_CUDA_INDEX=cu121 bash scripts/bash/setup_server.sh
REQUIRE_NVCC=1 bash scripts/bash/setup_server.sh
```

安装脚本可以安全重跑：已有 virtualenv 会被复用，pip 只更新不满足约束的包。
服务器安装入口默认要求 `.minitrain-storage.env` 已存在，避免忘记挂载数据盘。只有不保留
产物的临时 smoke 机器才使用：

```bash
ALLOW_SYSTEM_DISK_STORAGE=1 bash scripts/bash/setup_server.sh
```

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

## 5. `server` extra 内容

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

## 6. 环境预检

修改驱动、`CUDA_VISIBLE_DEVICES`、Python 环境或节点后重新运行：

```bash
source .venv/bin/activate
minitrain-check-server --expected-gpus 8 \
  --min-free-disk-gb 40 \
  --output artifacts/server_environment.json
```

它检查 Linux/Python、server 模块、PyTorch CUDA、NCCL、BF16、GPU 数和型号、拓扑、
checkout 完整性、挂载盘剩余空间、`/dev/shm` 和可选 `nvcc`。设置了
`MINITRAIN_STORAGE_ROOT` 或存在 `artifacts` 链接时，磁盘阈值针对实验挂载盘而不是
源码所在系统盘。退出码为 0 才表示硬性条件通过。

单独核对版本：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
python -c "import torch.distributed as dist; print(dist.is_nccl_available())"
python -c "import triton; print(triton.__version__)"
```

## 7. 可选 CUDA C++ 扩展

默认服务器配置使用 Triton，不需要 `nvcc`。只有运行 `minitrain.kernels.cuda_ext` 时才
安装与 PyTorch wheel ABI 兼容的 CUDA Toolkit，并执行：

```bash
nvcc --version
source .minitrain-storage.env
export CUDA_HOME=/usr/local/cuda-12.4
export MINITRAIN_CUDA_ARCHS=89
export MINITRAIN_CUDA_BUILD_PROFILE=workstation
export MINITRAIN_CUDA_MAX_JOBS=2
minitrain-check-server --expected-gpus 8 --require-nvcc
```

RTX 4090 是 SM89。`workstation` 当前编译模型需要的 head-dim 64、FP16/BF16 路径。
只有确实需要全部 bucket 时才使用 `MINITRAIN_CUDA_BUILD_PROFILE=full`；完整构建会明显
增加编译时间、主机内存和磁盘占用。

## 8. 代码、CPU 与 CUDA smoke

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

## 9. 分布式 benchmark 验收

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

## 10. SynBioS 预训练

长实验建议在 `tmux` 中启动：

```bash
tmux new -s minitrain
source .venv/bin/activate
source .minitrain-storage.env
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

## 11. Probe 与 Router

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

## 12. 一键正式实验

确认配置、磁盘和恢复策略后：

```bash
CONFIRM_FULL_EXPERIMENT=1 NPROC=8 \
  bash scripts/bash/synbios_full_experiment.sh ddp
```

它串行执行两个语料条件的预训练、smoke/pilot/formal Probe、Router analysis 和结果比较。
中断后重跑会恢复 committed 主训练 checkpoint，并复用 identity 一致的 Probe 产物。

## 13. 实验前最终清单

```text
[ ] nvidia-smi 显示全部 GPU，拓扑已保存
[ ] Git commit/dirty 状态已记录
[ ] source .venv/bin/activate
[ ] artifacts 指向挂载盘，.minitrain-storage.env 已加载
[ ] 挂载盘剩余空间满足正式实验和 checkpoint 保留策略
[ ] pip check 通过
[ ] minitrain-check-server 返回 ok=true
[ ] ruff 与 pytest 通过
[ ] 单 GPU smoke 通过
[ ] 目标卡数 DDP/FSDP smoke 通过
[ ] distributed benchmark weak/capacity 已验收
[ ] tmux/调度日志与恢复命令已确认
[ ] 正式 run YAML 和 model YAML 已归档
```

## 参考

- [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)
- [PyTorch 历史版本 CUDA wheel](https://pytorch.org/get-started/previous-versions/)
- [Triton 官方安装](https://triton-lang.org/main/getting-started/installation.html)
- [PyTorch C++/CUDA extension](https://docs.pytorch.org/docs/main/cpp_extension.html)
