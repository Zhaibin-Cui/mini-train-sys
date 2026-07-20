# SynBioS MoE 预训练与 Probe 启动手册

本文汇总 SynBioS MoE 的常用启动命令。Bash 脚本只负责路径、环境变量和任务编排；
数据生成、预训练、Probe、验证和 Router 分析的实际逻辑位于 Python 入口中。

## 1. 运行环境

正式实验面向 Linux/CUDA 服务器。先进入项目根目录并激活安装了项目依赖的 Python 环境：

```bash
cd /path/to/mini-train-sys
source .venv/bin/activate
```

服务器应把源码和 `.venv` 留在系统盘，把数据、checkpoint、Probe 结果及编译缓存放在
挂载 SSD。首次运行先初始化并检查链接：

```bash
bash scripts/bash/setup_storage.sh /data
source .minitrain-storage.env
readlink -f artifacts
df -h . artifacts
```

之后本手册中的 Bash 入口会自动读取 `.minitrain-storage.env`。完整分盘布局、已有
`artifacts` 迁移和磁盘验收见
[`docs/guides/server_setup.md`](../docs/guides/server_setup.md#2-系统盘与挂载盘分工)。

### 1.1 编译 CUDA 扩展（可选）

SynBioS 默认使用 Triton，不编译 CUDA 扩展也能运行。若要使用项目的 CUDA
FlashAttention，在 RTX 4090（SM89）服务器上执行：

```bash
cd /path/to/mini-train-sys
source .venv/bin/activate
source .minitrain-storage.env

# 检查 CUDA Toolkit 和编译工具。
nvcc --version
which c++
which ninja
export CUDA_HOME=/usr/local/cuda-12.4
EXPECTED_GPUS=$(nvidia-smi --list-gpus | wc -l)
minitrain-check-server --expected-gpus "$EXPECTED_GPUS" --require-nvcc

# 检查生成的 CUDA 实例文件是否与源码配置一致。
python minitrain/kernels/cuda_ext/generate_kernels.py --check

# 编译 SynBioS 当前 D64 模型需要的 FP16/BF16、SM89 扩展。
export MINITRAIN_CUDA_BUILD_PROFILE=workstation
export MINITRAIN_CUDA_ARCHS=89
export MINITRAIN_CUDA_MAX_JOBS=2
export MINITRAIN_CUDA_VERBOSE=1
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

如需编译所有 head-dim bucket，把 profile 改为 `full`：

```bash
export MINITRAIN_CUDA_BUILD_PROFILE=full
export MINITRAIN_CUDA_ARCHS=89
export MINITRAIN_CUDA_MAX_JOBS=8
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
```

`full` 会编译 48 个 CUDA 实例，耗时、宿主内存和磁盘占用都明显更高；应根据服务器
内存调低 `MINITRAIN_CUDA_MAX_JOBS`。编译中断后保持相同环境变量重试，Ninja 会复用
已完成的目标文件。不要手动设置 `TORCH_CUDA_ARCH_LIST`，架构统一由
`MINITRAIN_CUDA_ARCHS` 管理。更完整的编译和排错说明见
[`docs/kernels/cuda_ext_run_commands.md`](../docs/kernels/cuda_ext_run_commands.md)。

若要让 SynBioS 训练使用已编译扩展，将 `configs/synbios_moe/base.yaml` 中的
`backend.ops` 从 `triton` 改为 `cuda`；否则上述编译只会生成扩展，不会改变训练后端。

主要入口：

```text
scripts/bash/synbios_moe.sh       一个语料条件的预训练
scripts/bash/synbios_probes.sh    一个预训练 checkpoint 的 Probe 与 Router 分析
scripts/bash/synbios_full_experiment.sh
                                  两个语料条件的完整正式实验
```

语料条件：

```text
single             每人一篇 biography，固定属性顺序
multi5_permute     每人五篇 biography，每篇独立打乱属性句序
```

## 2. 预训练

### 2.1 单 GPU

```bash
bash scripts/bash/synbios_moe.sh single single
bash scripts/bash/synbios_moe.sh multi5_permute single
```

脚本首次运行时会自动生成缺失的 `profiles.jsonl`、数据 manifest 和 token shards。
两个条件使用相同的人物与事实表，脚本会在第二个条件生成后进行逐字校验。

### 2.2 四卡 DDP/FSDP

```bash
NPROC=4 bash scripts/bash/synbios_moe.sh single ddp
NPROC=4 bash scripts/bash/synbios_moe.sh multi5_permute ddp

NPROC=4 bash scripts/bash/synbios_moe.sh single fsdp
NPROC=4 bash scripts/bash/synbios_moe.sh multi5_permute fsdp
```

### 2.3 八卡 DDP/FSDP

```bash
NPROC=8 bash scripts/bash/synbios_moe.sh single ddp
NPROC=8 bash scripts/bash/synbios_moe.sh multi5_permute ddp

NPROC=8 bash scripts/bash/synbios_moe.sh single fsdp
NPROC=8 bash scripts/bash/synbios_moe.sh multi5_permute fsdp
```

SynBioS 的分布式启动预设只接受 4 或 8 个进程。`NPROC` 必须与所选 YAML 中的
`parallel.expected_world_size` 一致。

### 2.4 Checkpoint 恢复

发现同一 run 的 committed checkpoint 时，脚本默认追加 `--resume latest`：

```bash
NPROC=8 bash scripts/bash/synbios_moe.sh single ddp
```

从安全锚点恢复：

```bash
RESUME=safety NPROC=8 \
  bash scripts/bash/synbios_moe.sh single ddp
```

忽略现有 checkpoint，从头启动：

```bash
AUTO_RESUME=0 NPROC=8 \
  bash scripts/bash/synbios_moe.sh single ddp
```

checkpoint 目录格式：

```text
artifacts/synbios_moe/checkpoints/<run_name>/
  epoch_XXXXXX_step_XXXXXXXXX/
    distributed/
    runtime.pt
    rng_rank_*.pt
    model.pt
    COMMITTED
```

只有带 `COMMITTED` 的目录才会用于自动恢复或 Probe。

### 2.5 强制重建数据

```bash
FORCE_PREPARE=1 \
  bash scripts/bash/synbios_moe.sh single single
```

如果该语料条件已经存在 committed checkpoint，脚本会拒绝强制重建，以免把旧模型和
Adam 状态恢复到不同的数据上。需要重建时，应先人工归档对应 checkpoint。

## 3. Probe 整体顺序

Probe 按以下顺序运行：

```text
预训练 checkpoint
  → 一次性生成并验证 Probe mmap cache
  → 主模型 attribute-token accuracy gate
  → smoke：2 个任务，每个 500 step
  → pilot：全部 22 个任务，每个 3,000 step
  → formal：全部 22 个任务，每个 30,000 step
  → 独立 held-out validation
  → 六个属性的 Router analysis
```

默认 gate 在 10,000 篇 biography 上计算 teacher-forced attribute-token accuracy，要求
`micro_accuracy >= 0.90`。主模型尚未学会事实时，pipeline 会在训练 Probe 前停止。

每个 P/Q Probe 是独立的单 GPU 进程：主模型被冻结，只训练低秩输入扰动、归一化层和
分类器。多张 GPU 用于并行调度不同任务，不使用 DDP/FSDP，也不进行 Probe 间梯度同步。

## 4. 分阶段启动 Probe

命令格式：

```bash
STAGE=<smoke|pilot|formal> NPROC=<预训练卡数> \
  bash scripts/bash/synbios_probes.sh \
  <single|multi5_permute> <single|ddp|fsdp> <latest|checkpoint目录>
```

`NPROC` 用来定位预训练 run name。若主模型由 4 卡训练得到，必须设置 `NPROC=4`；若由
8 卡训练得到，必须设置 `NPROC=8`。

### 4.1 单卡预训练模型

```bash
STAGE=smoke \
  bash scripts/bash/synbios_probes.sh single single latest

STAGE=pilot \
  bash scripts/bash/synbios_probes.sh single single latest

STAGE=formal \
  bash scripts/bash/synbios_probes.sh single single latest
```

对 `multi5_permute` 条件运行：

```bash
STAGE=smoke \
  bash scripts/bash/synbios_probes.sh multi5_permute single latest

STAGE=pilot \
  bash scripts/bash/synbios_probes.sh multi5_permute single latest

STAGE=formal \
  bash scripts/bash/synbios_probes.sh multi5_permute single latest
```

### 4.2 四卡 DDP 预训练模型

```bash
STAGE=smoke NPROC=4 \
  bash scripts/bash/synbios_probes.sh single ddp latest

STAGE=pilot NPROC=4 \
  bash scripts/bash/synbios_probes.sh single ddp latest

STAGE=formal NPROC=4 \
  bash scripts/bash/synbios_probes.sh single ddp latest
```

### 4.3 八卡 DDP 预训练模型

```bash
STAGE=smoke NPROC=8 \
  bash scripts/bash/synbios_probes.sh single ddp latest

STAGE=pilot NPROC=8 \
  bash scripts/bash/synbios_probes.sh single ddp latest

STAGE=formal NPROC=8 \
  bash scripts/bash/synbios_probes.sh single ddp latest
```

FSDP 模型只需把第二个位置参数由 `ddp` 改为 `fsdp`：

```bash
STAGE=smoke NPROC=8 \
  bash scripts/bash/synbios_probes.sh single fsdp latest
```

## 5. Probe GPU 分配

Probe 默认自动使用 `NPROC` 张 GPU。预训练卡数和 Probe 卡数可以不同，例如主模型由
8 卡训练，但 Probe 只使用 3 张卡：

```bash
STAGE=pilot NPROC=8 PROBE_GPUS=3 \
  bash scripts/bash/synbios_probes.sh single ddp latest
```

显式选择 GPU：

```bash
STAGE=pilot NPROC=8 PROBE_DEVICES=cuda:1,cuda:3 \
  bash scripts/bash/synbios_probes.sh single ddp latest
```

也可以写成数字列表：

```bash
PROBE_DEVICES=1,3
```

每张 GPU 同时最多运行一个 Probe；完成当前任务后会从公共队列领取下一个任务。

## 6. 指定 checkpoint

默认的 `latest` 会选择对应 run 下目录名最大的 committed checkpoint：

```bash
STAGE=smoke NPROC=8 \
  bash scripts/bash/synbios_probes.sh single ddp latest
```

也可以传入具体目录：

```bash
STAGE=smoke NPROC=8 \
  bash scripts/bash/synbios_probes.sh single ddp \
  artifacts/synbios_moe/checkpoints/synbios_moe_single_ddp_8gpu/epoch_000540_step_001234567
```

指定目录必须包含 `COMMITTED`，且供 Probe 使用的 checkpoint 应包含 `model.pt`。

## 7. Probe 调试开关

跳过预训练 accuracy gate：

```bash
SKIP_PRETRAIN_GATE=1 STAGE=smoke NPROC=8 \
  bash scripts/bash/synbios_probes.sh single ddp latest
```

忽略 smoke/pilot/formal 的阶段依赖：

```bash
IGNORE_STAGE_PREREQUISITE=1 STAGE=formal NPROC=8 \
  bash scripts/bash/synbios_probes.sh single ddp latest
```

关闭 validation class coverage 强制检查：

```bash
REQUIRE_COVERAGE=0 STAGE=smoke NPROC=8 \
  bash scripts/bash/synbios_probes.sh single ddp latest
```

这些开关适合排错或消融，不建议用于正式对照实验。

## 8. Router Analysis

`formal` Probe 成功后默认依次分析六个属性：

```text
birth_date
birth_city
university
major
company
company_city
```

默认使用 `cuda:0`。指定设备：

```bash
ANALYSIS_DEVICE=cuda:3 STAGE=formal NPROC=8 \
  bash scripts/bash/synbios_probes.sh single ddp latest
```

关闭 Router 分析：

```bash
RUN_ROUTER_ANALYSIS=0 STAGE=formal NPROC=8 \
  bash scripts/bash/synbios_probes.sh single ddp latest
```

已有的属性 JSON 会被跳过，因而中断后可以续跑。

## 9. 一键运行两个正式实验条件

该入口会依次执行两个语料条件的预训练、smoke、pilot、formal、Router 分析和最终比较，
开销很大，必须显式确认：

```bash
CONFIRM_FULL_EXPERIMENT=1 \
  bash scripts/bash/synbios_full_experiment.sh single
```

四卡 DDP：

```bash
CONFIRM_FULL_EXPERIMENT=1 NPROC=4 \
  bash scripts/bash/synbios_full_experiment.sh ddp
```

八卡 FSDP：

```bash
CONFIRM_FULL_EXPERIMENT=1 NPROC=8 \
  bash scripts/bash/synbios_full_experiment.sh fsdp
```

主训练使用 8 卡、Probe 只使用 4 卡：

```bash
CONFIRM_FULL_EXPERIMENT=1 NPROC=8 PROBE_GPUS=4 \
  bash scripts/bash/synbios_full_experiment.sh ddp
```

## 10. 主要输出目录

```text
artifacts/synbios_moe/
├── single/
│   └── probe_cache/
├── multi5_permute/
│   └── probe_cache/
├── checkpoints/<run_name>/
└── results/<run_suffix>/probe_pipeline/
    ├── pretrain_gate.json
    ├── smoke/
    ├── pilot/
    └── formal/
        ├── training/
        ├── validation/
        ├── summary/
        ├── logs/
        └── router/
```

每个 stage 的 `pipeline.json` 记录运行状态和输入 identity。重跑时只复用 identity 完全
一致且产物仍然有效的任务；checkpoint、数据、缓存、模型配置或任务定义不一致时，必须
使用新的输出目录，避免混合实验结果。

## 11. Windows PowerShell 调用示例

正式训练推荐在 Linux/CUDA 服务器执行。若要从 Windows PowerShell 调用 Git Bash：

```powershell
Set-Location 'C:\Users\Zhai-Bin Cui\Desktop\GPTScratch\mini-train-sys'

& 'C:\Program Files\Git\bin\bash.exe' `
  'scripts/bash/synbios_moe.sh' `
  'single' `
  'single'
```

运行 Probe：

```powershell
$env:STAGE = 'smoke'
$env:NPROC = '8'
$env:PROBE_GPUS = '4'

& 'C:\Program Files\Git\bin\bash.exe' `
  'scripts/bash/synbios_probes.sh' `
  'single' `
  'ddp' `
  'latest'
```
