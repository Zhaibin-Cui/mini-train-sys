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
# 优先使用系统维护的 /usr/local/cuda 链接；若服务器没有该链接，再填 nvcc 对应目录。
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
test -x "$CUDA_HOME/bin/nvcc"
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

## 3. 明早执行流程：从 batch 回归开始

下面是四张4090、4卡FSDP预训练 checkpoint 的正式顺序。不要直接从 formal 开始，也不要
使用配置文件中的占位 batch。每个耗时步骤都在独立 `tmux` 中运行，并把外层输出同时写入
`artifacts/logs/`；启动和结束时按仓库根目录 `AGENTS.md` 追加 `HISTORY.md`。

### 3.1 登录后的只读检查

```bash
cd /path/to/mini-train-sys
source .venv/bin/activate
source .minitrain-storage.env

git status --short
tmux list-sessions || true
nvidia-smi
df -h . artifacts
python -m pytest -q tests/test_synbios_moe.py tests/test_runtime_logger.py
python -m pytest -q
```

确认四张卡没有其他计算任务。batch 脚本默认在任一卡已使用超过 1 GiB 显存时拒绝启动，
可通过 `PROBE_BENCHMARK_MAX_IDLE_MEMORY_MB` 调整，但不应为了绕过其他训练任务而提高它。

### 3.2 对精确 checkpoint 做四卡 batch 回归

优先使用 `multi5_permute` 的正式 checkpoint；P 会取最长 biography，Q 会取最长姓名，任务
固定为类别数最大的 `university_whole`。脚本会自动建立/验证 probe cache，先测真实
forward/backward/Adam，再独立测 forward-only validation。

```bash
tmux new -s synbios-probe-batch-$(date +%Y%m%d-%H%M)

cd /path/to/mini-train-sys
source .venv/bin/activate
source .minitrain-storage.env
mkdir -p artifacts/logs

CHECKPOINT=artifacts/synbios_moe/checkpoints/synbios_moe_multi5_permute_fsdp_4gpu/<epoch目录>
LOG=artifacts/logs/synbios_probe_batch_$(date +%Y%m%d_%H%M%S).log
bash scripts/bash/synbios_probe_batch_benchmark.sh \
  multi5_permute "$CHECKPOINT" 2>&1 | tee "$LOG"
```

脚本每次写入独立 UTC 时间目录并在退出时打印路径。选择必须同时满足：两张复测卡共同安全、
峰值 CUDA reserved memory 不超过92%、平均吞吐最高。若最优值仍是候选最大值，脚本会退出非零、
保留 `summary.json`，但不会生成 `recommended.env`；扩大对应范围后用新会话重跑，例如：

```bash
P_BATCHES=64,80,96,112,128 \
Q_BATCHES=256,320,384,448,512 \
P_VALIDATION_BATCHES=128,160,192,224,256 \
Q_VALIDATION_BATCHES=512,640,768,896,1024 \
bash scripts/bash/synbios_probe_batch_benchmark.sh \
  multi5_permute "$CHECKPOINT" 2>&1 | tee -a "$LOG"
```

只接受 `summary.json` 中 `ready_for_formal: true` 的目录：

```bash
BENCH_DIR=artifacts/synbios_moe/results/probe_batch_benchmark/multi5_permute/<UTC时间>
python -m json.tool "$BENCH_DIR/summary.json" | less
cat "$BENCH_DIR/recommended.env"
source "$BENCH_DIR/recommended.env"
export PROBE_BATCH_ENV="$BENCH_DIR/recommended.env"
```

同一份 `recommended.env` 必须贯穿两个数据条件的 smoke、pilot、formal。分布式 Bash 入口会
拒绝缺少这四个值的启动；只有显式设置 `ALLOW_DEFAULT_PROBE_BATCHES=1` 才能用论文默认 batch，
该开关仅供调试，不用于正式结果。

### 3.3 500-step smoke：最大分类头与恢复链路

Smoke 使用 P/Q `university_whole`（300类），验证选定 batch 能持续500步、dropout训练、恢复点、
完整 train accuracy、独立 validation 和 progressive cloze gate。两个条件分别执行：

```bash
PROBE_BATCH_ENV="$BENCH_DIR/recommended.env" PROBE_GPUS=4 STAGE=smoke NPROC=4 \
  bash scripts/bash/synbios_probes.sh single fsdp latest

PROBE_BATCH_ENV="$BENCH_DIR/recommended.env" PROBE_GPUS=4 STAGE=smoke NPROC=4 \
  bash scripts/bash/synbios_probes.sh multi5_permute fsdp latest
```

必须确认两个 `smoke/pipeline.json` 都是 `status: completed`，日志中没有 non-finite loss、OOM、
worker failure；不能用 `SKIP_PRETRAIN_GATE`、`IGNORE_STAGE_PREREQUISITE` 或
`REQUIRE_COVERAGE=0` 绕过正式门禁。

### 3.4 3,000-step pilot：全部22个分类器

```bash
PROBE_BATCH_ENV="$BENCH_DIR/recommended.env" PROBE_GPUS=4 STAGE=pilot NPROC=4 \
  bash scripts/bash/synbios_probes.sh single fsdp latest

PROBE_BATCH_ENV="$BENCH_DIR/recommended.env" PROBE_GPUS=4 STAGE=pilot NPROC=4 \
  bash scripts/bash/synbios_probes.sh multi5_permute fsdp latest
```

Pilot 是正式预算前最后一道结构门禁：检查22个训练 `.pt`、22个 validation JSON、summary
任务集合、逐位置准确率、四卡任务分配和 recovery 文件。若 batch 或保存周期需要修改，必须从
新 output 目录重新跑 smoke 和 pilot；pipeline identity 不允许把不同运行协议拼在一起。

### 3.5 30,000-step formal 与结果导出

```bash
PROBE_BATCH_ENV="$BENCH_DIR/recommended.env" PROBE_GPUS=4 STAGE=formal NPROC=4 \
  bash scripts/bash/synbios_probes.sh single fsdp latest

PROBE_BATCH_ENV="$BENCH_DIR/recommended.env" PROBE_GPUS=4 STAGE=formal NPROC=4 \
  bash scripts/bash/synbios_probes.sh multi5_permute fsdp latest

bash scripts/bash/export_test_results.sh
```

`formal` 完成后 Bash 会继续做六个 router analysis。最终比较使用两个 formal validation
目录；报告必须同时写明 MiniTrain MoE backbone、未公开原始词表、硬件选择 batch 这三项
fidelity 差异，不能把绝对数值表述成论文 dense GPT-2 的逐点复现。

### 3.6 实时查看

父终端每10秒显示任务、GPU、worker step、loss、accuracy、显存和 ETA。另开 SSH 终端：

```bash
tail -f artifacts/logs/<本次probe日志>
tensorboard --logdir artifacts/synbios_moe/results \
  --host 127.0.0.1 --port 6606
```

通过 SSH 转发6606后查看 TensorBoard。每个分类器独立记录，父 pipeline 同时提供聚合视图。
完整实验顺序为：

```text
已提交预训练 checkpoint
  → cache/source/checkpoint/GPU preflight
  → 四卡 batch 回归并找到右侧边界
  → smoke（2任务×500）
  → pilot（22任务×3,000）
  → formal（22任务×30,000）
  → held-out validation、router、single vs multi5+permute 比较
```

每个 P/Q Probe 是独立的单 GPU 进程：主模型参数冻结但训练 dropout 保持开启，只训练低秩
输入扰动、归一化层和分类器。多卡只调度不同任务，不做 Probe DDP/FSDP 或跨任务梯度同步。

## 4. 分阶段启动 Probe

命令格式：

```bash
STAGE=<smoke|pilot|formal> NPROC=<预训练卡数> \
  bash scripts/bash/synbios_probes.sh \
  <single|multi5_permute> <single|ddp|fsdp> <latest|checkpoint目录>
```

`NPROC` 用来定位预训练 run name。若主模型由 4 卡训练得到，必须设置 `NPROC=4`；若由
8 卡训练得到，必须设置 `NPROC=8`。

以下分布式示例均假设已经设置：

```bash
export PROBE_BATCH_ENV=artifacts/synbios_moe/results/probe_batch_benchmark/<variant>/<UTC时间>/recommended.env
```

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

四张4090在正式 Probe 前先做容量回归。完整命令和判定规则见第3.2节：

```bash
tmux new -s synbios-probe-batch
bash scripts/bash/synbios_probe_batch_benchmark.sh \
  multi5_permute artifacts/synbios_moe/checkpoints/<run>/<checkpoint>
```

压测先并发运行两份 P 和两份 Q 的训练步，再独立测 forward-only validation。只使用脚本生成且
`ready_for_formal: true` 的 `recommended.env`，不要把机器相关结果写回公共 YAML：

```bash
PROBE_BATCH_ENV=artifacts/synbios_moe/results/probe_batch_benchmark/<variant>/<UTC时间>/recommended.env \
PROBE_GPUS=4 STAGE=formal NPROC=4 \
  bash scripts/bash/synbios_probes.sh multi5_permute fsdp latest
```

默认每100 steps记录 loss/accuracy/梯度/吞吐/显存/GPU利用率，每10秒刷新父终端，每1000
steps原子保存一次 LoRA、分类头、LayerNorm、Adam、RNG和shuffle位置。分别通过
`PROBE_LOG_INTERVAL`、`PROBE_HEARTBEAT_SECONDS`、`PROBE_CHECKPOINT_INTERVAL` 调整。

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
CONFIRM_FULL_EXPERIMENT=1 PROBE_BATCH_ENV=<benchmark目录>/recommended.env \
  bash scripts/bash/synbios_full_experiment.sh single
```

四卡 DDP：

```bash
CONFIRM_FULL_EXPERIMENT=1 NPROC=4 PROBE_BATCH_ENV=<benchmark目录>/recommended.env \
  bash scripts/bash/synbios_full_experiment.sh ddp
```

八卡 FSDP：

```bash
CONFIRM_FULL_EXPERIMENT=1 NPROC=8 PROBE_BATCH_ENV=<benchmark目录>/recommended.env \
  bash scripts/bash/synbios_full_experiment.sh fsdp
```

主训练使用 8 卡、Probe 只使用 4 卡：

```bash
CONFIRM_FULL_EXPERIMENT=1 NPROC=8 PROBE_GPUS=4 \
PROBE_BATCH_ENV=<benchmark目录>/recommended.env \
  bash scripts/bash/synbios_full_experiment.sh ddp
```

分布式一键入口会在预训练前检查 `PROBE_BATCH_ENV`，避免长时间训练结束后才因缺少正式
batch 配置失败。单卡调试可不提供该文件。

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
