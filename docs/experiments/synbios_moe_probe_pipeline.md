# SynBioS Probe：缓存、分阶段训练、多卡调度与验证

本文是 P/Q probe 的当前运行手册。它说明程序边界、正式命令、任意单机 GPU 数调度、独立 validation 和结果后处理。论文差异仍以
[`synbios_moe_reproduction_plan.md`](synbios_moe_reproduction_plan.md) 为准。

## 1. 不变的实验定义

每个预训练 checkpoint 运行 22 个彼此独立的 probe：

```text
11 个标签任务 × (P-probe + Q-probe) = 22 个训练任务
```

11 个标签任务是 6 个 first-token 和 5 个 whole-attribute；生日没有 whole 分类。每个任务拥有独立的 embedding delta、normalization、linear head 和 AdamW，不共享梯度。多卡只并行独立进程，不把它们改造成 multi-task probe，因此不改变论文 probing 定义。

## 2. 程序架构和入口

```text
configs/synbios_moe/probe_pipeline.yaml
  └─ smoke/pilot/formal 的步数、任务和前置阶段

experiments/synbios_moe/probe_data.py
  ├─ 11 个论文任务定义
  ├─ 一次性 tokenizer/位置/标签缓存
  ├─ train/validation whole-label 覆盖检查
  └─ CachedProbeDataset（mmap，只读）

experiments/synbios_moe/probes.py
  ├─ P/Q 单任务 Dataset 的旧兼容路径
  ├─ rank embedding delta 与 AttributeProbe
  └─ 单任务训练和 accuracy

experiments/synbios_moe/probe_pipeline.py
  ├─ 任意设备列表解析
  ├─ 每卡一个进程的任务队列
  ├─ smoke/pilot/formal 任务计划
  └─ validation JSON → summary/comparison CSV

scripts/synbios_moe.py
  └─ cache-probes / probe / validate-probe / probe-pipeline /
     validate-probe-cache / summarize-probes

scripts/bash/synbios_probes.sh
  └─ Linux 服务器的短入口：定位 checkpoint、建缓存、启动阶段
```

Python CLI 是功能入口，Bash 只处理服务器路径、环境变量和 checkpoint 定位。

## 3. 为什么提前生成全部 probe 数据

旧路径每训练一个 P-probe 都重新读取和 tokenize 10 万或 50 万篇 biography。新缓存对每个数据条件只执行一次 GPT-2 tokenization，同时生成全部 11 个任务所需标签。

```text
artifacts/synbios_moe/<variant>/probe_cache/
├─ manifest.json                 来源、任务、类别和覆盖报告
├─ profile_labels.npy           [人物, 11任务]
├─ profile_splits.npy           人物级 train/validation
├─ p_tokens.bin                 所有 biography token，int32
├─ p_offsets.npy                P 变长序列边界
├─ p_positions.npy              每篇 biography 的6个观察位置
├─ p_profile_indices.npy        biography → profile
├─ q_tokens.bin                 BOS+姓名+EOS token
└─ q_offsets.npy                Q 变长序列边界
```

二进制 token 文件通过 mmap 只读共享；每个进程不再持有重复的全部 Python token 列表。不能缓存 backbone hidden states，因为 probe 的 trainable embedding delta 每一步都会改变 Transformer 输入。

### 生成和检查缓存

```bash
python scripts/synbios_moe.py cache-probes \
  --data artifacts/synbios_moe/single \
  --output artifacts/synbios_moe/single/probe_cache \
  --require-coverage

python scripts/synbios_moe.py cache-probes \
  --data artifacts/synbios_moe/multi5_permute \
  --output artifacts/synbios_moe/multi5_permute/probe_cache \
  --require-coverage

python scripts/synbios_moe.py validate-probe-cache \
  --probe-cache artifacts/synbios_moe/single/probe_cache
```

`--require-coverage` 要求 validation 中每个 whole/first 类别都至少在 probe train 人物中出现一次。正式 100k 数据应该满足；100 人 smoke 数据通常不会满足，因此小数据调试时省略它。

缓存记录原数据 manifest。`probe` 或 pipeline 若收到不匹配的 `--data` 与 `--probe-cache` 会直接拒绝运行。

## 4. 三阶段门控

默认配置：

| 阶段 | 任务 | steps | 前置条件 |
|---|---:|---:|---|
| `smoke` | birth-city whole 的 P 和 Q | 500 | 预训练 gate |
| `pilot` | 全部 22 个任务 | 3,000 | smoke 完成 |
| `formal` | 全部 22 个任务 | 30,000 | pilot 完成 |

每个阶段开始前，pipeline 在最多 10,000 篇 biography 上运行 teacher-forced attribute-token evaluation。默认要求 micro accuracy 不低于 0.90，避免在未学会事实的 checkpoint 上浪费 probe 预算。可用 `--gate-threshold` 修改；仅调试调用链时才用 `--skip-gate`。

阶段成功标志是：

```text
<output>/<stage>/pipeline.json  且 status == "completed"
```

正式阶段默认检查 pilot，pilot 默认检查 smoke。`--ignore-prerequisite` 只用于已通过等价外部检查的特殊情况。

## 5. 任意 GPU 数运行

### Bash 短入口

单卡：

```bash
STAGE=smoke NPROC=1 bash scripts/bash/synbios_probes.sh single single latest
STAGE=pilot NPROC=1 bash scripts/bash/synbios_probes.sh single single latest
STAGE=formal NPROC=1 bash scripts/bash/synbios_probes.sh single single latest
```

8卡预训练 checkpoint，使用8张卡跑 probe：

```bash
STAGE=smoke NPROC=8 bash scripts/bash/synbios_probes.sh single ddp latest
STAGE=pilot NPROC=8 bash scripts/bash/synbios_probes.sh single ddp latest
STAGE=formal NPROC=8 bash scripts/bash/synbios_probes.sh single ddp latest
```

probe 的卡数不必等于预训练卡数。例如预训练 checkpoint 名称来自8卡 DDP，但只用3张卡做 probe：

```bash
STAGE=pilot NPROC=8 PROBE_GPUS=3 \
  bash scripts/bash/synbios_probes.sh single ddp latest
```

指定不连续GPU：

```bash
STAGE=pilot NPROC=8 PROBE_DEVICES=cuda:1,cuda:3,cuda:6 \
  bash scripts/bash/synbios_probes.sh single ddp latest
```

`NPROC` 用来定位预训练 run；`PROBE_GPUS` 或 `PROBE_DEVICES` 控制 probe 实际并发度。调度器始终保证每张指定 GPU 同时最多运行一个 probe 进程。

`formal` P/Q/validation 完成后，Bash 入口默认继续在 `cuda:0` 顺序运行6个 router analysis，保持旧实验范围。可用 `ANALYSIS_DEVICE=cuda:3` 换卡；只想运行 P/Q 时设置 `RUN_ROUTER_ANALYSIS=0`。

### 直接 Python 入口

任意 N 张可见 GPU：

```bash
python scripts/synbios_moe.py probe-pipeline \
  --stage pilot \
  --data artifacts/synbios_moe/single \
  --probe-cache artifacts/synbios_moe/single/probe_cache \
  --model-config configs/synbios_moe/model.yaml \
  --checkpoint artifacts/synbios_moe/checkpoints/<run>/<checkpoint> \
  --output artifacts/synbios_moe/results/<run>/probe_pipeline \
  --devices auto --num-gpus 4 \
  --require-coverage
```

`--num-gpus` 可以是 1 到当前可见 GPU 数之间的任意整数，不只支持 1/4/8。也可以用 `--devices cuda:0,cuda:2` 明确指定，此时不要同时传 `--num-gpus`。

## 6. 独立 validation 实验

训练命令结束时仍会计算一次 validation accuracy，保持旧接口兼容；pipeline 随后从保存的 `.pt` 重新加载 probe，再独立运行一次 held-out validation。这一步验证落盘权重可恢复，并将训练与最终结果文件解耦。

单任务手工验证：

```bash
python scripts/synbios_moe.py validate-probe \
  --data artifacts/synbios_moe/single \
  --probe-cache artifacts/synbios_moe/single/probe_cache \
  --model-config configs/synbios_moe/model.yaml \
  --checkpoint artifacts/synbios_moe/checkpoints/<run>/<checkpoint> \
  --probe-checkpoint artifacts/synbios_moe/results/<run>/probe_pipeline/formal/training/p_university_whole.pt \
  --device cuda:0 \
  --output artifacts/synbios_moe/results/<run>/probe_pipeline/formal/validation/p_university_whole.json
```

P 的 `validation_accuracy` 是6个观察位置；Q 是一个姓名结束位置。validation 只读取 probe 权重和主模型权重，不读取主训练 Adam 状态。

## 7. 后处理与两个主实验比较

pipeline 每个阶段自动生成：

```text
<stage>/summary/summary.json
<stage>/summary/summary.csv
```

完成 `single` 与 `multi5_permute` formal 后，统一比较：

```bash
python scripts/synbios_moe.py summarize-probes \
  --run single=artifacts/synbios_moe/results/single_single/probe_pipeline/formal/validation \
  --run multi5_permute=artifacts/synbios_moe/results/multi5_permute_single/probe_pipeline/formal/validation \
  --output artifacts/synbios_moe/results/comparison
```

输出：

```text
comparison/summary.json    完整任务结果和来源
comparison/summary.csv     tidy格式：run/task/position/accuracy
comparison/comparison.csv  后续run相对第一个run的accuracy delta
```

P-probe 在 CSV 中每个任务有6行，Q-probe 每个任务有1行。将 `single` 放在第一个 `--run`，`delta > 0` 就表示 augmentation 条件更高。
新 validation 文件同时携带数据 manifest。比较器会检查两个 run 的 `profiles.jsonl` SHA256；人物事实表不一致时拒绝生成 delta，避免把数据变化误认为 augmentation 效果。

## 8. 中断、重跑和产物

```text
probe_pipeline/
├─ pretrain_gate.json
├─ smoke/
├─ pilot/
└─ formal/
   ├─ training/*.json       训练曲线和训练末尾validation
   ├─ training/*.pt         probe权重；训练任务完成标志
   ├─ validation/*.json     独立恢复后的held-out结果
   ├─ router/*.json         Bash formal入口保留的6个MoE分析
   ├─ logs/*.log            每个子进程的完整日志
   ├─ summary/{json,csv}
   └─ pipeline.json         设备、耗时、成功/失败状态
```

重跑同一命令时：

- 已有训练 `.pt` 的任务跳过训练；
- 已有 validation JSON 的任务跳过验证；
- 失败或缺失产物的任务重新进入队列；
- 一个任务失败后仍记录其他任务结果，最终 `pipeline.json` 标记 failed。

每个 stage 的 `pipeline.json` 保存 protocol identity，它绑定 dataset manifest、probe cache
manifest、model config、checkpoint `model.pt` 内容、seed、steps 和任务列表。只有 identity
完全一致才会复用已有输出；改变任何一项都要求新 output 目录。仅留下 `.pt/.json`、却删除
`pipeline.json` 的孤立产物不会被复用。子进程返回 0 但没有生成约定产物时，任务仍标记 failed。

pretrain gate 也与同一组公共 identity 绑定。更换 checkpoint、数据、cache 或模型配置后会
自动重算，不会复用旧模型的 gate 结果。summary 会检查本 stage 的任务集合完整；两个 run
任务集合不同、人物事实表不同或没有 validation 文件时拒绝生成比较。

## 9. Probe 训练监控

单个 probe 的 `probe_train` 事件同时写终端、JSONL 和 TensorBoard，包含：loss、学习率、
区间准确率、六个 P-probe 位置各自的准确率、全局梯度范数、DataLoader 等待时间与占比、
平均 step 时间、序列长度、吞吐、显存和 ETA。`grad_norm` 只用于发现数值异常，不做梯度裁剪，
因此不会改变论文 probe 的优化协议。validation 使用 `probe_validation` 事件，并记录总体及逐位置
running accuracy。

`probe-pipeline` 还会显示 training/validation 两个阶段的 queued、running、completed、failed
任务数和任务级 ETA。关键落盘位置：

```text
<stage>/pipeline.json                 当前原子快照；进程中断后仍能看到最后任务状态
<stage>/pipeline_events.jsonl         started/heartbeat/finished 追加事件
<stage>/operation_logs/...            pipeline 的 JSONL/TensorBoard
<stage>/logs/train_<task>.log          每个训练子进程的完整 stdout/stderr
<stage>/logs/validation_<task>.log     每个验证子进程的完整 stdout/stderr
```

正常观察 pipeline 时不必打开 22 个子进程日志；只有 `tasks_failed > 0` 时，再按照
`pipeline.json` 中的任务名查看对应日志。`--quiet-workers` 只压低子进程控制台事件，pipeline
总进度仍保留；`--quiet` 才关闭 pipeline 自己的终端输出。可用 `--log-interval` 控制单个 probe
训练指标频率，用 `--heartbeat-seconds 30` 控制长任务的顶层心跳频率，用
`--no-tensorboard` 关闭 TensorBoard。

端到端验证使用 `tests/synbios_moe_end_to_end.ipynb` 和专用的
`configs/synbios_moe/probe_pipeline_notebook_smoke.yaml`。notebook 把单 probe 训练、独立
`validate-probe`、两任务 pipeline、状态/事件检查和正式 smoke/pilot/formal 命令拆成不同
单元格。该配置只有 3 steps，只验证调用与监控契约，不能替代正式预算。
