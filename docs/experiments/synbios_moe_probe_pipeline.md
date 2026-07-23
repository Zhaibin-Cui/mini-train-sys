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
  └─ 单任务训练、完整train/validation accuracy和健康指标

experiments/synbios_moe/probe_checkpoint.py
  └─ 不含backbone的原子恢复点与精确shuffle恢复

experiments/synbios_moe/probe_benchmark.py
  └─ 最长样本batch容量压测与跨GPU安全值汇总

experiments/synbios_moe/probe_pipeline.py
  ├─ 任意设备列表解析
  ├─ 每卡一个进程的任务队列
  ├─ smoke/pilot/formal 任务计划
  └─ validation JSON → summary/comparison CSV

scripts/synbios_moe.py
  └─ cache-probes / probe / validate-probe / probe-pipeline /
     benchmark-probe-batches / validate-probe-cache / summarize-probes

scripts/bash/synbios_probes.sh
  └─ Linux 服务器的短入口：定位 checkpoint、建缓存、启动阶段

scripts/bash/synbios_probe_batch_benchmark.sh
  └─ 四卡并发P/Q容量回归
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
├─ p_positions.npy              每篇 biography 从左到右的6个观察位置
├─ p_profile_indices.npy        biography → profile
├─ q_tokens.bin                 BOS+姓名+EOS token
└─ q_offsets.npy                Q 变长序列边界
```

二进制 token 文件通过 mmap 只读共享；每个进程不再持有重复的全部 Python token 列表。不能缓存 backbone hidden states，因为 probe 的 trainable embedding delta 每一步都会改变 Transformer 输入。

这里没有复用预训练的 `shuffle_window` loader。预训练 loader 面向 token shard，会在窗口内打乱、拼接文档；probe 必须保留一整篇 biography 及其 6 个 P 位置，所以使用 `CachedProbeDataset + DataLoader`：训练集以固定 seed 做整条样本级 `shuffle=True`，validation 使用 `shuffle=False`。恢复点同时保存当前 epoch 的 shuffle 状态和已经消费的 batch 数，因此中断续跑不会换一套样本顺序。

`p_positions.npy` 的列严格表示论文的 `P0...P5`：先按属性 span 在最终 biography
中的字符起点排序，再映射到 GPT-2 token 位置。因此，即使 `permute` 改变了句子顺序，
`P0` 仍然是正文中第一个属性出现前的位置，而不是固定指代 `birth_date`。probe cache
格式版本为 2；`synbios_probes.sh` 会自动重建旧版本缓存，手工调用 CLI 时使用
`cache-probes --force`。

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

## 4. Tokenizer、标签、分类头与 loss

### 4.1 P-probe：完整 biography 与六个无泄漏边界

数据生成先把结构化属性填进模板，并在最终 Unicode 字符串上记录六个精确
`[start, end)` span；这一步还没有建立 first-token 类别。构建 probe cache 时，
`GPT2Codec` 才用 `tiktoken:gpt2` 编码整篇 biography。Python span 以 Unicode code point
计数，GPT-2 BPE 以 UTF-8 byte 工作，因此代码先把字符起点转成 byte 起点，再累计
`decode_single_token_bytes()` 得到每个 token 的 byte 终点。

P 位置只纳入满足 `token_end <= attribute_byte_start` 的完整 token。若 GPT-2 把属性前空格和
属性开头合成一个 token（例如 `" Cambridge"`），该跨界 token 整体位于观察位置之后；不会
为了对齐而破坏 BPE，也不会让 hidden state 提前看到属性 byte。模型 attention 使用 causal
mask，所以传入完整 biography 不代表早期位置能读取未来 token。P0 至 P5 按最终正文中的
六个 span 从左到右排列；较晚位置能看到已经真实出现过的事实，这是论文要测的 progressive
disclosure，而不是泄漏。

### 4.2 Q-probe：只输入姓名

Q 输入严格为 `EOS + tiktoken(full_name) + EOS`，在最后一个 EOS 读取最后层 hidden。
输入中没有 biography 和属性文本，因此不存在属性边界混 token。P 和 Q 使用同一套标签语义，
但拥有完全独立的 embedding delta、normalization、linear head、AdamW 和 checkpoint。

### 4.3 first-token 类别何时建立

First-token 类别不是在 `write_dataset()` 生成 `profiles.jsonl` 时建立，而是在
`cache-probes` 阶段从结构化 profile 建立。除生日外，标签定义为：

```python
first_token_label = str(gpt2.encode(" " + exact_attribute_value)[0])
```

显式前导空格固定 GPT-2 的 word-start 语义，不受 biography 模板前面的单词或标点影响；多个
完整属性若拥有同一个首 token，会正确合并成一个 first-token 类别。生日 first 直接取月份，
不建立不可学习的完整日期分类。Whole 标签则是 profile 中完整、精确的 Unicode 属性字符串。

每个任务先对全部 profile 标签去重排序，形成 `class_names` 和稳定的 `label -> integer ID`
映射；随后把每个人的 11 个整数标签写入 `profile_labels.npy`。分类头只输出 `M` 个 logits，
`argmax` 得到整数 ID，再由同一 `class_names` 解释；训练和验证都不从 biography 中 decode
属性字符串。Cache validation 会检查任务顺序、类别唯一性、label 范围、人物索引、offset 和
P 位置边界，并可强制要求 validation 类别均在 train 人物中出现。

当前生成器完整候选池在 GPT-2 tokenizer 下得到：

| 任务 | 类别数 | 任务 | 类别数 |
|---|---:|---|---:|
| birth_date_first | 12 | birth_city_first | 21 |
| birth_city_whole | 200 | university_first | 20 |
| university_whole | 300 | major_first | 20 |
| major_whole | 100 | company_first | 20 |
| company_whole | 263 | company_city_first | 21 |
| company_city_whole | 200 |  |  |

正式运行的权威类别数始终是对应 cache `manifest.json -> tasks[].class_names` 的实际长度；它由
该次 `profiles.jsonl` 中真正出现的值决定。P/Q 各训练这 11 个任务，所以总计 22 个独立分类器。

### 4.4 P/Q loss 与论文训练状态

单个 P 任务产生 `[B, 6, M]` logits。同一个人物标签扩展到六个位置后展平为 `6B` 个等权
cross-entropy 项，因此六个位置共享一套参数、一次反向传播；11个标签任务之间不合并 loss。
Q 产生 `[B, 1, M]`，同一实现退化成普通 batch cross entropy。训练期间 probe 处于
`train()`：冻结 backbone 参数但保留论文要求的 backbone dropout；验证时统一 `eval()`。
P 使用 rank-2 embedding delta + LayerNorm，Q 使用 rank-16 embedding delta + BatchNorm；
二者均使用 AdamW (`lr=1e-3, wd=.3, eps=1e-6`)、无 warmup，并从首步 1.0 精确线性衰减到
最后一步 0.0。

正式 pipeline 只按 P/Q 分别设置训练和验证 batch，不再为 first/whole 拆更多 batch。最大
300 类分类头相对 backbone activation 很小；容量回归固定使用 `university_whole`，得到的值
覆盖其他任务，避免22套难以复现且收益很小的配置。

## 5. 三阶段门控

默认配置：

| 阶段 | 任务 | steps | 前置条件 |
|---|---:|---:|---|
| `smoke` | university whole（最大300类）的 P 和 Q | 500 | 预训练 gate |
| `pilot` | 全部 22 个任务 | 3,000 | smoke 完成 |
| `formal` | 全部 22 个任务 | 30,000 | pilot 完成 |

每个阶段开始前，pipeline 在最多 10,000 篇原始 biography 上运行 progressive cloze
门禁：精确删除六个真实属性 span，按原文顺序 greedy 填回，并把较早预测放回后续上下文。
默认要求严格 `micro_field_accuracy` 不低于 0.90，避免在主模型尚不能直接生成事实时浪费
P/Q probe 预算。模糊字符相似度会记录但不参与放行。可用 `--gate-threshold` 修改；仅调试
调用链时才用 `--skip-gate`。

门禁缓存仍为 `<output>/pretrain_gate.json`。旧 teacher-forced 门禁不会被复用；pipeline
protocol version 已更新，因此旧 stage 也不会被误认为满足新门禁。

阶段成功标志是：

```text
<output>/<stage>/pipeline.json  且 status == "completed"
```

正式阶段默认检查 pilot，pilot 默认检查 smoke。`--ignore-prerequisite` 只用于已通过等价外部检查的特殊情况。

## 6. 任意 GPU 数运行

Probe 使用任务并行，而不是把一个分类器做 DDP：每张 GPU 加载一份只读 backbone，同时训练
一个独立的 P/Q 任务；完成后从公共队列领取下一个任务。四张卡因此同时推进4个分类器，
22个任务动态负载均衡，没有跨卡梯度通信。

### 四卡 batch 容量回归

正式实验前在服务器的 `tmux` 中运行一次四卡并发压测。脚本分别在两张卡复测 P 和 Q，
先用最长 biography 样本执行真实 forward/backward/Adam 测训练容量，再用 forward-only
测验证容量；每轮只推荐所有复测卡都低于默认 92% 峰值显存且吞吐最高的 batch：

```bash
tmux new -s synbios-probe-batch
cd /path/to/mini-train-sys
source .venv/bin/activate
source .minitrain-storage.env
bash scripts/bash/synbios_probe_batch_benchmark.sh \
  multi5_permute artifacts/synbios_moe/checkpoints/<run>/<checkpoint>
```

每次结果写入独立时间戳目录：

```text
artifacts/synbios_moe/results/probe_batch_benchmark/<variant>/<UTC时间>/
├─ summary.json
├─ recommended.env
├─ p/q 原始 JSON
└─ logs/
```

选择规则是：`university_whole` 最大分类头、最长输入、真实训练/验证计算、两张复测卡共同
安全、峰值 CUDA reserved memory 不超过默认92%，最后在共同安全集合中选择平均吞吐最高值。
若推荐值仍是候选最大值，脚本会保留 `summary.json`、拒绝生成 `recommended.env` 并退出，
因为搜索尚未找到右侧边界。只有 P/Q × training/validation 四组均有两份复测、均有推荐值且
都不在搜索边界时，`ready_for_formal` 才为 true。

正式阶段通过 `PROBE_BATCH_ENV=<目录>/recommended.env` 复用四个值。batch 会进入 pipeline
identity，不同 batch 的已有结果不会被误复用。
候选范围可通过 `P_BATCHES`、`Q_BATCHES`、`P_VALIDATION_BATCHES` 和
`Q_VALIDATION_BATCHES` 调整；如果推荐值落在候选范围最右端，应扩大对应范围后再测一次。

### Bash 短入口

单卡：

```bash
STAGE=smoke NPROC=1 bash scripts/bash/synbios_probes.sh single single latest
STAGE=pilot NPROC=1 bash scripts/bash/synbios_probes.sh single single latest
STAGE=formal NPROC=1 bash scripts/bash/synbios_probes.sh single single latest
```

分布式 Probe 先加载同一次容量回归文件。以下示例是8卡预训练 checkpoint，并使用8张卡调度
不同 probe 任务：

```bash
export PROBE_BATCH_ENV=artifacts/synbios_moe/results/probe_batch_benchmark/<variant>/<UTC时间>/recommended.env
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
  --p-batch-size <benchmark-p> --q-batch-size <benchmark-q> \
  --p-validation-batch-size <benchmark-p-val> \
  --q-validation-batch-size <benchmark-q-val> \
  --require-coverage
```

`--num-gpus` 可以是 1 到当前可见 GPU 数之间的任意整数，不只支持 1/4/8。也可以用 `--devices cuda:0,cuda:2` 明确指定，此时不要同时传 `--num-gpus`。

## 7. 独立 validation 实验

pipeline 的训练进程结束时完整扫描 train split，生成 `train_accuracy`；随后由独立 validation
进程从保存的 `.pt` 恢复 probe，再扫描 held-out split。这避免重复 validation，也验证最终
落盘权重确实可恢复。单独调用 `probe` 时仍可保留训练结束后的 validation。

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

P 的 `validation_accuracy` 是按正文从左到右排列的6个观察位置；Q 是一个姓名结束位置。validation 只读取 probe 权重和主模型权重，不读取主训练 Adam 状态。

## 8. 后处理与两个主实验比较

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

## 9. 中断、重跑和产物

```text
probe_pipeline/
├─ pretrain_gate.json
├─ smoke/
├─ pilot/
└─ formal/
   ├─ training/*.json       训练曲线和完整train-split accuracy
   ├─ training/*.pt         probe权重；训练任务完成标志
   ├─ recovery/*.pt         LoRA/分类头/Adam/RNG轻量恢复点
   ├─ validation/*.json     独立恢复后的held-out结果
   ├─ router/*.json         Bash formal入口保留的6个MoE分析
   ├─ logs/*.log            每个子进程的完整日志
   ├─ summary/{json,csv}
   └─ pipeline.json         设备、耗时、成功/失败状态
```

重跑同一命令时：

- 已有训练 `.pt` 的任务跳过训练；
- 未完成任务若有匹配 recovery，会恢复优化器、RNG 和精确 shuffle batch 位置；
- 已有 validation JSON 的任务跳过验证；
- 失败或缺失产物的任务重新进入队列；
- 一个任务失败后仍记录其他任务结果，最终 `pipeline.json` 标记 failed。

每个 stage 的 `pipeline.json` 保存 protocol identity，它绑定 dataset manifest、probe cache
manifest、model config、checkpoint `model.pt` 内容、seed、steps、batch/保存配置和任务列表。只有 identity
完全一致才会复用已有输出；改变任何一项都要求新 output 目录。仅留下 `.pt/.json`、却删除
`pipeline.json` 的孤立产物不会被复用。子进程返回 0 但没有生成约定产物时，任务仍标记 failed。

pretrain gate 也与同一组公共 identity 绑定。更换 checkpoint、数据、cache 或模型配置后会
自动重算，不会复用旧模型的 gate 结果。summary 会检查本 stage 的任务集合完整；两个 run
任务集合不同、人物事实表不同或没有 validation 文件时拒绝生成比较。

## 10. Probe 训练监控

单个 probe 的 `probe_train` 事件写入 JSONL 和 TensorBoard，包含：loss、学习率、
区间准确率、六个 P-probe 位置各自的准确率、全局梯度范数、DataLoader 等待时间与占比、
平均 step 时间、序列长度、吞吐、显存、NVML GPU利用率和 ETA。训练 batch accuracy 是
当前日志区间的真实训练准确率；任务结束后 `probe_train_evaluation` 再报告完整 train split
准确率。`grad_norm` 只用于发现数值异常，不做梯度裁剪。validation 使用
`probe_validation` 事件，并记录总体及逐位置 running accuracy。

`probe-pipeline` 父终端每10秒读取四个 worker 的结构化事件，显示 queued/running/completed/
failed、当前任务与 GPU、worker step、loss、accuracy、显存和任务级 ETA。默认每100 steps
写一次完整指标，每1000 steps 原子保存一次轻量 recovery，频率均可配置。关键落盘位置：

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
训练指标频率，用 `--heartbeat-seconds` 控制长任务的顶层心跳频率，用
`--no-tensorboard` 关闭 TensorBoard。

TensorBoard 指向整个 probe output 根目录即可同时看到 pipeline 总览和每个分类器的独立曲线：

```bash
tensorboard --logdir artifacts/synbios_moe/results/<run>/probe_pipeline \
  --host 127.0.0.1 --port 6606
```

通过 SSH 转发 `6606` 后在本机浏览器查看。不同分类器使用独立 run 目录，不会互相覆盖；
父 pipeline 还会按 `phase/task` 汇集关键 worker 指标，方便比较四张卡的实时进度。

端到端验证使用 `tests/synbios_moe_end_to_end.ipynb` 和专用的
`configs/synbios_moe/probe_pipeline_notebook_smoke.yaml`。notebook 把单 probe 训练、独立
`validate-probe`、两任务 pipeline、状态/事件检查和正式 smoke/pilot/formal 命令拆成不同
单元格。该配置只有 3 steps，只验证调用与监控契约，不能替代正式预算。
