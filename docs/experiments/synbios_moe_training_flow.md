# SynBioS MoE：数据、训练和 probe 全流程

## 1. 实验控制量

核心科学对照只有两个语料条件，人物表和事实完全一致：

| 条件 | 每人 biography | 句序 | epochs | 约总 token |
|---|---:|---|---:|---:|
| `single` | 1 | 固定属性顺序 | 540 | 3.999B |
| `multi5+permute` | 5 | 每篇独立打乱 | 108 | 4.001B |

当前完整 100,000 人生成数据测得每遍约 7,405,102 与 37,046,556 token。
540/108 使两边人次曝光和总 token 几乎相同，也接近 Allen-Zhu 论文
`80,000 × 96 × 512 = 3.932B` 的预算。这里不用梯度累计。

切换 single/DDP/FSDP 不改 epochs，只改变每卡 batch、global batch 和 optimizer
update 数。SynBioS 正式配置固定采用论文 LR `1e-3`、warmup 1,000 step 与 cosine
floor `1e-4`；实际 global batch 不为 96 是单独记录的 fidelity 差异。通用 batch
缩放机制详见 [`distributed_training.md`](../training/distributed_training.md)。

## 2. 代码地图

```text
experiments/synbios_moe/data.py       Profile、模板、augmentation、JSONL
scripts/synbios_moe.py                prepare/cache/probe/validate/pipeline/analyze 命令
minitrain/data/documents.py            JSONL 读取、清洗、字符切块
minitrain/data/preprocess.py           tokenizer 与 token shards
minitrain/data/dataloader.py           Dataset/Sampler/DataLoader
scripts/train.py                       组装训练运行时
minitrain/train/{trainer,runner}.py     单步和 epoch 循环
minitrain/train/checkpoint.py           DCP 保存/恢复、probe 模型导出
experiments/synbios_moe/probe_data.py  一次性probe mmap缓存与覆盖检查
experiments/synbios_moe/probe_pipeline.py 阶段计划、任意GPU数队列、结果汇总
experiments/synbios_moe/{evaluation,probes,router_analysis}.py
```

推荐阅读顺序就是上述顺序。数据底层的详细解释见
[`data_pipeline.md`](../data/data_pipeline.md)。

## 3. Profile 如何生成

`generate_profiles(num_people=100_000, seed=1337)` 先构造名字、城市、学校、专业、
公司等有限候选池。`_expanded_names()` 用稳定数字后缀扩充姓名，例如基础池
`Alden ... Mina` 会继续产生 `Alden1 ... Mina19`。

每个 `Profile` 固定：

```text
person_id, full_name, birth_date, birth_city,
university, major, company, company_city, probe split
```

两种语料用相同 seed 和人数，生成的 `profiles.jsonl` 必须逐字相同。augmentation
只改变 biography 的表达，不重新抽取事实。公司和公司城市共享索引，保证关系稳定。

## 4. Biography 与 fidelity

`render_biography()` 为六个属性各选择一个模板，再拼成文章。模板同时存在
`{subject}` 与 `{pronoun}`，因为最终句序确定后只有文章第一句使用人物完整姓名，
后续句用代词。

- `single`：六个属性保持固定顺序，每人生成一篇；第一句使用真实姓名。
- `multi5+permute`：每人生成五篇；每篇重新选模板并打乱六句；打乱后的第一句
  使用真实姓名，其余句用代词。因此 birthday 若被 permute 到第一句，也使用真实姓名。

这与原论文 fidelity 一致：姓名承担篇章锚点，但不会在每句话机械重复。模板
`"{subject} calls {value} a birthplace."` 并非 single 禁用；是否选中由该属性的
模板随机选择决定，`subject` 最终按句子位置解析为姓名或代词。

随机性由 seed、person_id、variant、sample 共同派生，所以可复现且每个 augmentation
独立。`attribute_spans` 记录六个事实在最终字符串中的字符区间，供 token 定位。

## 5. 落盘格式

```text
artifacts/synbios_moe/<variant>/
├── profiles.jsonl       # 一行一人，probe 标准答案
├── biographies.jsonl    # 一行一篇 biography，训练预处理输入
├── biographies.txt      # 便于人工查看
├── manifest.json        # 生成参数、版本、数量和 SHA256
└── token_shards/
    ├── manifest.json
    ├── train/shard_XXXXX.bin
    ├── train/documents.idx
    └── validation/documents.idx
```

`biographies.jsonl` 每一行被 `iter_documents()` 解析成一个 `Document`。这里不是
`TokenShardWriter` 猜测“一行一个 bio”：JSONL reader 明确以一行作为记录边界。
正常 biography 远小于 `max_document_chars=100_000`，因此通常：

```text
一行 biography = 一个 Document = 一个 chunk = 一条 documents.idx
```

`max_document_chars` 按 Python 字符串 `len()`/切片统计 Unicode code point，不是 byte，
也不是 tokenizer token。超长 Document 会尽量在目标字符数前的换行/空白边界切块；
实在没有边界才硬切。每个 chunk 独立 tokenize，并写入 boundary/EOT token。

chunk 是逻辑文档边界，shard 是物理文件容量边界。writer 在写入前检查剩余 token
容量；放不下完整 chunk 就先封口当前 shard，再写下一个。因此 shard 通常不会恰好
100% 填满，换来 chunk 不跨 shard 和稳定 `documents.idx`。若单个 chunk 大于 shard
容量，预处理应直接报错，不能静默丢词。对 SynBio 来说保证每篇 biography 小于
`max_document_chars`，就能让一篇对应一个 chunk；真正“不丢词”还依赖 reader、清洗、
tokenizer、writer 的数量校验，不能只靠这个参数声称完全保证。

## 6. DataLoader 如何消费

`RandomizedDocumentBlockDataset` 根据 `documents.idx` 读取完整逻辑 chunk，按 epoch
确定性打乱文档顺序，再打包为固定 512-token block。`RandomizedDocumentSampler`
负责把 block index 分给不同 rank；`DataLoader` 负责 worker、prefetch 和 pinned memory。

`pin_memory=True` 后，训练端 `.to(cuda, non_blocking=True)` 可让 CPU→GPU DMA 与当前
GPU kernel 重叠。它不会让同一 GPU 同时训练两个 batch，也不会增加模型计算负载；
只会占用少量 pinned host memory、copy engine 和目标 batch 的显存。真正的重叠需要
worker/prefetch 提前准备数据以及 CUDA stream 调度条件满足。

每个 epoch 调用 sampler 的 `set_epoch(epoch)`，所以顺序变化但可复现。分布式
`drop_last` 可能丢掉不足以平均分给所有 rank 的尾部极小部分；要做逐 token 完全相同
的对照需记录 manifest 和 sampler 长度。

## 7. 配置组织

```text
configs/synbios_moe/
├── model.yaml
├── base.yaml                    # reference optimizer/LR/checkpoint
├── variants/
│   ├── single.yaml              # 数据路径 + 540 epochs
│   └── multi5_permute.yaml      # 数据路径 + 108 epochs
├── strategies/
│   ├── single.yaml
│   ├── ddp.yaml
│   └── fsdp.yaml
└── runs/                        # 单卡及显式 4/8 卡组合
```

`extends` 深度合并这些层。每个 run name 都包含 variant 与 strategy，日志和 checkpoint
不会互相覆盖。单卡 24 GB RTX 4090 入口：

```bash
bash scripts/bash/synbios_moe.sh single single
bash scripts/bash/synbios_moe.sh multi5_permute single
```

多卡试验：

```bash
NPROC=4 bash scripts/bash/synbios_moe.sh single ddp
NPROC=8 bash scripts/bash/synbios_moe.sh single ddp
NPROC=4 bash scripts/bash/synbios_moe.sh single fsdp
NPROC=8 bash scripts/bash/synbios_moe.sh single fsdp
```

## 8. Epoch checkpoint、恢复与 probe

训练、evaluate、P/Q probe 和 router analysis 现在共享 `ProgressReporter`，终端都显示
当前/总 batch（或 step）、ETA、吞吐与显存，JSONL/TensorBoard 保存完整数值。CLI 可用
`--log-dir`、`--log-interval`、`--no-tensorboard` 和 `--quiet` 控制实验阶段日志。

主训练 TensorBoard 会把纯 token CE 与加权 aux/z loss 分开，并记录每层×每 expert 的
选中比例和概率热力图。判断事实学习优先使用纯 CE；判断 MoE 是否 collapse 再看 load CV、
dead expert、entropy 和固定色标热力图，不能用总 loss 的变化替代两类判断。

主训练使用全局梯度范数安全阈值 `5.0`。日志中的 `grad_norm` 是裁剪前值，
`grad_clip_coefficient` 是本 step 实际缩放比例，`grad_clip_fraction` 是本日志区间的触发比例。

每个完整 epoch 保存一次，保留最新两个滚动点和一个每10 epoch 轮换的较老安全锚点。
最多只有3个完整 DCP+Adam 恢复目录，并且只有最新目录保留额外 `model.pt`。checkpoint 是带 `COMMITTED` 标记的
目录，包含 DCP 模型/Adam、scheduler、GradScaler、step/epoch/tokens、每 rank RNG 和
resolved config。`--resume latest` 从下一个未完成 epoch 继续；若中断发生在 epoch
中间，会从上一个完整 epoch 重做本轮。
若最新滚动点损坏或训练发散，使用 `--resume safety` 从带 `SAFETY` 标记的锚点恢复。
服务器短入口等价命令是：

```bash
RESUME=safety NPROC=8 bash scripts/bash/synbios_moe.sh single ddp
```

配置同时导出 `model.pt`。evaluate、P-probe、Q-probe、router analysis 只加载这个
完整模型文件，不读取 Adam；probe 随后创建自己的小 optimizer。这样训练恢复完整，
分析又不会因为 Adam 的两个 moment tensor 占用额外内存。

## 9. 训练后的检查

每个预训练模型执行：

- 1 次原始 biography progressive-cloze gate，先确认模型能直接生成被挖掉的真实事实；
- 11 个 P-probe：六个 first-token 加五个 whole-value，观察事实何时可读；
- 11 个 Q-probe：只给姓名，检查 person→fact 是否已编码；
- 6 个 router analysis：统计 expert load、entropy 和属性/expert NMI。

两种预训练条件共 2 次大模型训练、22 个 P、22 个 Q 和 12 个 router analysis。
科学结论应按 loss → attribute evaluate → Q → P → router 的顺序判断，避免把模型
未收敛或 expert collapse 误解为知识组织现象。

## 10. 当前 Probe 执行流程

正式 probe 不再用 shell 串行重复 tokenize。每个 variant 先运行一次：

```bash
python scripts/synbios_moe.py cache-probes \
  --data artifacts/synbios_moe/single \
  --output artifacts/synbios_moe/single/probe_cache \
  --require-coverage
```

随后按 `smoke(500) → pilot(3000) → formal(30000)` 运行。每一阶段先在原始 biography
上逐步挖空生成并检查 strict field accuracy，再把相互独立的任务放入单机设备队列；一张卡同样使用该队列，
只是并发度为1。训练落盘后由 `validate-probe` 重新加载权重，在人物级 held-out split 上
复算结果，最后生成 tidy CSV/JSON。

完整架构、1卡/任意N卡命令、显式GPU编号、产物和两个主实验的比较命令见
[`synbios_moe_probe_pipeline.md`](synbios_moe_probe_pipeline.md)。

服务器上一键完成两个实验条件时使用：

```bash
CONFIRM_FULL_EXPERIMENT=1 NPROC=8 \
  bash scripts/bash/synbios_full_experiment.sh ddp
```

该入口只负责编排，所有训练、恢复、gate、任务调度和汇总仍进入可单独测试的 Python/阶段
入口。想逐步观察时继续使用 `synbios_moe.sh` 和 `synbios_probes.sh`。
