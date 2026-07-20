# 从独立文本到 MiniTrain DataLoader

本文解释三件事：nanochat 的数据/分词架构实际做了什么；byte-level BPE、
tiktoken 与 token 存储宽度分别是什么；MiniTrain 新增的数据预处理 pipeline
如何从零散文档一直走到训练 batch。

第一次阅读如果只想理解本项目，可先看第 3 节，再回到第 1/2 节补 tokenizer 和参考
项目背景。最短代码路径是 `documents.py → tokenizer.py → preprocess.py → dataloader.py`。

实施前计划见 [`data_preprocessing_plan.md`](../design-notes/data_preprocessing_plan.md)。代码入口：

- `minitrain/data/documents.py`：reader、清洗、chunk；
- `minitrain/data/tokenizer.py`：自训 byte-BPE 与现成 tiktoken 的统一接口；
- `minitrain/data/preprocess.py`：split、tokenize、shard、manifest；
- `minitrain/data/dataloader.py`：memory-map token shards；
- `scripts/prepare_data.py`：命令行入口。

## 1. nanochat 调研结论

当前本机 nanochat 已从 FineWeb-Edu 切到 ClimbMix。`nanochat/dataset.py` 下载
已经重打包好的 zstd Parquet；最后一个 shard 固定作为 validation，其余为 train。
`dev/repackage_data_reference.py` 记录了上游离线流程：shuffle 文档，按大约 2.5 亿
字符聚合，每 1024 文档一个 row group，再写成约 100 MB 压缩 shard。

tokenizer 训练和运行链路是：

```text
Parquet text
  -> 每篇最多取 10,000 字符，总计最多 2B 字符
  -> rustbpe 根据固定 regex 学 byte-BPE merge ranks
  -> 将同一 pattern + 同一 mergeable ranks + special token ids
     构造成 tiktoken.Encoding
  -> 保存 tokenizer.pkl
```

预训练时并未预先写 token `.bin`。`nanochat/dataloader.py` 按 Parquet row group
流式读文本，在 worker/rank 上在线批量 tokenize，每篇 prepend BOS，然后用
best-fit 将多篇文档装入 `T+1` 的训练行：能完整放下就装入；没有文档能完整
放下时，选择一篇 crop 到恰好填满。这样无 padding、每行从 BOS 开始，但 nanochat
自己记录了大约 35% token tail 被裁掉这一代价。

### “训练用自训 tokenizer，推理用 tiktoken”会不会训推不一致？

不会，前提是使用 nanochat 的正常 artifact 路径。这里混淆了两个层次：

- `rustbpe` 是**训练 merge rules 的实现**；
- `tiktoken.Encoding` 是持有并执行**同一份词表、同一份 merge ranks、同一 regex、
  同一 special-token id**的运行时。

因此文本在训练/推理时会得到相同 token ids。真正危险的是：预训练使用自训 ranks，
推理却调用 `tiktoken.get_encoding("gpt2")` 或 `cl100k_base`。这时词表和 id 语义
都变了，即便 vocab size 偶然相同也完全不兼容。nanochat 的
`from_pretrained(tiktoken_name)` 是另一条显式模式，不是自训模型推理时的替换动作。

MiniTrain 将这个不变量做得更显式：tokenizer artifact 有 SHA-256 fingerprint；
corpus manifest 记录 fingerprint；`DataConfig.tokenizer_fingerprint` 可在创建
dataloader 时再次强校验。

## 2. Byte-level BPE 原理

### 2.1 从 Unicode 到 byte

文本首先按 UTF-8 变成 byte 序列。例如 ASCII 字符通常是 1 byte，中文字符通常是
3 bytes。byte-level tokenizer 的初始字母表覆盖 256 个 byte，因此不会产生传统
word tokenizer 的 unknown character；任何 UTF-8 文本都有表示。

### 2.2 学 merge

初始时每个 byte 是一个 symbol。训练器反复统计相邻 symbol pair，选择高频 pair
合并成新 symbol：

```text
h e l l o
  高频 (l,l) -> h e ll o
  高频 (he,ll) -> hell o
  高频 (hell,o) -> hello
```

每次 merge 增加一个词表项，直到达到 `vocab_size` 或没有满足 `min_frequency`
的 pair。词表变大通常缩短序列，但 embedding/output head 更大；词表变小则相反。
MiniTrain 的 custom backend 用 Hugging Face `tokenizers` 训练和执行同一个
byte-level BPE artifact，避免训练器与运行时转换带来的额外格式面。

### 2.3 “一个 token 是多少 byte”与“.bin 每个 id 多少 byte”不同

这两个 byte 数经常被混为一谈：

1. 一个 BPE token 解码后覆盖多少 UTF-8 bytes，是数据压缩率问题，长度可变；
2. `.bin` 中一个 token **id** 占多少 bytes，是整数存储格式问题，长度固定。

MiniTrain 根据词表范围选择后者：最大 id 能装进 0..65535 时用 little-endian
`uint16`（2 bytes/id），否则用 `uint32`（4 bytes/id）。dtype、bytes/token、
vocab size 都写入 manifest；不再依靠文件后缀猜测。自训 BPE 默认 32K，因此通常
是 uint16。tiktoken `gpt2` 也是 uint16；超过 65,536 的词表自动切 uint32。

## 3. MiniTrain pipeline

```text
独立 txt/md 文件或 JSONL/Parquet 行
  -> Document(id, text, source, metadata)
  -> 清洗与 exact SHA-256 dedup
  -> 对超长文档做自然边界 chunk（默认无 overlap）
  -> hash(document id, seed) 稳定切 train/validation
  -> [boundary token] + tokenizer.encode(chunk)
  -> 文档对齐的 uint16/uint32 shards
  -> manifest.json
  -> ShardedTokenBlockDataset (np.memmap)
  -> DataLoader
  -> {input_ids: [B,T], targets: [B,T]}
```

### 3.1 文档读取

- `.txt/.text/.md`：一个文件是一篇文档；
- `.jsonl/.jsonlines`：一行是一篇，默认读 `text`，可用 `--text-field` 修改；
- `.parquet`：按 row group 流式读取指定 text column，pyarrow 是可选 data 依赖。

每篇有稳定 id。JSONL 自带 `id` 时优先使用；否则使用绝对 source path + row。
split 对 id 做 SHA-256，所以输入文件换顺序不会让同一文档在 train/validation 间漂移。

### 3.2 清洗范围

默认 baseline 会：统一 CRLF、Unicode NFC；删除异常 C0 control；收敛水平空白和
过多空行；过滤空/过短、字母比例极低、control 比例过高、重复行比例过高的文档；
对清洗后的全文做 exact SHA-256 dedup。每一种拒绝原因有独立 counter。

FineWeb 是大规模 Common Crawl pipeline，包含语言过滤、质量规则、重复内容处理等
经消融验证的步骤；官方数据说明其使用 Datatrove，并对英文概率低于 0.65 的文档
做过滤。MiniTrain 的本地 baseline 只实现不依赖外部模型且可审计的子集，不声称
等价于 FineWeb。近似去重（MinHash/LSH）、语言识别、PII、质量/教育分类器留作
可插拔 stage，而不是写几个未经标定的 regex 冒充工业清洗。

参考：[FineWeb NeurIPS 论文](https://arxiv.org/abs/2406.17557)、
[FineWeb 数据卡](https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/v1.4.0/README.md)、
[FineWeb 技术说明](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html)。

### 3.3 chunk 与文档之间如何处理

超长文档优先在空段、换行、空格处切；只有后半窗口没有自然边界才 hard split。
默认没有 overlap，因为 overlap 会重复训练目标；如果未来加入 overlap，必须同时
加入 loss mask/权重策略。

不同文档绝不先拼成一段 raw string。每个 chunk 先独立 tokenize，再 prepend 一个
boundary token，之后才进入共同 token stream。因此模型可看到明确边界。writer
优先提前结束 shard，避免普通文档被物理 shard 切断。单个 chunk 自身大于 shard
上限时才按 token 切 continuation，并给每个 continuation 再 prepend boundary，
保证每个物理 shard 的第一个训练 block 有显式起点。

### 3.4 manifest

`manifest.json` 是数据集契约，包含：

- format version、dtype、bytes per token、vocab size、boundary id；
- tokenizer fingerprint；
- cleaning 配置和所有 accept/reject counter；
- chunk 策略、split hash/seed/fraction；
- 每个 split 的文档/chunk/token 数；
- 每个 shard 的相对路径、token/byte 数和 SHA-256。

shard 和 `documents.idx` 先写 `.tmp`，执行 flush + fsync 后再原子 rename；其 checksum
以 1 MiB block 流式计算，不把大文件读入 RAM。manifest 当前也是先写临时文件再原子
replace，但 manifest 写入路径尚未显式执行 flush + fsync；因此它具备进程级的完整文件
发布语义，若要求掉电级 durability，还需要补齐文件及父目录 fsync。

### 3.5 DataLoader

DataLoader 训练阶段不做在线 tokenize。它消费随机 token、单文件预 tokenized
数据，或预处理生成的 token shards。三种 `data.source` 如下：

| `data.source` | 数据在哪里 | 加载方式 | 主要用途 |
|---|---|---|---|
| `random` | 按 seed 生成 | 全部在 CPU RAM 中 | 训练链路、kernel 和 smoke test |
| `tokens` | `.pt/.pth/.npy/.bin` 单文件 | 读成一条 CPU token tensor；`.bin` 按 `uint16` 解释 | 兼容小型旧数据和快速实验 |
| `token_shards` | `manifest.json` + 多个 shard | shard 按 worker 懒加载为只读 `np.memmap` | 正式大规模训练 |

`random` 和 `tokens` 都使用 `TokenBlockDataset`；`token_shards` 再由
`data.packing` 选择下面两种打包模式。`random` 和 `tokens` 只支持默认的
`packing: contiguous`。

#### 3.5.1 共同的 causal block 语义

设上下文长度为 T，第 i 个样本的起点是 `i * T`，而不是每次只移动一个
token。每个样本读取 `T+1` 个 token：

```text
token stream:  t0 t1 t2 ... tT t(T+1) ...
sample 0 input:  t0 ... t(T-1)    target: t1 ... tT
sample 1 input:  tT ... t(2T-1)   target: t(T+1) ... t(2T)
```

因此训练 block 的 stride 是 T，相邻的 `T+1` 原始窗口只共享用于 causal
对齐的那一个边界 token，不是 one-token sliding-window 采样。一条 N-token 流产生
`floor((N - 1) / T)` 个完整样本，只忽略 split 最后无法组成完整 causal
block 的 token。

#### 3.5.2 `packing: contiguous`（默认）

`ShardedTokenBlockDataset` 把同一 split 的所有物理 shard 视为一条逻辑连续的 token
流。物理 shard 只是存储和 I/O 边界，不是训练样本边界：一个 block 可以从前一个
shard 开始、到后一个 shard 结束。这避免了每个 shard 各自丢弃尾部；只有整个
split 的最后一小段可能被忽略。固定 token 网格也意味着文档边界不会决定 block
边界；文档可能被 block 切中，但预处理插入的 boundary token 仍保留在流中。

`ShardAwareBlockSampler` 不改变 block 内容，只改变已经固定的 block 出现顺序：

1. 按 `seed + epoch` 确定性打乱物理 shard 对应的 block ranges；
2. 在每个 range 内按 `shuffle_window` 分窗口，打乱窗口顺序；
3. 再打乱每个窗口内的 block，默认每窗口 1024 blocks。

这是 shard-aware 有界 shuffle：它不为全语料构造一个巨大全局 `randperm`，也避免让
磁盘读取在整个数据集上完全随机跳转。`shuffle: false` 时不做上述打乱，按
全局 block index 顺序读取。

#### 3.5.3 `packing: randomized_documents`

该模式用于提高 Allen-Zhu bioS 实验中“文档随机重排”这一项的数据 packing fidelity；
它本身不代表模型、优化器和评测也已达到整篇论文的完整 fidelity。预处理会为每个 split
原子写入 `documents.idx`；每条记录是两个 little-endian `uint64`，表示一篇
boundary-prefixed 文档在原始 split token 流中的全局 `(offset, length)`。manifest
记录索引路径、entries、dtype、columns、byte size 和 SHA-256 checksum。

每个 epoch 的处理顺序是：

1. 用 `seed + epoch` 对完整文档做一次确定性、无放回 permutation；
2. 保持每篇文档的内部 token 顺序，按新文档顺序逻辑拼接 token spans；
3. 从重排后的逻辑流上，仍以 stride T 生成固定 causal blocks。

这不是“每个 sample 对齐一篇 bio”。一个 block 可以包含多篇短文档，也可以切中一篇
文档；但由于文档顺序每个 epoch 变化，被固定 token 网格切中的文档也会变化。
sampler 将一个 block 所需的只读 span 列表传给 worker，因此它能从任意物理
shard 组合该 block，`persistent_workers` 也无需共享可变 epoch 状态。

`shuffle: false` 时不做文档 permutation，退化为按存储顺序 packing。此模式不使用
`shuffle_window`。它必须配合含 `document_index` 的新 manifest；默认
`contiguous` 不依赖该索引，所以旧 manifest 和原有训练仍然可用。

#### 3.5.4 从 shard 到 block：一个可手算的例子

`token`、`bio/document`、`block`、`batch` 和 `shard` 是五个不同层级：

| 名称 | 含义 |
|---|---|
| token | tokenizer 产生的一个整数 ID |
| bio/document | 一篇带 boundary token 的独立文档；SynBio 平均约 75～79 tokens |
| block/sample | 模型的一条 T-token 输入及其右移 targets；T=512 时通常含约 6～7 篇 bio |
| batch | 一次训练 step 交给模型的多条 blocks |
| shard | 磁盘上的二进制 token 容器；10M-token SynBio shard 通常含约 12～13 万篇 bio |

假设 T=4，两个物理文件分别保存 10 个 token：

```text
shard 0: t0 t1 t2 t3 t4 t5 t6 t7 t8 t9
shard 1: t10 t11 t12 t13 t14 t15 t16 t17 t18 t19
```

block i 的全局起点是 `i*T`，所以 block 2 读取全局 `[8, 13)`：

```text
block 2 raw:    t8  t9 | t10 t11 t12
input_ids:      t8  t9 | t10 t11
targets:        t9 t10 | t11 t12
physical read:  shard 0 | shard 1
```

它按起点 8 归入 shard 0 的 sample range，只是为了让 sampler 对每个 block 有唯一的
I/O 调度归属；实际读取不受该归属限制。`contiguous` shuffle 也只重排完整 block 的
出现顺序，例如 `block 2 -> block 0 -> block 3`，绝不会打乱某个 block 内部的 token。

这里的“跨 shard shuffle”也不是所有 blocks 的一次完全全局 `randperm`。实现会先打乱
shard-owned ranges，再处理当前 range 内被打乱的 windows 和 blocks；一个 range 处理完后
才进入下一个。这样仍覆盖整个 split，但不会产生在所有 shard 文件间逐 block 随机跳转的
极端 I/O 模式。

#### 3.5.5 `randomized_documents` 如何做到逻辑重排而不重写 shard

一篇 bio 不对应一个 shard。假设物理布局是：

```text
shard 0: [A:3 tokens][B:4][C:2]
shard 1: [D:5][E:3]
```

`documents.idx` 使用 split 全局坐标记录：

```text
A=(0,3), B=(3,4), C=(7,2), D=(9,5), E=(14,3)
```

某个 epoch 的 permutation 若为 `D -> B -> E -> A -> C`，内存中改变的只是这些
16-byte index records 的顺序：

```text
[(9,5), (3,4), (14,3), (0,3), (7,2)]
```

磁盘仍保持原始 shard 内容，没有生成一份重排后的巨大 token 副本。sampler 对重排后的
文档长度做 prefix sum，将 `block_index*T` 映射到“第几篇文档、文档内第几个 token”，
再产生一个 `PackedBlockSpec`。若 T=6，第一个 block 需要 7 个 raw tokens，规格可能是：

```text
[(9,5), (3,2)] = shard 1 中完整的 D + shard 0 中 B 的前两个 token
```

worker 收到该只读 span 列表后才用 memmap 取出片段，拼成当前小 block。一个 block 因此
可以按 `shard 1 -> shard 0 -> shard 1` 的顺序组合多个 span，而完整 token payload 始终
不会一次性进入 RAM。对于一般语料，index entry 严格说对应预处理后的 document chunk；SynBio 文本远小于
chunk 上限时，通常就是一篇 bio 对应一条 entry。

无放回 permutation 保证每条 document-index entry 在“逻辑排列”中出现一次，但不保证其
所有 token 在一次训练 epoch 中都被消费：逻辑流末尾不足 T+1 的部分、DDP 等长截断以及
`drop_last` 仍可能不进入优化步骤。`shuffle: true` 时这些尾部落到哪些 bio 上会随 epoch
变化；`shuffle: false` 时尾部固定。

#### 3.5.6 为什么 shuffle 前不需要加载 token payload

初始化只读取 manifest 的 shard token counts，并建立累计边界。例如 `[10, 20, 28]`
表示三个 shard 的全局范围为 `[0,10)`、`[10,20)`、`[20,28)`。block 起点和长度由
`block_index*T` 与 `T+1` 直接算出；通过累计边界二分查找即可知道它覆盖哪些文件，不需要
先查看 token 值。

`np.memmap` 只把文件映射到虚拟地址空间。某个 `__getitem__` 真正切片时，操作系统才按页
载入涉及的区域；实现只主动分配当前 block 的 `T+1` 个 `int64` 拼接缓冲区。操作系统可以
做页级预读和 page cache，但 Python 不会因此一次性把完整 shard 复制进 RAM。每个 worker
又用独立的 `max_open_shards` LRU 限制同时打开的映射数量。

#### 3.5.7 DDP、batch 尾部和完整性校验

两种 packing sampler 都让所有 rank 基于同一个确定性全局顺序，再按位置的
`rank/world_size` 取模分配等长、互不重复的子序列。不足 `world_size` 的最后少量
blocks 被丢弃，保证各 rank 的 DataLoader 长度一致。`drop_last: true` 还会丢弃每个
rank 最后不足 `batch_size` 的样本，以保持固定 batch shape。这两种“尾部丢弃”分别发生在
DDP 分区层和 batch 组装层，不是每个物理 shard 丢尾。

`token_shards` 创建 dataset 时校验 manifest version、dtype 和每个 shard 的 byte
size；模型 vocab 必须至少容纳 manifest vocab，显式配置的 tokenizer fingerprint
必须精确相等。每个 worker 最多保留 `max_open_shards` 个 memmap，超出后用 LRU
关闭最久未使用的映射，避免超大语料无限累积打开文件。

manifest 保存 shard 和 document index 的 SHA-256，但当前 DataLoader 启动路径为了避免
每次训练前顺序扫描整个 corpus，只校验文件 byte size，**不会自动重算 SHA-256**。checksum
用于数据生成审计、复制/下载后的显式完整性验证；仅有 checksum 字段不等于训练启动时已经
执行过 checksum verification。

数据并行相关配置如下：

| 配置 | 影响 |
|---|---|
| `num_workers` | 每 rank worker 数；`null` 自动按节点预算分配，0 表示主进程同步加载 |
| `worker_budget` | 自动模式下单机所有 rank 的 worker 总预算 |
| `max_workers_per_rank` | 自动模式下每 rank 上限 |
| `worker_cpu_affinity` | Linux 下让 worker 单线程并尽量绑定独立 CPU |
| `prefetch_factor` | 每个 worker 预取的 batch 数；仅 `num_workers > 0` 时传给 PyTorch |
| `pin_memory` | 将 batch 放到 pinned CPU memory，配合 non-blocking CUDA copy |
| `persistent_workers` | epoch 之间保留 worker；要求 `num_workers > 0` |
| `max_open_shards` | 每个 dataset worker 的 memmap LRU 上限 |
| `drop_last` | 是否丢弃每 rank 最后不完整 batch |
| `shuffle_window` | 仅影响 `contiguous` 的有界 block shuffle |

训练 loop 在每个 epoch 调用 sampler 的 `set_epoch(epoch)`，所以在 seed 固定时，
每次完整运行可复现，不同 epoch 又能获得不同顺序。`TrainingRunner` 将首个训练 epoch
编号为 1，因此实际首次训练使用 `seed + 1`；测试或直接使用 sampler 时仍可显式设置
`epoch=0`，这只影响排列编号，不改变确定性机制。

选型原则：通用预训练使用 `token_shards + contiguous + shuffle: true`；Allen-Zhu
bioS/SynBio 实验使用 `token_shards + randomized_documents + shuffle: true`；
`shuffle: false` 主要用于调试顺序、构造可手算的测试或做明确的 no-shuffle ablation。
对应的核心配置是：

```yaml
# 通用大规模预训练
data:
  source: token_shards
  path: artifacts/my_corpus/manifest.json
  packing: contiguous
  shuffle: true
  shuffle_window: 1024
```

```yaml
# Allen-Zhu bioS/SynBio 文档重排模式
data:
  source: token_shards
  path: artifacts/synbios/manifest.json
  packing: randomized_documents
  shuffle: true
```

## 4. 两种 tokenizer 的完整命令

先安装可选依赖：

```powershell
python -m pip install -e ".[data]"
```

### 4.1 自训 byte-level BPE

```powershell
python scripts/prepare_data.py train-tokenizer raw_docs `
  --output artifacts/my_tokenizer `
  --vocab-size 32768 `
  --max-training-chars 2000000000 `
  --max-document-chars 10000

python scripts/prepare_data.py tokenize raw_docs `
  --tokenizer artifacts/my_tokenizer `
  --output artifacts/my_corpus `
  --max-shard-tokens 10000000 `
  --validation-fraction 0.01 `
  --split-seed 42
```

### 4.2 现成 tiktoken

```powershell
python scripts/prepare_data.py use-tiktoken `
  --encoding gpt2 `
  --output artifacts/gpt2_tokenizer

python scripts/prepare_data.py tokenize raw_docs `
  --tokenizer artifacts/gpt2_tokenizer `
  --output artifacts/gpt2_corpus
```

查看结果：

```powershell
python scripts/prepare_data.py inspect artifacts/my_corpus
```

训练 YAML 可参考 `configs/data_token_shards_example.yaml`，把 manifest 的 fingerprint
复制进配置：

```yaml
data:
  source: token_shards
  path: artifacts/my_corpus/manifest.json
  shuffle: true
  packing: contiguous
  tokenizer_fingerprint: <manifest 中的值>
```

## 5. 与 nanochat 的取舍

| 方面 | nanochat | MiniTrain |
|---|---|---|
| tokenizer | rustbpe 训练，同 ranks 转 tiktoken runtime | custom byte-BPE artifact 或 named tiktoken |
| tokenize 时机 | dataloader 在线 | preprocessing 离线 |
| 文档存储 | zstd Parquet text | 原始文档输入；训练消费 token shards |
| packing | BOS best-fit，满利用率，可能 crop tail | 默认 split 连续 block；可选每 epoch 文档重排后再 packing |
| DDP 单元 | Parquet row group | shard-aware 有界 shuffle 后的等长 block 子序列 |
| 复现契约 | tokenizer pickle + 数据目录约定 | tokenizer fingerprint + versioned manifest + shard checksum |

离线 tokenization 牺牲了“随时换 tokenizer”的便利，换来训练阶段 CPU 更轻、数据量
可精确统计、tokenizer mismatch 可提前失败。`randomized_documents` 在不在线 tokenize、
不改模型和训练 loop 的前提下提供受控的 epoch-level 动态 packing；更复杂的在线数据
mixture 仍可在现有 `Tokenizer`/`Document` 接口上增加 IterableDataset。

## 6. 已验证的最小闭环

仓库 fixture 已真实跑通：4 个 JSONL 文档 → 自训 300-vocab byte-BPE → 4 个
document-aligned uint16 shards → manifest → `build_training_dataloader` → `(2,8)`
inputs/targets。相关 hermetic tests 位于 `tests/test_data_pipeline.py`。
