# Data 模块

```text
documents.py   原始 txt/jsonl/parquet → Document → clean/chunk
tokenizer.py   ByteLevelBPE 与 tiktoken 的统一接口和 fingerprint
preprocess.py  split、tokenize、TokenShardWriter、manifest/documents.idx
dataloader.py  mmap Dataset、packing-aware Sampler、DataLoader worker
```

Document/chunk 是逻辑文本边界，shard 是物理文件容量边界；一个 shard 可包含很多
chunks，但 writer 不把一个 chunk 跨 shard 写入。训练 sample 是固定 `seq_len+1` token
窗口，input/target 错开一位。

多卡数据分片由 sampler 根据 rank/world size 完成，不由 DDP/FSDP wrapper 完成。完整
说明见 [`docs/data/data_pipeline.md`](../../docs/data/data_pipeline.md)。
