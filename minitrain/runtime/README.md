# Runtime 模块

Runtime 把配置转换成可运行对象，但不执行 epoch：

- `config.py`：typed config、验证和递归 `extends`；
- `factory.py`：构造 ops backend、model、parallel strategy；
- `device.py`：按 `LOCAL_RANK` 选择设备；
- `scaling.py`：无梯度累计的 global-batch LR/warmup 缩放；
- `logger.py`：console、JSONL、TensorBoard 标量/直方图/专家热力图组合；
- `monitoring.py`：统一进度、ETA、吞吐、跨 rank 标量、小型矩阵和显存统计；
- `provenance.py`：Git/环境元数据。

新增 YAML 字段应先加入 dataclass 和验证，再由明确的消费者读取，避免配置存在但运行时
静默忽略。

训练、SynBioS 评估和 probe 都发送同一种结构化 event；展示方式由 logger 决定。字段含义和
checkpoint 恢复流程见 [`docs/training/monitoring_and_recovery.md`](../../docs/training/monitoring_and_recovery.md)。
