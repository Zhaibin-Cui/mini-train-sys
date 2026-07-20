# Scripts 入口说明

| 入口 | 状态与用途 |
|---|---|
| `train.py` | 当前通用训练入口 |
| `prepare_data.py` | 当前通用 tokenizer/shard 预处理入口 |
| `synbios_moe.py` | SynBioS prepare/cache/probe/validate/pipeline/summarize/analyze；统一实验入口 |
| `run_dist_bench.py` | 当前单机多卡 benchmark CLI/worker |
| `powershell/` | Windows 启动与清理 |
| `bash/setup_storage.sh` | 将实验产物和编译缓存安全映射到 Linux 挂载盘 |
| `bash/setup_server.sh` | Linux/NVIDIA 服务器一键创建环境、安装与预检 |
| `bash/synbios_moe.sh` | Linux 上运行一个 SynBioS 预训练条件 |
| `bash/synbios_probes.sh` | 对一个 checkpoint 分阶段运行 probe 与 router |
| `bash/synbios_full_experiment.sh` | 两个条件的预训练→全部 probe→router→最终比较 |
| `bash/` 其他入口 | Linux 单卡、多卡 benchmark 与清理 |
| `eval.py` | 通用 scaffold，尚未实现 |
| `sample.py` | 通用 scaffold，尚未实现 |

Python 文件负责可测试逻辑，Bash/PowerShell 只做环境变量、路径和 launcher 编排。
不要在 shell 脚本中复制训练实现。

正式全流程入口有意要求 `CONFIRM_FULL_EXPERIMENT=1`，避免误触发两次大模型训练。它仍然
逐个调用可独立复跑的预训练与 probe 入口，不隐藏 smoke/pilot/formal 门禁。

SynBioS 的预训练、checkpoint 恢复、分阶段 Probe、GPU 分配和完整实验命令汇总见
[`synbios_moe_runbook.md`](synbios_moe_runbook.md)。

服务器首次安装、PyTorch CUDA wheel、NCCL/BF16 验收和正式实验前检查见
[`../docs/guides/server_setup.md`](../docs/guides/server_setup.md)。
