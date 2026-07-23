# Scripts 入口说明

| 入口 | 状态与用途 |
|---|---|
| `train.py` | 当前通用训练入口 |
| `prepare_data.py` | 当前通用 tokenizer/shard 预处理入口 |
| `synbios_moe.py` | SynBioS prepare/cache/probe/validate/diagnostics/pipeline/summarize/analyze；统一实验入口 |
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

Q-whole 的 oracle 首 token 干预和 bad-case MoE route 分支分析分别使用
`validate-probe-oracle-first-token` 与 `validate-probe-bad-case-routes`；协议、命令和
产物结构见 [`../docs/experiments/synbios_moe_probe_pipeline.md`](../docs/experiments/synbios_moe_probe_pipeline.md#11-q-whole-的两个推理验证)。

两个 formal 条件完成后，`report-formal-study` 会严格校验 pipeline、checkpoint、dataset、
cache、profile 和任务身份，再从 44 个独立 validation JSON 重建主表与正式图；canonical
命令见 [`../reports/synbios_moe/probes/README.md`](../reports/synbios_moe/probes/README.md#复现正式图表)。

两个 inference-only val 完成后，`report-probe-diagnostics` 会校验 formal/diagnostic
checkpoint、dataset、cache、probe、split 和 raw hashes，并生成全部 P/Q-whole 原结果、
oracle 结果、五属性×12层 route contrast 的统一图表；canonical 命令见
[`../reports/synbios_moe/probes/diagnostics/README.md`](../reports/synbios_moe/probes/diagnostics/README.md#重建命令)。

`audit-synbios-repository` 会从 raw manifests 一直校验到正式 checkpoint、cloze/probe/
diagnostic 身份与运行配置，同时生成 dataset lineage、path contract 和 log catalog。
它只做 CPU/storage 校验，不训练或推理；规范输出见
[`../results/formal_runs/synbios_moe/results/repository_audit_20260724/summary.json`](../results/formal_runs/synbios_moe/results/repository_audit_20260724/summary.json)。

正式全流程入口有意要求 `CONFIRM_FULL_EXPERIMENT=1`，避免误触发两次大模型训练。它仍然
逐个调用可独立复跑的预训练与 probe 入口，不隐藏 smoke/pilot/formal 门禁。

SynBioS 的预训练、checkpoint 恢复、分阶段 Probe、GPU 分配和完整实验命令汇总见
[`synbios_moe_runbook.md`](synbios_moe_runbook.md)。

服务器首次安装、PyTorch CUDA wheel、NCCL/BF16 验收和正式实验前检查见
[`../docs/guides/server_setup.md`](../docs/guides/server_setup.md)。
