# Retained log archive

本目录是 append-only 的 Git-safe 日志镜像。为避免破坏 `HISTORY.md` 中已有的运行路径，
历史文件保持原名，不按目录移动或删除。

规范化的 81-file 分类、大小、修改时间、mounted/Git-safe 路径和 SHA256 索引位于
[`log_catalog.csv`](../formal_runs/synbios_moe/results/repository_audit_20260724/log_catalog.csv)。
类别包括 probe、pretraining、pretraining validation、benchmark、dataset、
infrastructure 和 engineering validation。正式 run 的用途、命令、状态和产物仍以
`HISTORY.md` 为准；catalog 只解决文件发现与完整性校验。
