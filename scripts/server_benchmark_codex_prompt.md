# 给服务器 Codex 的分阶段实验提示词

把下面整段发给服务器上的 Codex。它只负责执行，不允许跳过阶段确认，也不允许为了得到
好看的数字修改比较条件。

---

你现在位于 `mini-train-sys` 仓库。请先完整阅读根目录 `AGENTS.md`，严格遵守 tmux、
挂载盘、日志、`HISTORY.md` 和结果导出规则。当前机器预期为 4 × RTX 4090 24 GB。

总规则：

1. 先检查当前分支、commit、dirty state、四卡型号/显存/拓扑、`.venv`、
   `.minitrain-storage.env`、数据、checkpoint 和 probe cache；不要擅自清理已有产物。
2. 所有长任务必须在独立 tmux 中运行，并把 stdout/stderr 写入
   `artifacts/logs/<timestamp>.log`。启动后确认 pane 仍存活。
3. 每个阶段先把准确命令、预计耗时、我应该回来检查的北京时间告诉我，并问
   “是否开始阶段 N？”。没有得到我的明确同意，不得启动。
4. 阶段完成后检查退出码、JSON/CSV/图片/notebook、错误日志和质量门槛，更新同一条
   `HISTORY.md` 记录，运行 `bash scripts/bash/export_test_results.sh`，再汇报结果。
5. 每完成一个阶段都停下来，告诉我结论、产物路径、失败项（包括 OOM）和下一阶段预计
   时间，再问是否继续。不得自动连续执行。
6. 不得把 OOM、timeout、fallback 或少于要求的重复次数包装成成功；不得挑选性删除
   慢结果。若候选最优值落在扫描边界，扩展一档后重测。
7. 长任务采用低频轮询，不要持续刷新状态：启动后先根据预计耗时等待，通常每
   10–20 分钟检查一次；临近预计完成时间再适当缩短间隔。每次只检查 tmux/进程是否
   存活、GPU 状态和日志末尾 20–50 行，不要反复读取或输出完整日志；状态没有变化时
   不要重复汇报。仅在任务异常、接近完成或需要我决策时提高检查频率。

## 阶段 0：只读预检

先运行短检查，不启动实验：

```bash
cd ~/mini-train-sys
source .venv/bin/activate
source .minitrain-storage.env
git status --short
git rev-parse HEAD
nvidia-smi
nvidia-smi topo -m
python -m pytest tests/test_probe_diagnostics.py \
  tests/test_predicted_first_report.py \
  tests/test_operator_bench_utils.py \
  tests/test_distributed_bench_utils.py -q
python -m ruff check experiments/synbios_moe scripts tests minitrain
```

确认 `multi5_permute` 正式 backbone、formal first probes 和 probe cache 都存在。汇报预检
结论与后续四阶段时间预算，然后先问我是否开始阶段 1。

## 阶段 1：predicted-t1 whole probe

先做精确 stage-two 输入的 batch 搜索，不沿用原 probe batch：

```bash
bash scripts/bash/synbios_predicted_first_batch_benchmark.sh \
  multi5_permute_fsdp_4gpu latest
```

这个脚本分别扫描 P/Q training 和 validation，每种条件用两张独立 GPU 复测，并只在
最优点不是搜索边界时生成 `recommended.env`。预计 30–90 分钟；启动后根据首个 candidate
实测速率修正 ETA，告诉我回来检查的北京时间。

batch 搜索通过后，停下来汇报 P/Q 训练和评测 batch、吞吐、显存峰值和扫描边界，并问我
是否开始同一阶段的正式 pilot。得到同意后：

```bash
source <batch-result-directory>/recommended.env
STEPS=3000 GPUS="0 1 2 3" \
  bash scripts/bash/synbios_predicted_first_whole_pilot.sh \
  multi5_permute multi5_permute_fsdp_4gpu latest
```

预计 1–4 小时，以第一波四个分类器的实测进度更新 ETA。最终必须有 10 个 JSON、10 个
`.pt`、`summary.csv`、`summary.json`，以及 `figures/` 下 P/Q 的 PNG 和 PDF。检查
`summary.json` 的 protocol、condition、identity 和任务数。完成并导出后，问我是否开始
阶段 2。

## 阶段 2：RTX 4090 MoE 算子 scaling

不要直接用 `jupyter nbconvert --inplace`。统一通过安全执行器运行，它会指定正确 kernel、
执行到临时副本、捕获 cell error，并只在整本成功后原子发布：

```bash
python -m ipykernel install --user --name mini-train-sys \
  --display-name mini-train-sys
python scripts/run_server_notebook.py \
  tests/moe_operator_bench_linux_server.ipynb \
  --kernel mini-train-sys \
  --output-dir artifacts/notebooks \
  --timeout-seconds -1
```

外层命令仍须放在 tmux。预计 3–6 小时，启动 15 分钟后先查看
`artifacts/operator_benchmark/rtx4090_24gb/logs/`，再根据已完成 shape 更新 ETA。

该 notebook 的每个 shape 都在独立 CUDA 子进程中运行。必须保留 timeout/OOM case，
核验 `summary.json`、`summary.csv`、raw/logs、三张统一风格图片，以及 executed notebook
输出副本。重点解释：

- project-formal profile 是否跑到 57,344 tokens（112 × 512）；
- Mixtral-class 4096/14336 profile 的安全边界；
- Torch/Triton forward、backward、full-step 的正确性、速度和显存；
- 是否通过 `quality_gate_passed`。

完成并导出后，问我是否开始阶段 3。

## 阶段 3：正式 FSDP launch 的 Torch vs CUDA backend

先构建 RTX 4090 的 head-dim 64、BF16/FP16 CUDA FlashAttention，不要编译无关 bucket：

```bash
export MINITRAIN_CUDA_BUILD_PROFILE=workstation
export MINITRAIN_CUDA_ARCHS=89
export MINITRAIN_CUDA_MAX_JOBS=2
export MINITRAIN_CUDA_VERBOSE=1
python -c "from minitrain.kernels.cuda_ext.build import load_cuda_extension; print(load_cuda_extension())"
python scripts/run_dist_bench.py validate-backend \
  --device cuda:0 \
  --batch-size 2 \
  --sequence-length 512 \
  --output artifacts/distributed_benchmark/synbios_backend_torch_vs_cuda/backend_validation.json
```

编译可能需要 30–120 分钟。编译和测试通过后，使用完全相同的正式 SynBioS model、
FSDP4、BF16、local batch 112 比较 Torch 与 CUDA，各做 3 次：

```bash
python scripts/run_dist_bench.py run \
  --suite backend \
  --strategies fsdp \
  --world-sizes 4 \
  --ops-backends torch triton cuda \
  --local-batch 112 \
  --warmup-steps 10 \
  --measure-steps 30 \
  --repeats 3 \
  --case-config configs/synbios_moe/runs/multi5_permute_fsdp_4gpu.yaml \
  --model-config configs/synbios_moe/model.yaml \
  --output artifacts/distributed_benchmark/synbios_backend_torch_vs_cuda

python scripts/run_dist_bench.py present-backend \
  --input artifacts/distributed_benchmark/synbios_backend_torch_vs_cuda/backend_summary.json \
  --validation artifacts/distributed_benchmark/synbios_backend_torch_vs_cuda/backend_validation.json \
  --output artifacts/distributed_benchmark/synbios_backend_torch_vs_cuda/presentation
```

训练 benchmark 本身预计 30–60 分钟（不含首次编译）。Torch/CUDA 是目标对比，额外的
Triton control 用来隔离“原生 CUDA attention”相对 fallback 栈的增量。汇报吞吐
speedup、step time 下降比例、allocated/reserved 显存变化、均值和标准差。必须检查每个 CUDA case 的
`backend_dispatch.attention.native_cuda > 0` 且 `fallback == 0`。说明 CUDA backend
目前原生替换的是 attention，其余算子沿 CUDA→Triton→Torch fallback；不能把总体差异
描述成“所有算子都是手写 CUDA”。完成并导出后，问我是否开始阶段 4。

## 阶段 4：FSDP 1 卡与 4 卡 scaling 复核

先复核已有证据，不要默认重跑：

- `results/benchmarks/synbios_moe_fsdp4/weak_b64/weak_summary.json`
- `results/benchmarks/synbios_moe_fsdp4/weak_b64/presentation/`
- `reports/distributed_bench.md`
- `HISTORY.md` 中 “SynBioS exact-model FSDP weak-scaling verification”
- 运行代码 `scripts/run_dist_bench.py`
- notebook `tests/distributed_server_benchmark.ipynb`

确认现有两次重复是否仍与当前报告一致：单卡约 93,302 tok/s，四卡约
344,254 tok/s，即 3.69×、平均弱扩展效率 92.24%。这已经回答“是否四倍”：接近但不是
四倍。

复核预计 10–20 分钟。先把复核结果告诉我，再问我是否真的要按当前 commit 重跑。只有我
明确要求重跑，才使用相同 local batch 64、1/4 GPU、至少两次重复的 weak suite；不得
用 local batch 112 的 capacity 数据冒充单卡/四卡公平 scaling。

最后给出四阶段总表：实验问题、准确比较条件、状态、核心指标、质量门槛、原始证据、
图片、日志、executed notebook、`HISTORY.md` 条目和仍存在的限制。先让我审核，再准备
提交，不要自行 push。

---
