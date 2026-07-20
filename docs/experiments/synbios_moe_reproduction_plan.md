# Allen-Zhu bioS P/Q-probe 的 MoE 复刻计划

## 目标与诚实边界

复刻 Allen-Zhu 与 Li《Physics of Language Models: Part 3.1》中 bioS 的知识存储实验，并按项目要求把每层 FFN 换成 MoE。原论文的 probing 实验使用 dense GPT-2，而当前 24 GB RTX 4090 配置使用每卡 batch 8，并允许单机 1/4/8 卡；因此 **MoE 是主架构实验轴，优化与并行协议是额外 fidelity 差异**，不能把所得数值称为论文 dense 数值的严格复现。其余可控量尽量保持一致，并把无法从论文恢复的资产单独列出。

主比较至少运行 `single` 与 `multi5+permute`：前者应只在属性出现前的位置容易被 P-probe 读出，后者应从姓名后的早期位置就更易读出。实验要回答：知识增强是否也会让 MoE 更早把 `person -> attribute` 关联编码进隐状态，以及这种变化是否伴随稳定的 expert 专门化。

## 从论文锁定的控制量

| 项目 | 论文 bioS | 本复刻 |
|---|---:|---:|
| 人物数 | 100,000 | 100,000 |
| 属性 | birth date/city, university, major, company/city | 相同；company city 由 company 决定 |
| 候选数 | date: 200 years × 12 × 28；city 200；university 300；major 100；company 263 | 相同 |
| 文本 | 每人 6 句，约 50 个模板/属性 | 同样的计数与增强规则；措辞为公开描述的再实现 |
| tokenizer/context | GPT-2，512 tokens，条目间 EOS | `tiktoken:gpt2`，512，EOS 边界 |
| packing | 随机采样 BIO 条目、EOS 分隔并拼成 512-token 序列 | 每 epoch 确定性随机重排完整 BIO 条目，再切非重叠 512-token blocks；允许切中条目 |
| 主干 | 12 layers, 12 heads, d=768, RoPE | 相同尺寸；FFN 改为 MoE |
| 预训练 | AdamW, wd=.1, eps=1e-6, lr=1e-3, warmup 1k, cosine 到 1e-4, batch 96, 80k steps；约 540 passes | 每卡 batch 8、无梯度累积；single 540 epochs、multi5+permute 108 epochs，约 4.0B token；LR/warmup 按实际 global batch 缩放 |
| P-probe | 冻结主干；embedding rank 2；LayerNorm+linear；batch 50；30k steps | 相同；AdamW lr=1e-3, wd=.3, eps=1e-6，linear decay |
| Q-probe | 仅 BOS+姓名+EOS；embedding rank 16；BatchNorm+linear；batch 200；30k steps | 相同；AdamW lr=1e-3, wd=.3, eps=1e-6，linear decay |

MoE 主配置采用 8 experts、top-2、dropless routing、SwiGLU、负载均衡系数 0.01 和 router z-loss 0.001。单 expert intermediate size 设为 1024：top-2 SwiGLU 每 token 激活 `2 × 3 × 768 × 1024` 个 FFN 权重，近似论文 GPT-2 MLP 的 `2 × 768 × 3072`；同时绑定输入/输出 embedding。这样 active 参数量约等于论文的 124M，而 MoE total 参数量约为 293M。报告必须同时给出 total parameters 与 active parameters，不能只报其中一个。

还需说明 MiniTrain 主干本身是 Llama 风格的 RMSNorm/SwiGLU、无 bias 投影和全 head RoPE，而论文 probing 主干是改用 1/4 rotary dimension 的 GPT-2。项目现有模型没有 GPT-2 LayerNorm/GELU block，因此这里保持论文的宏观尺寸和训练协议，使用 MiniTrain MoE block；这些实现差异会进入 fidelity ledger。若需要论文 dense 数值的逐点复现，应另加 GPT-2 dense control，不能从本 MoE run 反推。

训练不使用梯度累计，因此只有实际 global batch 恰好为 96 时才能匹配论文 batch；optimizer-update 轨迹仍会因 packing 和实现差异而不同。本复刻优先保持总 token/人物曝光预算：`single:multi5+permute` 使用严格的 `540:108 = 5:1` epoch 比例。运行时按 `actual_global_batch / 96` 线性缩放峰值 LR，并反向缩放 warmup，使 warmup 覆盖的 token 数近似不变。例如单卡 batch 8 使用 LR `8.333e-5`、warmup 12,000；8 卡每卡 batch 8 使用 global batch 64、LR `6.667e-4`、warmup 1,500。该线性规则是工程折中，不是论文原始设置。

论文给出了每类模板数量，但没有随论文发布完整模板表、姓名/城市/学校/专业/公司原始清单。实现会固定生成器版本、seed、候选数、依赖关系和模板数，并保存 manifest/hash；这能做到可重复的机制复刻，但不是原始数据逐条复原。

## 实现阶段

1. `experiments/synbios_moe/data.py`
   - 确定性生成 100k 个唯一英文全名和六属性 profile。
   - 支持 `single`、`fullname`、`permute{1,2,5}`、`multi{2,5}` 及组合。
   - 输出 profiles、biographies、P/Q probe 元数据、token shards 和 manifest。
   - probe 位置由 UTF-8 byte span 与 GPT-2 token bytes 对齐，不依赖脆弱的字符串 token 数猜测。
2. `experiments/synbios_moe/probes.py`
   - 低秩 embedding delta 独立于冻结主干；每个任务独立训练。
   - P-probe 在六个“属性首次出现前”的位置取最后层 hidden；Q-probe 在姓名 EOS 位置取 hidden。
   - birth date 只做 first-token/month；其他五属性同时支持 first-token 与 whole-attribute 分类，共 11 个任务。
3. `experiments/synbios_moe/router_analysis.py`
   - 用 hooks 收集每层 top-k expert、权重、位置和属性标签。
   - 输出 expert load、entropy、姓名/属性位置的 expert mutual information，以及可选的 expert ablation。
4. CLI 与配置
   - `prepare`、`pretrain`、`probe`、`analyze` 分阶段可恢复；所有产物置于 run 目录。
   - token manifest 保存 document offset/length index；SynBio 使用 `packing: randomized_documents`，通用训练默认 `contiguous`。
   - 每次运行保存 resolved config、git revision/dirty 状态、环境、seed、数据与 tokenizer fingerprint、JSONL 指标和 checkpoint。
   - epoch 末 checkpoint 包含模型、AdamW、LR scheduler、GradScaler、RNG 和训练计数器；实验配置每 epoch 保存，最多保留最新两个加一个较老安全锚点，只有最新目录保留额外模型导出。
   - `--resume latest` 恢复完整训练状态并从下一个 epoch 开始；probe/evaluate 只加载 checkpoint 的模型权重。
5. 验证
   - 本机只运行 tiny profile/token-position/model/probe smoke tests。
   - 先运行 smoke 配置，再用分布式 benchmark 在目标 24 GB RTX 4090 上确认 batch 8 的显存余量；若 OOM，应降低每卡 batch，运行时会按比例调整学习率/warmup，不改变 epoch/token 预算。

## 判定与报告

核心结果是 11 个任务的 P-probe `6 positions × accuracy` 矩阵和 Q-probe accuracy，分别按训练/验证人物拆分报告；人物划分必须先于 biography augmentation，避免同一人物的改写跨集合泄漏。额外报告预训练 loss、attribute-token next-token accuracy、router load/entropy、expert/attribute normalized mutual information、总参数和 active 参数估算。

Probe 使用确定性的 50%/50% 人物划分；论文 Appendix E 对 birthdate whole-attribute 不成立的解释给出 probe 训练人物数为 `N/2`，但没有完整写出 validation 划分和 checkpoint 选择协议。因此本项目把另一半人物作为固定 held-out validation，这是保守的实现选择，不宣称为论文逐项规定。所有同一人物的多模板/排列版本固定留在同一侧，并在缓存阶段报告 validation 类别是否全部被 train 覆盖。

一次成功的机制复刻应看到 `multi5+permute` 相比 `single` 在早期 P 位置和 Q-probe 上显著提高；若没有，先检查预训练是否已把属性 token 拟合到高准确率、probe split/位置是否正确、MoE 是否发生 expert collapse，再讨论架构结论。

## 依据

- Allen-Zhu & Li, *Physics of Language Models: Part 3.1: Knowledge Storage and Extraction*, [arXiv:2309.14316](https://arxiv.org/abs/2309.14316)（尤其 Appendix A/B/C/E/F）。
- 作者项目页：[Physics of Language Models — Part 3.1](https://physics.allen-zhu.com/part-3-knowledge/part-3-1)。
- 项目现有 `MiniMoETransformer`、dropless top-k router 与 fused MoE 路径；复刻直接复用这些实现，避免另造一套模型栈。
