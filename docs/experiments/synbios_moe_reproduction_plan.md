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
| 预训练 | AdamW, wd=.1, eps=1e-6, lr=1e-3, warmup 1k, cosine 到 1e-4, batch 96, 80k steps；约 540 passes | 无梯度累积；single 540 epochs、multi5+permute 108 epochs，约 4.0B token；硬件 batch 可不同，但固定使用论文 LR/warmup，不做线性放大 |
| P-probe | 冻结主干但开启 dropout；embedding rank 2；LayerNorm+linear；batch 50；30k steps | 参数化协议、dropout、30k steps 与优化器相同；batch 由4090最长样本容量回归确定并记录 |
| Q-probe | 仅 BOS+姓名+EOS；冻结主干但开启 dropout；embedding rank 16；BatchNorm+linear；batch 200；30k steps | 输入、参数化协议、dropout、30k steps 与优化器相同；batch 由容量回归确定并记录 |

MoE 主配置采用 8 experts、top-2、dropless routing、SwiGLU、负载均衡系数 0.01 和 router z-loss 0.001。单 expert intermediate size 设为 1024：top-2 SwiGLU 每 token 激活 `2 × 3 × 768 × 1024` 个 FFN 权重，近似论文 GPT-2 MLP 的 `2 × 768 × 3072`；同时绑定输入/输出 embedding。这样 active 参数量约等于论文的 124M，而 MoE total 参数量约为 293M。报告必须同时给出 total parameters 与 active parameters，不能只报其中一个。

还需说明 MiniTrain 主干本身是 Llama 风格的 RMSNorm/SwiGLU、无 bias 投影和全 head RoPE，而论文 probing 主干是改用 1/4 rotary dimension 的 GPT-2。项目现有模型没有 GPT-2 LayerNorm/GELU block，因此这里保持论文的宏观尺寸和训练协议，使用 MiniTrain MoE block；这些实现差异会进入 fidelity ledger。若需要论文 dense 数值的逐点复现，应另加 GPT-2 dense control，不能从本 MoE run 反推。

训练不使用梯度累计，因此只有实际 global batch 恰好为 96 时才能匹配论文 batch；optimizer-update 轨迹仍会因 packing 和实现差异而不同。本复刻优先保持总 token/人物曝光预算：`single:multi5+permute` 使用严格的 `540:108 = 5:1` epoch 比例。为避免大 batch 线性缩放偏离论文且引发不稳定，所有 SynBioS 正式配置固定使用论文峰值 LR `1e-3`、warmup 1,000 step 和 cosine floor `1e-4`。global batch 不为 96 仍是明确记录的 fidelity 差异。

论文给出了每类模板数量，但没有随论文发布完整模板表、姓名/城市/学校/专业/公司原始清单；作者 FAQ 也明确说明 Part 3 代码和 bioS 数据尚未公开。实现会固定生成器版本、seed、候选数、依赖关系和模板数，并保存 manifest/hash；这能做到可重复的机制复刻，但不是原始数据逐条复原。

这项缺失尤其影响 first-token 随机基线：候选总数与论文一致，不代表候选字符串经过 GPT-2
tokenization 后的首 token 分布一致。当前再实现完整池得到的 first-token 类别数为
`12/21/20/20/20/21`（生日、出生城市、大学、专业、公司、公司城市），而论文图中的 majority
baseline 反映其未公开原始词表。报告可以比较本项目 `single` 与 `multi5+permute` 的机制差异，
但不能把 first-token 绝对准确率逐点声称为论文数值复现。Whole 类别数和公司城市依赖关系仍按
公开协议保持。该差异必须随正式结果一并报告，不能通过事后调整类别表隐藏。

用户已明确选择让正式 probe 使用服务器吞吐最优 batch，而不是强制论文的 P=50/Q=200。
这会改变固定30k steps下的样本曝光量，是另一项有意的 fidelity 差异；两个主数据条件必须使用
同一份 `recommended.env`，并在 `HISTORY.md` 和结果 provenance 中记录，才可做公平对照。

## 实现阶段

1. `experiments/synbios_moe/data.py`
   - 确定性生成 100k 个唯一英文全名和六属性 profile。
   - 支持 `single`、`fullname`、`permute{1,2,5}`、`multi{2,5}` 及组合。
   - 输出 profiles、biographies、P/Q probe 元数据、token shards 和 manifest。
   - probe 位置由 UTF-8 byte span 与 GPT-2 token bytes 对齐，不依赖脆弱的字符串 token 数猜测。
2. `experiments/synbios_moe/probes.py`
   - 低秩 embedding delta 独立于冻结主干；每个任务独立训练。
   - P-probe 在六个“属性首次出现前”的位置取最后层 hidden，并按最终 biography 从左到右编号为 `P0...P5`；Q-probe 在姓名 EOS 位置取 hidden。
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
   - 先运行 smoke 配置，再用分布式 benchmark 在目标 24 GB RTX 4090 上确认安全 batch；若 OOM，应降低每卡 batch，但仍保持论文学习率和 warmup，不改变 epoch/token 预算。

## 判定与报告

核心结果是 11 个任务的 P-probe `6 positions × accuracy` 矩阵和 Q-probe accuracy，分别按训练/验证人物拆分报告；人物划分必须先于 biography augmentation，避免同一人物的改写跨集合泄漏。额外报告预训练 loss、attribute-token next-token accuracy、router load/entropy、expert/attribute normalized mutual information、总参数和 active 参数估算。

Probe 使用确定性的 50%/50% 人物划分；论文 Appendix E 对 birthdate whole-attribute 不成立的解释给出 probe 训练人物数为 `N/2`，但没有完整写出 validation 划分和 checkpoint 选择协议。因此本项目把另一半人物作为固定 held-out validation，这是保守的实现选择，不宣称为论文逐项规定。所有同一人物的多模板/排列版本固定留在同一侧，并在缓存阶段报告 validation 类别是否全部被 train 覆盖。

一次成功的机制复刻应看到 `multi5+permute` 相比 `single` 在早期 P 位置和 Q-probe 上显著提高；若没有，先检查预训练是否已把属性 token 拟合到高准确率、probe split/位置是否正确、MoE 是否发生 expert collapse，再讨论架构结论。

## 依据

- Allen-Zhu & Li, *Physics of Language Models: Part 3.1: Knowledge Storage and Extraction*, [arXiv:2309.14316](https://arxiv.org/abs/2309.14316)（尤其 Appendix A/B/C/E/F）。
- 作者项目页：[Physics of Language Models — Part 3.1](https://physics.allen-zhu.com/part-3-knowledge/part-3-1)。
- 项目现有 `MiniMoETransformer`、dropless top-k router 与 fused MoE 路径；复刻直接复用这些实现，避免另造一套模型栈。
