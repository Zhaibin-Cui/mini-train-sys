# SynBioS 全属性 GPT-2 Token 分布

## 编码规则

统计使用与 Probe 一致的 GPT-2 tiktoken，并对属性值编码：

```python
tokens = encoding.encode(" " + value)
```

前导空格不能省略。Biography 模板在属性前都包含空格，而 GPT-2 是 byte-level BPE，通常会把
空格与后续单词合并为一个 token。项目中的非生日 first-token 标签正是：

```python
codec.encode(" " + value)[0]
```

Whole Probe 不使用 token 序列作为标签，而是把完整原字符串作为一个分类类别。生日 first
标签当前直接使用月份字符串；在真实 biography tokenization 中，月份仍然对应带前导空格的
month token，并且 12 个月与 12 个 first token 一一对应。

以下结果由 seed 1337 的 100,000 个正式 profiles 计算。Single 与 multi5+permute 使用同一批
profiles；multi 只把每个属性值重复到五篇 biography，因此归一化分布不变。

## 总体分布

| 属性 | 实际唯一值 | First-token 类别 | 平均 tokens | 中位数 | P95 | 范围 |
|---|---:|---:|---:|---:|---:|---:|
| Birth date | 51,858 | 12 | 4.361 | 4 | 5 | 4–5 |
| Birth city | 200 | 21 | 3.906 | 4 | 4 | 3–4 |
| University | 300 | 20 | 2.934 | 3 | 3 | 2–3 |
| Major | 100 | 20 | 2.097 | 2 | 3 | 1–4 |
| Company | 263 | 20 | 3.031 | 3 | 4 | 2–4 |
| Company city | 200 | 21 | 3.915 | 4 | 4 | 3–4 |

“实际唯一值”按 100k profiles 中真正出现的值统计。生日理论空间更大，但本次 profile
样本中出现了 51,858 个不同完整日期。

## Token 长度直方图

| 属性 | 1 token | 2 tokens | 3 tokens | 4 tokens | 5 tokens |
|---|---:|---:|---:|---:|---:|
| Birth date | — | — | — | 63.897% | 36.103% |
| Birth city | — | — | 9.443% | 90.557% | — |
| University | — | 6.623% | 93.377% | — | — |
| Major | 15.176% | 63.941% | 16.927% | 3.956% | — |
| Company | — | 6.787% | 83.300% | 9.913% | — |
| Company city | — | — | 8.504% | 91.496% | — |

这说明当前的 first/whole 二分非常粗：除 15.18% 的 major 外，绝大多数完整属性都包含多个
token。

## 各属性的位置结构

### Birth date

```text
T1: month             12 类，近似均匀
T2: day               28 类
T3: comma             固定 ","
T4/T5: year           有些年份为一个 token，有些拆成两个
```

T1 最常见月份是 ` June`，占 8.53%；其余月份也约为 8.2%–8.5%。月份 first token 基本均匀，
所以 12 类 birth-date first Probe 是清晰的单 token 任务。

### Birth city

```text
T1: city root         21 类
T2: 数字后缀或 York   11 类
T3: comma 或州缩写    11 类
T4: 州缩写            10 类
```

200 个 whole 类别被压缩到 21 个 first-token 类别。除 `New York, NY` 的特殊组外，大部分
first token 对应 10 个完整城市，因此只知道 T1 时无法区分数字后缀和州。

### University

```text
T1: city/root         20 类
T2: 数字后缀或 University
T3: University        有数字后缀时出现
```

每个 T1 **精确对应 15 个 whole 类别**。因此 20 类 first-token 准确率接近 100% 时，并不
代表 300 类 university whole 已确定；还需要识别 T2 的 15 种后缀。

### Major

```text
T1: major root        20 类
T2–T4: 数字后缀、复合词剩余部分
```

每个 T1 精确对应 5 个 whole 类别。15.18% 的 profile major 是单 token，63.94% 是两个
token，所以 major 比 university 更容易从 first-token 表示延伸到 whole，这与其较高的
Q-whole 准确率一致。

### Company

```text
T1: surname/root      20 类
T2: 数字后缀或被 BPE 拆开的词根
T3/T4: Group
```

每个 first token 对应 13–14 个 whole company。需要注意 GPT-2 BPE 不保证 T1 是完整姓氏，
例如 `Ford...` 的 first token 可能是 ` For`，`Stone...` 可能从 ` St` 开始。因此不能把
first token 直接当作人类语义词。

### Company city

Token 结构与 birth city 相同，但 profile 频率不同：

- ` New`：13.67%，主要来自 36/263 个 company 都映射到 New York；
- ` Cambridge`：7.54%；
- ` Dallas`：6.40%；
- ` Raleigh`：6.19%。

所以 company-city first/whole 的先验明显不均匀，不能简单使用 \(1/21\) 或 \(1/200\)
作为准确率基线。

## First token 对 whole 类别的碰撞

| 属性 | Whole 类别 / first token：最小 | 平均 | 中位数 | 最大 |
|---|---:|---:|---:|---:|
| Birth date | 4,265 | 4,321.5 | 4,325.5 | 4,357 |
| Birth city | 1 | 9.52 | 10 | 10 |
| University | 15 | 15.00 | 15 | 15 |
| Major | 5 | 5.00 | 5 | 5 |
| Company | 13 | 13.15 | 13 | 14 |
| Company city | 1 | 9.52 | 10 | 10 |

生日 whole 类别没有训练 Probe，这里只展示完整日期如何共享 month first token。

该表直接解释了为什么不能把 `attr::t1` 视作完整事实：

- University：知道 T1 后仍有 15 个候选；
- Company：仍有 13–14 个候选；
- Birth/company city：通常仍有 10 个候选；
- Major：只剩 5 个候选，因此 whole 相对容易。

Company/company-city whole 最终明显优于 university，不是因为 first-token 碰撞更少，而是
额外利用了两个字段之间的确定性关联。

## 前导空格的实际影响

六类属性中，**100% 的唯一值**在：

```python
encode(value)
```

与：

```python
encode(" " + value)
```

之间产生了不同 token 序列。First-token 类别数量在本数据上碰巧保持相同，但 token ID
全部进入带空格版本的词表项，平均 token 长度也发生变化：

| 属性 | 裸字符串平均长度 | 带空格平均长度（唯一值） |
|---|---:|---:|
| Birth date | 4.359 | 4.359 |
| Birth city | 4.500 | 3.905 |
| University | 3.533 | 2.933 |
| Major | 2.850 | 2.100 |
| Company | 3.529 | 3.030 |
| Company city | 4.500 | 3.905 |

因此后续建立 `attr::tokenj` Probe 时，必须统一使用带前导空格的完整属性编码。否则 token ID、
序列长度和 token 位置都会与 biography 中真实出现的属性不一致。

## 机器结果

- [Summary JSON](../../../results/probes/synbios_moe/data/attribute_token_distribution/summary.json)
- [Attribute summary](../../../results/probes/synbios_moe/data/attribute_token_distribution/attribute_summary.csv)
- [Length distribution](../../../results/probes/synbios_moe/data/attribute_token_distribution/length_distribution.csv)
- [Full position-token distribution](../../../results/probes/synbios_moe/data/attribute_token_distribution/position_token_distribution.csv)

重建命令：

```bash
python -m experiments.synbios_moe.pretraining.token_statistics \
  --num-people 100000 \
  --seed 1337 \
  --output results/probes/synbios_moe/data/attribute_token_distribution
```
