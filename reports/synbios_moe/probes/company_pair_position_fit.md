# Company / Company-city 的 P-whole 位置增长拟合

## 结论

`multi5_permute` 中 company 与 company-city 的 P-whole 准确率并不是因为位置本身靠后而
增长。两条曲线几乎精确服从“当前位置前是否已经出现 `company` 或 `company_city` 任一
字段”的组合概率。这说明 frozen backbone 中已经形成可双向利用的工作属性关联：

- 看见 company 后，company-city 几乎可以无损恢复；
- 看见 company-city 后，company 也明显更容易恢复；
- 反向恢复的饱和值较低，与多个 company 共用同一 city 的数据歧义一致。

![Company pair position fit](../../../results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/company_pair_position_fit/company_pair_position_fit.png)

## 数据与双字段模型

输入是正式 `multi5_permute`、person-held-out validation 的 P-whole P0–P5 准确率，共
250,590 篇 validation biographies。六个属性在每篇 biography 中随机排列。

Company 和 company-city 任一出现在 Pj 之前的概率是：

\[
x_{\mathrm{pair}}(j)
=1-\frac{\binom{4}{j}}{\binom{6}{j}}
=[0,\;1/3,\;3/5,\;4/5,\;14/15,\;1].
\]

拟合模型为：

\[
A_j=\beta_0+\beta_1x_{\mathrm{pair}}(j),
\]

其中 \(\beta_0\) 是尚未见过相关字段时的基础准确率，\(\beta_0+\beta_1\) 是关联信息完全
可用时的饱和准确率。使用 RMSE、\(R^2\) 和留一预测 RMSE 衡量组合概率曲线与实际位置
曲线的一致程度。

## 拟合结果

### Company

双字段模型：

\[
A_j=45.055\%+49.538\%\,x_{\mathrm{pair}}(j).
\]

| P 位置 | 实际 | 双字段模型 |
|---|---:|---:|
| P0 | 45.13% | 45.06% |
| P1 | 61.79% | 61.57% |
| P2 | 74.42% | 74.78% |
| P3 | 84.29% | 84.69% |
| P4 | 91.25% | 91.29% |
| P5 | 95.09% | 94.59% |

拟合指标：RMSE **0.311pp**，\(R^2=\) **0.99968**，留一预测 RMSE **0.447pp**。

### Company city

双字段模型：

\[
A_j=48.442\%+51.275\%\,x_{\mathrm{pair}}(j).
\]

| P 位置 | 实际 | 双字段模型 |
|---|---:|---:|
| P0 | 48.56% | 48.44% |
| P1 | 65.44% | 65.53% |
| P2 | 79.11% | 79.21% |
| P3 | 89.40% | 89.46% |
| P4 | 96.29% | 96.30% |
| P5 | 99.86% | 99.72% |

拟合指标：RMSE **0.097pp**，\(R^2=\) **0.99997**，留一预测 RMSE **0.187pp**。

双字段模型以 0.1–0.3 个百分点的误差重建了全部六个位置，说明位置增长曲线与“前面已经
出现两个关联字段中的任意一个”高度一致。

## 为什么两个方向不完全对称

数据生成器通过同一个 `company_index` 同时设置 company 和 company-city，因此
`company -> company_city` 完全确定。263 个 company 映射到 200 个 city，其中 36 个
company 共享 `New York, NY`，少量 city 还对应两个 company。因此反向
`company_city -> company` 很强但不完全可逆。

拟合饱和值与该结构一致：

| 预测目标 | 双字段模型饱和值 |
|---|---:|
| Company | 94.59% |
| Company city | 99.72% |

## 证据边界

该拟合强烈支持 Probe 正在利用两个关联字段中的任意一个。它仍是位置级 aggregate 分析，
不单独构成逐样本因果证明。最终确认应在现有
formal checkpoint 上按随机字段顺序输出四组条件准确率：`none`、`company only`、
`company_city only`、`both`。该验证不需要重新训练 backbone 或 Probe。

## 可复现产物

- [Machine summary](../../../results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/company_pair_position_fit/summary.json)
- [Fit metrics](../../../results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/company_pair_position_fit/fit_metrics.csv)
- [PNG figure](../../../results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/company_pair_position_fit/company_pair_position_fit.png)
- [Vector PDF](../../../results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/company_pair_position_fit/company_pair_position_fit.pdf)

重建命令：

```bash
python -m experiments.synbios_moe.mechanisms.company_relation \
  --input results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/formal_probe_metrics.csv \
  --output results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/company_pair_position_fit
```
