# `_moe_router_scatter_kernel` 可视化说明

文件：`fused_moe_kernels.py`

这个 kernel 不搬运 token 数据，只为每条 `(token, top-k)` 路由计算新的排序位置，并生成后续 grouped GEMM 使用的索引和 tile 元数据。

## 示例输入

假设：

```text
T = 4                  # token 数
K = 2                  # 每个 token 的专家数
E = 3                  # expert 数
TOKENS_PER_BLOCK = 2
BLOCK_M_TOKEN = 2
```

路由结果：

```text
token 0: [2, 0]
token 1: [1, 2]
token 2: [0, 1]
token 3: [2, 1]
```

展平后的原始 route：

| entry | token | k | expert |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 2 |
| 1 | 0 | 1 | 0 |
| 2 | 1 | 0 | 1 |
| 3 | 1 | 1 | 2 |
| 4 | 2 | 0 | 0 |
| 5 | 2 | 1 | 1 |
| 6 | 3 | 0 | 2 |
| 7 | 3 | 1 | 1 |

```text
entry = token * K + k
```

## pid 内部排序

`pid 0` 处理 token 0、1：

```text
entry:  0  1  2  3
expert: 2  0  1  2
```

排序后：

```text
expert:       0  1  2  2
local_offset: 1  2  0  3
```

`pid 1` 处理 token 2、3：

```text
entry:  4  5  6  7
expert: 0  1  2  1
```

排序后：

```text
expert:       0  1  1  2
local_offset: 0  1  3  2
```

代码把 expert 和局部 offset 打包：

```python
kv_pairs = (expert << 16) | local_offset
kv_pairs = tl.sort(kv_pairs, 0)
```

高 16 位是 expert，低 16 位是局部 offset，因此排序首先按 expert 分组。

## `within_expert`

`within_expert_rank` 表示当前 route 在当前 pid 的 expert 分组中的排名：

```text
pid 0:
sorted expert: [0, 1, 2, 2]
rank:          [0, 0, 0, 1]

pid 1:
sorted expert: [0, 1, 1, 2]
rank:          [0, 0, 1, 0]
```

每个 expert 的总路由数为：

```text
expert 0: 2
expert 1: 3
expert 2: 3
```

因此 expert 在全局 sorted 数组中的起点是：

```text
expert_start = [0, 2, 5, 8]
```

`partial_sum[expert, pid]` 表示当前 pid 之前，同一个 expert 已经出现了多少条 route：

```text
             pid 0  pid 1
expert 0        0      1
expert 1        0      1
expert 2        0      2
```

于是：

```python
within_expert = partial_sum[expert, pid] + within_expert_rank
s_reverse = expert_start[expert] + within_expert
```

`s_reverse` 就是最终的全局 `sorted_pos`。

## 两个 pid 的计算结果

```text
pid 0:
expert:          [0, 1, 2, 2]
within_rank:     [0, 0, 0, 1]
previous_count:  [0, 0, 0, 0]
within_expert:   [0, 0, 0, 1]
expert_start:    [0, 2, 5, 5]
s_reverse:       [0, 2, 5, 6]

pid 1:
expert:          [0, 1, 1, 2]
within_rank:     [0, 0, 1, 0]
previous_count:  [1, 1, 1, 2]
within_expert:   [1, 1, 2, 2]
expert_start:    [0, 2, 2, 5]
s_reverse:       [1, 3, 4, 7]
```

## 全局 sorted 结果

```text
sorted_pos:  0  1 | 2  3  4 | 5  6  7
expert:      0  0 | 1  1  1 | 2  2  2
entry:       1  4 | 2  5  7 | 0  3  6
token:       0  2 | 1  2  3 | 0  1  3
```

输出索引为：

```text
s_scatter_idx         = [1, 4, 2, 5, 7, 0, 3, 6]
s_reverse_scatter_idx = [5, 0, 2, 6, 1, 3, 7, 4]
x_gather_idx           = [0, 2, 1, 2, 3, 0, 1, 3]
```

三个数组的含义：

```text
s_scatter_idx[sorted_pos]         = 原始 entry
s_reverse_scatter_idx[entry]      = sorted_pos
x_gather_idx[sorted_pos]          = 原始 token
```

例如：

```text
entry 0 -> sorted_pos 5
sorted_pos 5 -> token 0
```

所以：

```python
s_reverse_scatter_idx[0] = 5
x_gather_idx[5] = 0
```

## GEMM tile 元数据

`BLOCK_M_TOKEN = 2`，所以每个 expert 每两条 route 生成一个 tile：

```text
expert 0: rows [0,1]       -> 起点 row=0
expert 1: rows [2,3,4]     -> 起点 row=2、4
expert 2: rows [5,6,7]     -> 起点 row=5、7
```

最终：

```text
tile_row_start = [0, 2, 4, 5, 7]
tile_expert    = [0, 1, 1, 2, 2]
```

对应：

```text
tile 0: row=0, expert=0，处理 rows [0,1]
tile 1: row=2, expert=1，处理 rows [2,3]
tile 2: row=4, expert=1，处理 row  [4]
tile 3: row=5, expert=2，处理 rows [5,6]
tile 4: row=7, expert=2，处理 row  [7]
```

判断 tile 起点的代码是：

```python
is_tile_start = (within_expert % BLOCK_M_TOKEN) == 0
t_within = within_expert // BLOCK_M_TOKEN
flat_tile_idx = tile_base + t_within
```

注意：tile 边界由 `within_expert` 决定，而不是由 `pid` 决定。因此一个 expert 的下一个 tile 起点落在另一个 pid 中是正常的。

## 一句话总结

```text
局部按 expert 排序
-> 计算 expert 内排名
-> 加上之前 pid 的数量
-> 得到全局 sorted_pos
-> 每个 expert 每 BLOCK_M_TOKEN 条生成一个 GEMM tile
```
