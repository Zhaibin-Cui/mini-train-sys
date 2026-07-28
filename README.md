<div align="center">

<img src="assets/readme-banner.svg" alt="MiniTrainSys" width="78%" />

<p>
  <img src="https://img.shields.io/badge/Models-Dense%20%26%20MoE-2563eb?style=flat-square" alt="Dense and MoE models" />
  <img src="https://img.shields.io/badge/Kernels-PyTorch%20%C2%B7%20Triton%20%C2%B7%20CUDA-f97316?style=flat-square" alt="PyTorch, Triton, and CUDA kernels" />
  <img src="https://img.shields.io/badge/Distributed-DDP%20%C2%B7%20FSDP-7c3aed?style=flat-square" alt="DDP and FSDP training" />
  <img src="https://img.shields.io/badge/Research-Model%20Interpretability-0891b2?style=flat-square" alt="Model interpretability research" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-16a34a?style=flat-square" alt="MIT License" /></a>
</p>

</div>

---

<a id="overview"></a>

## 🧭 Overview

MiniTrainSys is a **from-source LLM training system** centered on a formal 12-layer,
293.49M-parameter sparse MoE workload. The repository implements the path from raw documents to
token shards, Transformer and MoE blocks, PyTorch/Triton/CUDA operator dispatch, single- and
multi-GPU training, distributed checkpoints, benchmarks, and post-training experiments.

The project is split at real system boundaries: data preparation, model structure, operator
backends, distributed execution, and training state. The same model and run configuration can
switch between PyTorch references, Triton kernels, and native CUDA attention, so backend comparisons
do not require a second model implementation.

The systems work is measured end to end. Custom forward-and-backward kernels reach
**1.22–6.51×** the corresponding PyTorch operator speed; the CUDA backend reaches **2.201×**
PyTorch token throughput at the matched memory budget; and four-GPU FSDP reaches **3.69×** weak
scaling. The repository keeps the parity checks, capacity sweeps, raw measurements, and generated
figures behind those numbers.

The same stack supports a controlled MoE knowledge-retrieval study. Two matched, approximately
4B-token pretraining runs vary only biography presentation, then use source-text recall, frozen P/Q
probes, route difference-in-differences, and an oracle first-token intervention to trace how a
memorized fact becomes readable.

### 🗺️ Scope

| Area | What is implemented here |
|---|---|
| **Training system** | Configurable Dense and top-2 MoE decoders, token-shard data loading, BF16 training, AdamW, logging, and resumable checkpoints. |
| **GPU kernels** | PyTorch references plus Triton RMSNorm, RoPE, SwiGLU, cross entropy, FlashAttention, router, and fused MoE paths; native CUDA FlashAttention. |
| **Distributed runtime** | Single GPU, DDP, and FSDP with rank-safe metrics, DCP optimizer/model state, consolidated evaluation checkpoints, and recovery anchors. |
| **Research workflow** | Matched `single` and `multi5_permute` pretraining, cloze recall, frozen P/Q probes, relation controls, route DiD, and first-token intervention. |

## 📖 Contents

| Read the results | Understand the system | Run the project |
|---|---|---|
| [🔥 Performance](#benchmarks) | [🧩 System Design](#system-design) | [⚡ Kernel Engineering](#kernel-engineering) |
| [🚄 Training Performance](#training-performance) | [🧠 Model Interpretability](#interpretability) | [🚀 Quick Start](#quick-start) |
| [📊 Reproducibility](#reproducibility) | [🗂️ Project Structure](#project-structure) | [📦 Data and Configuration](#data-and-configuration) |
| [🛠️ Training Workflows](#training-workflows) | [🛡️ Reliability](#reliability) | [📚 Documentation](#documentation) |
<a id="benchmarks"></a>

## 🔥 Performance at a glance

<img src="assets/readme-highlights.svg" alt="MiniTrainSys portfolio highlights" width="100%" />

| Measurement | Result |
|---|---:|
| Kernel suite, full forward + backward | **1.22–6.51×** over the corresponding PyTorch operator |
| Fused Linear Cross Entropy, peak allocated memory | **94.0% lower** than the explicit-logits PyTorch path |
| 293M MoE, local batch 24 on every backend | CUDA: **1.456×** PyTorch token throughput |
| 293M MoE, matched memory budget | CUDA: local batch **24 → 96** and **2.201×** PyTorch token throughput |
| FSDP weak scaling, 1 → 4 GPUs | **3.69×** throughput; **92.24%** parallel efficiency |
| `multi5_permute` Q-first macro, name-only context | **98.79%**, compared with **12.83%** for `single` |
| `multi5_permute` P-whole, oracle first token appended | **45.28% → 96.38%** after training a fresh probe |

All server results were produced on the same 4× RTX 4090 24 GB machine with PyTorch 2.5.1+cu118, CUDA 11.8, and Triton 3.1.

<a id="system-design"></a>

## 🧩 System design

Source documents are converted into indexed token shards and read by the selected distributed
strategy. The model calls the configured operator backend; training writes metrics and checkpoints;
experiment code reads those checkpoints together with their data manifests.

<div align="center">
  <img src="assets/system-design.svg" alt="MiniTrainSys system design" width="100%" />
</div>

`OpsBackend` is the main boundary: model code calls the same interface whether an operator is
implemented in PyTorch, Triton, or CUDA.

| Layer | Main code | Responsibility |
|---|---|---|
| Data | `minitrain/data` | Tokenization, shards, datasets, and loading |
| Model and kernels | `minitrain/model`, `minitrain/kernels` | Dense/MoE Transformer and operator backends |
| Training | `minitrain/distributed`, `minitrain/train` | Single/DDP/FSDP, optimization, and checkpoints |
| Analysis | `experiments/synbios_moe` | Recall, P/Q probes, and route analysis |

<a id="kernel-engineering"></a>

## ⚡ Kernel engineering

Each custom kernel is compared with the equivalent PyTorch path at shapes taken from the 293.49M
MoE workload. Every reported number covers **forward + backward**.

- **Speedup** is elapsed-time improvement over PyTorch at the same shape; `2.00×` means half the time.
- **Peak allocation reduction** is the reduction in PyTorch-reported peak allocated GPU memory.

The representative workload uses BF16, hidden size 768, 8 experts, top-2 routing, vocabulary size
50,257, and up to 57,344 tokens per rank. Each case runs in a fresh CUDA process with warmup,
synchronization, repeated trials, and P50/P95 latency collection. Results are published only after
forward/backward parity, dtype and reduction, boundary and strided-shape, and fallback checks pass.

![Kernel benchmark overview](results/benchmarks/operator_benchmark/resume_summary/kernel_benchmark_overview.png)

| Operator | Implementation | Primary benefit |
|---|---|---|
| RMSNorm | Normalize and scale in one row-wise kernel | Fewer launches and memory round-trips |
| RoPE | Tile token/head rows; reuse Q/K sinusoids | Reduces small-kernel launch and bandwidth cost |
| SwiGLU | Keep activation and gate product fused | Avoids a temporary activation tensor |
| Cross Entropy | Stream vocabulary blocks with online log-sum-exp | Avoids the full probability matrix |
| Fused Linear CE | Consume LM-head logits immediately | Avoids materializing `[tokens, vocab]` logits |
| FlashAttention† | Specialized Triton attention path | Reduces attention I/O and saved tensors |
| Native CUDA Attention†‡ | Compiled FlashAttention shape buckets | Low-level, shape-specialized attention |
| Router postprocess | Fuse softmax, top-k, weights, losses, and stats | Avoids repeated scans of router logits |
| Fused MoE | Fuse dispatch, expert execution, and weighted gather | Reduces irregular MoE dispatch overhead |

<sub>† Attention rows are measured against PyTorch's <code>aten::_scaled_dot_product_flash_attention</code> backend.</sub>

<sub>‡ Native CUDA Attention adapts the upstream FlashAttention 2.8.4 CUDA kernels; MiniTrain owns the tensor interface and build specialization.</sub>

<details open>
<summary><strong>What the main kernels do</strong></summary>

<details>
<summary><strong>RMSNorm</strong> — Speed: <strong>6.51×</strong> · Peak allocation: <strong>−78.7%</strong></summary>

For each hidden row $x \in \mathbb{R}^{H}$, the forward pass is

$`r = \frac{1}{\sqrt{\mathrm{mean}(x^2) + \varepsilon}}, \qquad y = x \odot r \odot \gamma.`$

One Triton program owns one row (or packs short rows), reduces `sum(x²)` in FP32, and writes `y`
once. It saves `r` per row. Backward reads `x`, `γ`, `r`, and `dy` to form `dx` and `dγ`; it does
not re-run the reduction. Parallelism is over rows, with the hidden dimension reduced inside the
program.

</details>

<details>
<summary><strong>RoPE</strong> — Speed: <strong>2.57×</strong> · Peak allocation: <strong>−14.3%</strong></summary>

For every rotary pair $(a,b)$ at position $t$, forward applies

$`a' = a\cos(t) - b\sin(t), \qquad b' = a\sin(t) + b\cos(t).`$

One program owns one `(batch, token)` position and lays out all Q/K heads and rotary pairs as a
tile. `sin(t)` and `cos(t)` are loaded once for that position and shared across heads; there is no
host-side loop over heads. Backward applies the inverse rotation to `dq` and `dk` using the same
tile. The gain is mostly fewer launches and less temporary traffic.

</details>

<details>
<summary><strong>SwiGLU</strong> — Speed: <strong>1.42×</strong> · Peak allocation: <strong>−20.0%</strong></summary>

With gate projection $g$ and up projection $u$, forward is

$`y = \mathrm{SiLU}(g) \odot u, \qquad \mathrm{SiLU}(g) = g\,\sigma(g).`$

Each program covers a tile of the activation. It loads `g` and `u`, evaluates the gate and product
in registers, then stores only `y`. Backward uses $dy$ to form
$du = dy \odot \mathrm{SiLU}(g)$ and
$dg = dy \odot u \odot \mathrm{SiLU}'(g)$ in the same tiled layout. **20.0% lower peak allocation.**

</details>

<details>
<summary><strong>Cross Entropy</strong> — Speed: <strong>4.54×</strong> · Peak allocation: <strong>−83.3%</strong></summary>

For a target class $c$ and logits $z$, forward evaluates

$`L = -z_c + \log\!\sum_v \exp(z_v), \qquad \frac{\partial L}{\partial z_v} = \frac{\mathrm{softmax}(z)_v - \mathbf{1}[v=c]}{N_{\mathrm{valid}}}.`$

The vocabulary is read in blocks. Online log-sum-exp retains only a running maximum and running
sum, rather than a full probability vector. A program owns one token row while lanes cover a
vocabulary block. **83.3% lower peak allocation** at the largest common shape.

</details>

<details>
<summary><strong>Fused Linear Cross Entropy</strong> — Speed: <strong>1.39×</strong> · Peak allocation: <strong>−94.0%</strong></summary>

The LM head and CE are fused across token chunks. For a chunk $C$:

$`Z_C = X_C W^\top, \qquad \frac{\partial L}{\partial X_C} = \frac{\partial L}{\partial Z_C}W, \qquad \frac{\partial L}{\partial W} \mathrel{+}= \left(\frac{\partial L}{\partial Z_C}\right)^\top X_C.`$

**Forward.** Choose the largest power-of-two number of tokens whose temporary
`[chunk_tokens, vocab]` logits buffer fits the 64 MiB workspace budget. For one chunk, it adds the
online CE loss to a running scalar, overwrites `Z_C` with `∂L/∂Z_C`, computes the gradients above,
then reuses that workspace for the next chunk—there is never an `[all_tokens, vocab]` logits matrix.

**Backward.** The custom autograd forward has already materialized the chunk gradients, so backward
normally returns saved `dX` and `dW`, only scaling them when the upstream loss gradient is not one.
At 57,344 tokens, the fused path adds 356 MiB while the
explicit-logits baseline does not fit (**94.0% lower peak allocation**).

</details>

<details>
<summary><strong>FlashAttention</strong> — Triton (Speed: <strong>1.22×</strong>; Peak allocation: <strong>−44.1%</strong>) · CUDA (Speed: <strong>1.15×</strong>; Peak allocation: <strong>−22.1%</strong>)</summary>

Forward computes $O = \mathrm{softmax}(QK^\top / \sqrt{d})V$ without storing the
$S \times S$ score matrix. A program owns
one `(batch, head, query-tile)` and streams successive K/V tiles. It carries online softmax
statistics `(m, l)` for the query tile and updates `O` as each K/V tile arrives; heads are
independent programs.

Backward first runs a small preprocess kernel for each query row:

$`D_i = \sum_d O_{id}\,dO_{id}.`$

`D` is one FP32 scalar per row (named `Delta` in the code). The two tiled backward paths then
recompute $P = \exp(QK^\top / \sqrt{d} - \mathrm{LSE})$ from Q/K and the saved row LSE, rather
than reading a saved probability matrix. For each tile:

$`dP = dO\,V^\top, \qquad dS = P \odot (dP - D),`$

$`dV = P^\top dO, \qquad dK = \frac{dS^\top Q}{\sqrt{d}}, \qquad dQ = \frac{dS K}{\sqrt{d}}.`$

The K/V-tile kernel scans query tiles and accumulates $dK$ and $dV$; the Q-tile kernel scans K/V
tiles and accumulates $dQ$.
Thus `dQ` is written by its owning query tile, while `dK` and `dV` are written by their owning K/V
tile—no atomic accumulation and no $S \times S$ score/probability tensor. Tile sizes are selected per
shape. The baseline is PyTorch Flash-SDPA (`aten::_scaled_dot_product_flash_attention`), not naive
$QK^\top \rightarrow \mathrm{softmax} \rightarrow V$: Triton reduces allocation by **44.1%**; the native CUDA path reduces it by
**22.1%**. See [raw dispatch evidence](results/benchmarks/operator_benchmark/resume_summary/torch_attention_backend.json).

</details>

<details>
<summary><strong>Router postprocess</strong> — Speed: <strong>2.43×</strong> · Peak allocation: <strong>−65.2%</strong></summary>

For router logits $r_t$, the forward pass computes $p_t = \mathrm{softmax}(r_t)$, selects
$\mathrm{topk}(p_t)$, and optionally renormalizes the selected weights:

$`w_{tj} = \frac{p_{t,e_j}}{\sum_{j' \in \mathrm{topk}(p_t)} p_{t,e_{j'}}}.`$

The same pass accumulates mean expert probability, z-loss, and entropy statistics. A program covers
a small block of token rows and all experts for those rows, writing only `k` weights and indices per
token rather than a retained probability matrix. Backward receives gradients from selected weights,
load-balance statistics, and z-loss, then writes `dr` through the softmax/top-k normalization path;
expert indices are non-differentiable. **65.2% lower peak allocation.**

</details>

<details>
<summary><strong>Fused MoE</strong> — Speed: <strong>1.46×</strong> · Peak allocation: <strong>−5.3%</strong></summary>

For selected expert $e_j$ and routing weight $w_{tj}$, forward is

$`h_{tj} = \mathrm{SiLU}(x_t W_{\mathrm{gate},e_j}^\top) \odot (x_t W_{\mathrm{up},e_j}^\top), \qquad y_t = \sum_j w_{tj}\,h_{tj}W_{\mathrm{down},e_j}^\top.`$

It first builds expert counts, prefix sums, and scatter indices, then groups the `T × k` routed
copies into expert-contiguous tiles. Each expert tile runs fused gate/up projection plus SwiGLU,
then down projection; a final kernel gathers weighted outputs back to token order.

Backward expands `dy` to routed copies, computes `dWdown`, `dWgate/up`, `dx`, and `dw` in those same
expert tiles, then gathers per-route `dx` to each original token. Routing indices have no gradient;
the weight gradient returns to the router. This removes standalone dispatch, expert, and gather
stages.

</details>

</details>

<a id="training-performance"></a>

## 🚄 Training performance

The backend benchmark measures the complete training step: data loading, forward, backward,
optimizer update, logging, and synchronization.

![Equal-memory backend comparison](results/benchmarks/synbios_backend_capacity/20260725T113500Z/presentation/capacity_backend_comparison.png)

<sub>Backend routing: <code>cuda</code> inherits <code>triton</code> and overrides only FlashAttention; all other operators use the same Triton paths. Unsupported native CUDA attention dispatches <code>CUDA → Triton → PyTorch</code>.</sub>

| Condition | PyTorch | Triton | CUDA |
|---|---:|---:|---:|
| Equal local batch | 24 | 24 | 24 |
| Relative throughput | 1.000× | 1.393× | **1.456×** |
| Largest memory-safe local batch | 28 | 96 | **112** |
| Equal-budget relative throughput | 1.000× | 1.994× | **2.201×** |

Capacity sweeps select the highest measured throughput below the accepted memory ceiling. This may
be smaller than the largest allocation that completes without OOM.

### 📈 FSDP scaling

![FSDP weak scaling](results/benchmarks/synbios_moe_fsdp4/weak_b64/presentation/weak_overview.png)

FSDP reaches **3.69× weak scaling from one to four GPUs**, or **92.24% parallel efficiency**.
Residual loss comes from FSDP all-gather/reduce-scatter and synchronization over a PCIe `SYS`
topology; this machine has no NVLink.

<a id="interpretability"></a>

## 🧠 Factual retrieval in a sparse MoE

We ask:

> If two MoE models memorize the same facts, can the presentation of those facts alone change where
> they become readable?

To answer this question, we adapt the controlled bioS experiments of
[Allen-Zhu and Li](https://arxiv.org/abs/2309.14316) to a sparse model.

Both conditions contain the same 100,000 people and the same six facts per person. They use the same
12-layer, hidden-768, 8-expert top-2 MoE, with 293.49M total parameters and 123.62M
token-active parameters. The experimental change is the number, wording, and sentence order of the
biographies presented for each person.

| Dataset | People | Biographies / person | Documents | Epochs | Optimizer steps | Scheduled tokens |
|---|---:|---:|---:|---:|---:|---:|
| `single` | 100,000 | 1, fixed field order | 100,000 | 540 | 17,280 | 3,963,617,280 |
| `multi5_permute` | 100,000 | 5, independently rewritten and shuffled | 500,000 | 108 | 17,388 | 3,988,389,888 |

Both models start from random initialization and use the same tokenizer, next-token objective,
optimizer, global batch 448, and seed 1337. `multi5_permute` contains five times as many documents,
so it runs for one fifth as many epochs. The scheduled-token budgets differ by 0.62%. The two
`profiles.jsonl` files are byte-identical.

### 🧪 Result 0: both models memorize the source biographies

We first rule out failed memorization. The cloze evaluator removes all six fact spans from each
training biography, regenerates them from left to right, and inserts each prediction before
generating the next field. It fits no new parameters.

| Recall metric | `single` | `multi5_permute` |
|---|---:|---:|
| Strict field accuracy | **100.000000%** | **99.991533%** |
| All-six-fields biography accuracy | **100.0000%** | **99.9626%** |
| Evaluation coverage | 100k biographies / 600k fields | 500k biographies / 3M fields |

The augmented model misses 254 of 3,000,000 fields; the fixed-order model misses none. Failed
memorization cannot explain the differences below. These numbers are source-corpus recall, not
generalization to unseen people.

### 🔎 Reading facts with frozen P/Q probes

We next ask where each fact can be read. P and Q follow the definitions introduced by Allen-Zhu and
Li. P reads a final-layer state inside a biography, immediately before an attribute value. Q sees
only `[EOS, full_name, EOS]` and reads the final EOS state.

| Probe | Input and read position | `first` target | `whole` target |
|---|---|---|---|
| P | Biography prefix; state immediately before the attribute span | First BPE token | Complete attribute class |
| Q | `[EOS, full_name, EOS]`; final EOS state | First BPE token | Complete attribute class |

The bracketed tokens below are the six P read positions in one generated `single` biography:

```text
Jonah15 Blair13 Carter36 entered the world [P0: on] October 24, 2046.
He was born [P1: in] Tacoma9, NC.
He received mentorship [P2: at] Chicago8 University.
He specialized [P3: in] Political Science2.
He was employed [P4: by] Brooks9 Group.
He was professionally based [P5: in] Albany3, GA.

P-company: biography prefix ending at P4 → predict " Brooks9 Group"
Q-company: [EOS, Jonah15 Blair13 Carter36, EOS] → predict " Brooks9 Group"
```

The backbone is frozen in every probe experiment. Given its embedding table
$E \in \mathbb{R}^{V \times H}$, the probe fits a rank-`r` input-embedding update:

$$
E' = E + AB, \qquad e'_t = e_t + A_tB.
$$

`B` is initialized to zero, so training starts from the original embeddings. The selected hidden
state is normalized and passed to a linear classifier:

$$
\mathrm{logits} = W_{\mathrm{cls}}\,\mathrm{Norm}(h_L[\mathrm{position}]) + b_{\mathrm{cls}}.
$$

Only `A`, `B`, the normalizer, and the classifier are trainable. Attention, MoE, backbone
normalization, and LM-head parameters remain fixed.

| Probe | Readout | Rank | First / whole budget |
|---|---|---:|---:|
| P | P0–P5 in the full biography | 2 | 4,000 / 12,000 steps; batch 128 |
| Q | Final EOS after the name | 16 | 4,000 / 12,000 steps; batch 768 |

Each head trains on 49,882 people and is evaluated on the other 50,118. This split tests whether a
probe fitted on one group of pretrained people transfers to another group; all 100,000 people were
seen by the backbone. Non-date `first` targets are the first GPT-2 token of
`" " + full_value`; `whole` treats the complete value as one class. Birth date uses the month as
its first-token target and has no whole-value task.

#### Result 1: augmentation moves first-token readout to the name

![Knowledge augmentation changes where facts become linearly readable](results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/figures/formal_study_overview.png)

| Probe endpoint | `single` | `multi5_permute` |
|---|---:|---:|
| P0-first, excluding the fixed first field | 6.76% | **98.63%** |
| Q-first macro | 12.83% | **98.79%** |

In `single`, first-token accuracy rises only when P reaches the target sentence. In
`multi5_permute`, every attribute is already 97.18–99.76% readable at P0, and the name-only Q probe
reaches 98.79%. Rewriting and sentence permutation move the first token from its fixed biography
position to the name state.

![P-first position heatmap](results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/figures/formal_p_first_heatmaps.png)

<div align="center">
  <img src="results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/figures/formal_q_probe_table.png" alt="Q-first results" width="64%" />
</div>

#### Result 2: augmentation does not make complete values equally readable

The same result does not hold for the complete value. The augmented MoE reaches 98.79% on Q-first
but only 33.15% on Q-whole. Allen-Zhu and Li report 92.58% Q-whole accuracy for their dense
`bioS multi5+permute` model.

| Five-attribute macro | `single` MoE | `multi5_permute` MoE | Dense `multi5+permute` |
|---|---:|---:|---:|
| P1-whole | 13.92% | **39.66%** | **93.56%** |
| Q-whole | 3.18% | **33.15%** | **92.58%** |

<sub>The dense P1-whole value is the Figure 13(c) rank-2 macro of 96.4, 76.0, 96.0, 99.7, and
99.7. The Q-whole value is the Figure 7 macro of 96.1, 72.6, 94.9, 99.6, and 99.7. Both are
reported by [Allen-Zhu and Li](https://arxiv.org/pdf/2309.14316).</sub>

The augmented Q-whole heads reach 74.51–98.18% on their training people but only 8.48–51.04% on
the person-held-out probe split. The heads can fit the task, but their whole-value boundaries do not
transfer across people as reliably as their first-token boundaries.

In the fixed-order `single` heatmap, warm-gold outlines mark the position immediately before each
attribute's own sentence.

![P-whole position heatmap](results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/figures/formal_p_whole_heatmaps.png)

<div align="center">
  <img src="results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/figures/formal_q_whole_probe_table.png" alt="Q-whole results" width="64%" />
</div>

### 🔬 Why does the first token transfer but the complete value does not?

Three questions remain. Do the company curves rise because related context has already appeared?
Do different continuations use different MoE routes? If they do, is the first token sufficient to
recover the value? We answer them in that order.

#### Control: the company–city curves follow relation exposure

The P-whole matrix contains a separate relational effect. Company and company-city improve as the
biography advances, while birth city, university, and major remain nearly flat. No new probe is
trained for this analysis; the completed held-out P-whole measurements are fitted directly.

The data generator contains 263 companies and 200 company cities. Each company maps to one city;
171 cities map back to one candidate company and 28 map to two. A company therefore determines its
city, while a city narrows the company to 1.315 candidates on average.

With six independently permuted sentences, the probability that either related field has appeared
before P position `j` is

$$
\Pr(\text{company or company city before } P_j)
= 1 - \frac{\binom{4}{j}}{\binom{6}{j}}.
$$

For P0 through P5, the values are 0, 1/3, 3/5, 4/5, 14/15, and 1.

| Target | Fitted baseline | Saturation | RMSE | R² |
|---|---:|---:|---:|---:|
| Company | 45.06% | 94.59% | 0.311 pp | **0.99968** |
| Company city | 48.44% | 99.72% | 0.097 pp | **0.99997** |

![Company and company-city position fit](results/formal_runs/synbios_moe/results/formal_probe_comparison_20260724/company_pair_position_fit/company_pair_position_fit.png)

The exposure model reconstructs all six positions within 0.1–0.3 percentage points. The two
readouts rise when either side of the relation is already present in the prefix, rather than merely
because P is later in the biography. This aggregate fit is consistent with use of the relation in
both directions, but it does not identify which direction is used for an individual example.

The fit accounts for the position-dependent company curves without a generic late-position
advantage. It does not account for the name-only first/whole gap.

#### Test 1: `t2` changes the MoE route after a shared `t1`

The routing analysis uses the frozen augmented backbone without a probe embedding update or
classifier. It selects 162,044 person-held-out cases for which Q-first is correct, Q-whole is wrong,
and the target value contains at least two tokens. The exact target tokens `t1` and `t2` are appended
after the name, and the top-2 selected experts are recorded at both positions in every layer.

Pairs share an attribute and `t1`. Pairs with the same `t2` are the control; pairs with different
`t2` form the contrast. For an attribute/layer group `g` with sampled pair set `P_g`, mean route
overlap is

$$
J_g(t) = \frac{1}{\lvert \mathcal{P}_g \rvert}
\sum_{(a,b) \in \mathcal{P}_g}
\frac{\lvert E_a(t) \cap E_b(t) \rvert}{\lvert E_a(t) \cup E_b(t) \rvert}.
$$

The difference-in-differences subtracts the route-overlap change in the same-`t2` control from the
change in the different-`t2` pairs:

$$
\mathrm{DiD} = [J_{\mathrm{different}}(t_1) - J_{\mathrm{different}}(t_2)] - [J_{\mathrm{same}}(t_1) - J_{\mathrm{same}}(t_2)].
$$

All 12 layer aggregates are positive. The largest DiD is **0.676** at layer 1, and the effect is
strongest in the first four layers.

![Attribute-by-layer route branching DiD](results/formal_runs/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/report/figures/route_attribute_layer_did_heatmap.png)

Among examples that share `t1`, different `t2` values lose more route overlap than the matched
same-`t2` control. Thus, in these error cases, expert selection depends on the continuation token.
This result still does not show that providing `t1` is enough to recover the value.

#### Final test: oracle `t1` restores whole-value readout

For each P0–P5 prefix, the intervention appends the exact first token from the cached target label
and reads the resulting token state. It does not run or reuse a P-first classifier. The backbone
remains frozen, and a fresh rank-2 P-whole probe is trained at the appended token with the same
person split, class mapping, and seed as the formal probes.

Each of the five whole-value heads trains for 4,000 steps with batch 128; validation uses batch
3,072. The no-intervention baseline is the original 12,000-step P-whole probe at the pre-attribute
position.

| Condition | Readout region | Original P-whole | P-whole with oracle `t1` |
|---|---|---:|---:|
| `single` | Five fixed source positions (diagonal) | 51.21% | **94.09%** |
| `multi5_permute` | P0, before any fact is exposed | 32.59% | **95.62%** |
| `multi5_permute` | All 30 position × attribute cells | 45.28% | **96.38%** |

In `single`, recovery is concentrated at the five positions where the corresponding fact is about
to appear. In `multi5_permute`, it extends across the full 30-cell matrix. The exact first token is
therefore sufficient for a much stronger whole-value readout, especially after augmentation.
Because this test changes both the input and the read coordinate, it does not measure an end-to-end
prediction chain.

#### `single`

![Single oracle-first-token P-whole](results/formal_runs/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/diagnostics/ground_truth_first_whole_p_pilot4000_20260726T033800Z/figures/ground_truth_first_p_overview.png)

#### `multi5_permute`

![Multi5 oracle-first-token P-whole](results/formal_runs/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/formal/diagnostics/ground_truth_first_whole_rank_matched_pilot4000_20260725T100100Z/figures/ground_truth_first_p_overview.png)

### 🧩 From the observations to the mechanism

The experiments now give the following sequence:

1. Near-perfect cloze recall rules out failed source memorization.
2. P-first and Q-first show that rewriting and field permutation make each attribute's first token
   readable from the name; fixed-order training leaves it tied to the source position.
3. Q-whole remains low even though its training accuracy is high, so the missing held-out readout is
   not explained by an unfitted head.
4. Route DiD shows that examples sharing `t1` select different experts when `t2` differs.
5. The final oracle intervention restores P-whole accuracy at the expected `single` positions and
   across the augmented matrix, including 95.62% at P0.

The two probe targets separate cleanly. A Q-first head can recover `t1` from the augmented name
state, while a separately trained Q-whole head cannot transfer the complete value nearly as well.
After exact `t1` is added to the tested P prefixes, a fresh P-whole head recovers almost all of the
value. Meanwhile, continuations that differ at `t2` select different top-2 routes.

```text
name-only state ── Q-first probe ──> t1
exact t1 → exact t2 ── route capture ──> token-dependent top-2 routes
biography prefix + oracle t1 ── fresh P-whole probe ──> complete value
```

**For this MoE and dataset, factual readout proceeds in stages. Augmentation exposes the first
attribute token at the name state. After that token is supplied to a biography prefix, the complete
value becomes linearly readable, and different continuations are accompanied by different expert
routes.** The experiments observe readout and routing. They do not place a fact inside one expert.

The comparison with the published dense model concerns linear readout, not a matched architectural
ablation. The dense reference reports high whole-value accuracy from the name state; the MoE tested
here does not. A matched dense backbone, additional seeds, and an intervention driven by predicted
rather than oracle `t1` would be required to attribute that difference specifically to sparsity or
routing.

### 🧬 What this suggests for pretraining

The data comparison changes only presentation: the people, facts, model, optimizer, and token budget
are matched. Nevertheless, fixed-order biographies and rewritten, permuted biographies produce
different access patterns. Repetition alone is sufficient for memorization. Variation makes the
first token readable before its original source position.

The effect stops at a meaningful boundary: Q-first reaches 98.79%, while Q-whole remains at 33.15%.
Augmentation changes how retrieval begins, but it does not make the complete value uniformly
readable from the name.

The broader pretraining hypothesis is: **information that must later be selected, continued, or
recombined independently should appear across varied contexts and orders during pretraining.** In
this experiment, that variation makes the first token of each attribute readable from the name
across people; later tokens still depend on the continuation. Natural corpora and larger models are
the next test of this hypothesis.

<a id="quick-start"></a>

## 🚀 Quick start

MiniTrainSys supports Python 3.10–3.12. Create an environment:

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install the CPU development stack:

```bash
python -m pip install -e ".[data,dev]"
```

Run the smallest complete training loop on CPU:

```bash
python scripts/train.py \
  --config configs/train_debug.yaml \
  --model-config configs/model_debug_dense.yaml \
  --device cpu
```

Then continue with [single-GPU, FSDP, and interpretability workflows](#training-workflows), or read
the [project walkthrough](docs/guides/project_walkthrough.md). Triton is only needed for the
Linux/x86_64 GPU workflow and is installed by the server setup path below.

<a id="reproducibility"></a>

## 📊 Reproducibility and evidence

Large mutable artifacts live on the data disk; Git stores compact, auditable evidence:

```text
artifacts/ → /data/mini-train-sys/artifacts/   datasets, checkpoints, logs, raw runs

results/                                       Git-safe evidence
├── benchmarks/                               raw cases, aggregates, figures
├── datasets/                                 manifests, lineage, checksums
├── formal_runs/                              metrics, events, summaries
├── logs/                                     benchmark, experiment, validation logs
├── notebooks/                                executed benchmark notebooks
├── tensorboard/index.csv                     TensorBoard ownership index
├── catalog/artifacts.json                    exported-file catalog
├── catalog/retention.json                    retained large-artifact catalog
└── MANIFEST.sha256                           snapshot integrity

reports/                                       long-form engineering and research reports
```

Raw biographies, token caches, model weights, optimizer/DCP shards, and large per-example route
records remain under `/data`. Git records their logical location, size, category, retention state,
and available hashes.

After every benchmark or validation cycle:

```bash
bash scripts/bash/export_test_results.sh
sha256sum --quiet -c results/MANIFEST.sha256
```

Cross-experiment headline metrics live in
[`results/BENCHMARK_SUMMARY.md`](results/BENCHMARK_SUMMARY.md). Exact commands and lifecycle records,
including failed and stopped runs, are retained under [`results/logs/`](results/logs/).

<a id="project-structure"></a>

## 🗂️ Project structure

```text
minitrain/
├── data/          document sources, token shards, mmap datasets, samplers
├── model/         Transformer, attention, Dense/MoE blocks
├── kernels/       PyTorch reference, Triton kernels, CUDA extension
├── distributed/   single, DDP, and FSDP strategy adapters
├── train/         steps, optimizer, LR, clipping, checkpoint state
└── runtime/       config, factories, devices, logging

experiments/       SynBioS data generation, cloze evaluation, probes, route analysis
configs/           composable model/data/strategy/hardware/run YAML
scripts/           training, benchmarking, export, and server entry points
tests/             executable benchmark notebooks, runners, and shared utilities
reports/           engineering and experiment reports
results/           versioned machine-readable evidence
```

<a id="data-and-configuration"></a>

## 📦 Data and configuration

### Data preparation and loading

The training stack accepts three data sources. `random` is useful for smoke tests before a corpus
exists; `tokens` reads a small legacy token file (`.pt`, `.pth`, `.npy`, or `.bin`); and
`token_shards` is the scalable path used for real corpora.

| Data source | Input | Typical use |
|---|---|---|
| `random` | Seeded synthetic token IDs | Debugging the model, kernels, and distributed loop |
| `tokens` | One token file in memory | Small experiments and compatibility data |
| `token_shards` | `manifest.json` plus binary shard files | Reusable, large-corpus training |

### Building a token-shard corpus

`scripts/prepare_data.py` separates tokenizer creation from corpus tokenization, so a tokenizer
can be trained once and reused across corpora. It reads `.txt`, `.text`, `.md`, `.jsonl`,
`.jsonlines`, and `.parquet`; JSONL and Parquet use `text` by default, or another field selected
with `--text-field`.

Choose either tokenizer route:

```bash
# Train a byte-level BPE tokenizer on your own text.
python scripts/prepare_data.py train-tokenizer raw_docs \
  --output artifacts/my_tokenizer --vocab-size 32768

# Or record a named, existing tiktoken encoding for reuse.
python scripts/prepare_data.py use-tiktoken \
  --encoding gpt2 --output artifacts/gpt2_tokenizer
```

Then tokenize any corpus with that recorded artifact:

```bash
python scripts/prepare_data.py tokenize raw_docs \
  --tokenizer artifacts/my_tokenizer \
  --output artifacts/my_corpus
```

Preparation normalizes and filters text by default, applies exact-deduplication, splits documents
deterministically into train/validation, and chunks very long documents at natural boundaries.
Cleaning, document-size, split, and shard-size limits are all CLI options. The resulting directory
has this layout:

```text
raw documents
  → recorded tokenizer artifact
  → bounded token shards
  → documents.idx + manifest.json
```

- **Token shards** hold the token stream in bounded files.
- **`documents.idx`** stores each document's token `(offset, length)`, so documents can be found
  without parsing the source again.
- **The manifest** records the tokenizer fingerprint, vocabulary/storage format, split and
  cleaning settings, shard paths, token counts, sizes, and hashes.

`python scripts/prepare_data.py inspect artifacts/my_corpus` prints that manifest. Put its
fingerprint in `data.tokenizer_fingerprint` to reject a tokenizer/corpus mismatch before training.

Training uses `data.source: token_shards` and opens the shard files with `mmap`; the full corpus is
not copied into RAM. Each worker keeps only a bounded LRU of open shard mappings. The loader
supports two packing modes:

| Packing mode | Behavior |
|---|---|
| `contiguous` | Read fixed-length blocks from the token stream; used for ordinary token-shard training. |
| `randomized_documents` | Reorder complete documents at each epoch, then pack them into fixed-length blocks; used by the SynBioS experiment. |

`randomized_documents` changes only the logical order: it uses `documents.idx` to locate document
spans and does not rewrite the shard files. In distributed runs, each rank receives a disjoint,
equal-length block sequence. `set_epoch(epoch)` changes the order deterministically, so reruns with
the same seed match. When CUDA is used, DataLoader batches use pinned CPU memory and non-blocking
H2D copies. `num_workers: null` derives a per-rank worker count from `worker_budget` and
`max_workers_per_rank`.

Start from [`configs/data_token_shards_example.yaml`](configs/data_token_shards_example.yaml) for
the complete data-loader configuration, including shuffle, worker, prefetch, memory-pinning, and
drop-last controls.

### Configuration

Commands use two YAML files:

```bash
python scripts/train.py \
  --model-config configs/synbios_moe/model.yaml \
  --config configs/synbios_moe/runs/single_fsdp_4gpu.yaml \
  --device cuda
```

The model YAML defines the network. The run YAML selects the dataset, backend, optimizer, parallel
strategy, logging, and checkpoint policy. Run YAML files can use `extends`; `expected_world_size`
guards multi-GPU configurations from being launched with the wrong number of ranks.

<details>
<summary><strong>Formal 293M SynBioS MoE configuration</strong></summary>

| Model setting | Value |
|---|---|
| Architecture | Decoder-only Transformer, RoPE, tied input/output embeddings |
| Total / token-active parameters | 293,494,272 / approximately 123.62M |
| Layers / hidden size / heads | 12 / 768 / 12 |
| Sequence length | 512 |
| FFN | 8 experts, top-2, SwiGLU, expert intermediate size 1,024 |
| Router | Dropless, auxiliary coefficient 0.01, z-loss coefficient 0.001 |
| Dropout | 0.1 |
| Backend | Switchable PyTorch / Triton / CUDA |

| Run setting | Value |
|---|---|
| Parallelism | 4-GPU FSDP full shard, Transformer-block auto wrap |
| Precision | BF16, no GradScaler |
| Batch | Local 112/GPU, global 448, 229,376 tokens/step |
| Gradient accumulation | None |
| Optimizer | AdamW with μP-style parameter groups |
| Learning rate | Peak `1e-3`, 1,000-step warmup, cosine decay, floor `1e-4` |
| Stability | Global gradient-norm clip 5.0, fail immediately on NaN/Inf |
| Packing | Randomized documents, shuffle window 1,024 |

</details>

<a id="training-workflows"></a>

## 🛠️ Training workflows

MiniTrainSys supports Python 3.10–3.12. After completing the [Quick Start](#quick-start), use the
following entry points for GPU and distributed runs.

Validated Linux/NVIDIA server setup:

```bash
bash scripts/bash/setup_storage.sh /data
bash scripts/bash/setup_server.sh
```

Single-GPU CUDA training:

```bash
python scripts/train.py \
  --device cuda \
  --config configs/server/rtx4090_24gb/runs/single_1gpu.yaml \
  --model-config configs/model_125m_moe.yaml
```

Four-GPU FSDP:

```bash
NPROC=4 MODEL_CONFIG=configs/model_125m_moe.yaml \
  bash scripts/bash/distributed.sh fsdp
```

SynBioS training:

```bash
NPROC=4 bash scripts/bash/synbios_moe.sh single fsdp
NPROC=4 bash scripts/bash/synbios_moe.sh multi5_permute fsdp
```

Resume from the safety anchor:

```bash
RESUME=safety NPROC=4 bash scripts/bash/synbios_moe.sh single fsdp
```

Formal probing requires a batch recommendation produced by a completed probe-capacity benchmark:

```bash
STAGE=formal NPROC=4 PROBE_GPUS=4 \
PROBE_BATCH_ENV=<recommended.env> \
  bash scripts/bash/synbios_probes.sh multi5_permute fsdp latest
```

TensorBoard:

```bash
tensorboard --logdir artifacts/runs --host 0.0.0.0 --port 6006
```

### Static validation

```bash
ruff check .
```

<a id="reliability"></a>

## 🛡️ Reliability and recovery

Formal runs are gated by operator parity, training smokes, distributed checkpoint validation, and
multi-step stability at the selected batch.

Each logged point is aggregated across the full logging interval and then across ranks:

| Metric | What it shows |
|---|---|
| `loss/lm_cross_entropy` | Pure next-token loss without router regularization |
| `loss/moe_regularization_total` | Weighted auxiliary and z losses |
| `loss/total` | Actual loss used by backward |
| `tokens_per_sec`, `step_time_ms` | System throughput and complete step latency |
| `data_wait_ms` | Detect DataLoader bottlenecks |
| `gpu_compute_utilization_*` | Per-rank NVML compute utilization |
| `gpu_memory_*_percent_max` | Worst-rank current, reserved, and interval-peak memory |
| `grad_norm`, `grad_clip_fraction` | Early signs of excessive LR or numerical instability |
| `expert_load_cv`, `dead_expert_count` | MoE load balance |

MoE monitoring also records layer-by-expert top-k load and complete router probability heatmaps with
a fixed color scale.

FSDP model and Adam states are distributed, so training checkpoints use DCP rather than a rank-0
`state_dict()`. Evaluation and probing load a consolidated `model.pt`, avoiding the training
optimizer state. `--resume latest` selects only committed checkpoints; `--resume safety` bypasses
the most recent checkpoints when manual rollback is needed.

<a id="documentation"></a>

## 📚 Documentation

Use the [project walkthrough](docs/guides/project_walkthrough.md) for a guided tour, the
[architecture guide](docs/guides/architecture.md) for component boundaries, and the
[artifact layout contract](docs/guides/artifact_layout.md) for reproducibility rules.

### Executable notebooks and entry points

| Entry point | What it runs |
|---|---|
| `tests/example_training.ipynb` | Minimal model, LR, and checkpoint experiment |
| `tests/synbios_moe_end_to_end.ipynb` | Small data → training → evaluation → probe → route loop |
| `tests/operator_bench_linux_server.ipynb` | RTX 4090 Dense/Transformer kernel benchmark |
| `tests/moe_operator_bench_linux_server.ipynb` | Isolated router and fused-MoE benchmark |
| `tests/distributed_server_benchmark.ipynb` | Single/DDP/FSDP scaling and capacity |
| `scripts/bash/synbios_backend_benchmark.sh` | 293M MoE equal-batch and equal-memory backend comparison |

For deeper implementation and evidence:

- [Kernel engineering report](reports/engineering/kernels.md)
- [FSDP and end-to-end report](reports/engineering/distributed_training.md)
- [SynBioS evidence map](reports/synbios_moe/README.md)
- [Artifact layout contract](docs/guides/artifact_layout.md)
- [Project walkthrough](docs/guides/project_walkthrough.md)

<a id="references"></a>

## 📖 References and upstream work

The table below records the projects, implementations, and research used by MiniTrainSys.
“Adapted” means that source-derived code is present in this repository; its license notice is kept
beside the code.

| Reference | Relationship to MiniTrainSys |
|---|---|
| [PyTorch](https://pytorch.org/) · [FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) · [Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) | Tensor/autograd runtime, distributed primitives, FSDP, and DCP APIs. |
| [OpenAI Triton](https://triton-lang.org/) · [Fused Attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) · [Fused Softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) | Triton language and the main algorithmic references for the local attention and row-reduction kernels. |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) · [FlashAttention-2](https://arxiv.org/abs/2307.08691) · [CUTLASS](https://github.com/NVIDIA/cutlass) | The native CUDA attention extension adapts the upstream FlashAttention 2.8.4 source and builds against CUTLASS; their notices are retained under `minitrain/kernels/cuda_ext/csrc/third_party/`. |
| [Liger Kernel](https://github.com/linkedin/Liger-Kernel) | The fused MoE implementation is adapted from Liger Kernel under BSD-2-Clause; see `minitrain/kernels/triton/THIRD_PARTY_NOTICES`. Its RMSNorm, RoPE, SwiGLU, and fused-loss work also informed operator design. |
| [nanoGPT](https://github.com/karpathy/nanoGPT) · [nanochat](https://github.com/karpathy/nanochat) | Compact, script-first training-system references. |
| [TorchTitan](https://github.com/pytorch/torchtitan) · [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) · [DeepSpeed](https://github.com/microsoft/DeepSpeed) | References for separating model code, runtime/configuration, distributed execution, optimizer state, and low-level extensions. |
| [Megatron-Core MoE](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html) · [vLLM](https://github.com/vllm-project/vllm) · [SGLang](https://github.com/sgl-project/sglang) | References for treating Top-K selection, token dispatch, grouped expert execution, and combination as distinct stages. |
| [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) | Collective-communication semantics and operational tuning reference. |
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) · [tiktoken](https://github.com/openai/tiktoken) | References for corpus/tokenization workflow and recorded tokenizer identity. |
| [Allen-Zhu & Li, *Physics of Language Models: Part 3.1*](https://arxiv.org/abs/2309.14316) · [project page](https://physics.allen-zhu.com/part-3-knowledge/part-3-1) | The SynBioS factual-recall setup and the P/Q probing distinction. This repository studies the corresponding mechanism in sparse MoE; it is not a claim of a strict dense-model reproduction. |

The repository-level [reference map](docs/references/reference_map.md) records how the system-level
references influenced individual boundaries; [references.md](docs/references/references.md) is the
shorter source index.

## 📄 License

MiniTrainSys is released under the [MIT License](LICENSE). Third-party CUDA, FlashAttention, and
CUTLASS sources retain their respective upstream licenses.
