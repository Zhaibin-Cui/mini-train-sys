# Single fresh P-whole launch decision with ground-truth `t1`

## Question or hypothesis

On the `single` backbone, does appending the correct first attribute token after each biography
prefix make the complete attribute more linearly readable by a fresh P-whole probe?

## Exact compared conditions

The baseline is the completed `single` formal no-`t1` P-whole probe. The intervention uses the
same frozen single backbone, person split, whole-value labels, P `AttributeProbe`, rank-2
low-rank input delta, seed 1337, and five whole attributes. It changes only the input by appending
cached ground-truth `t1` and reading that token. Each head uses 4,000 steps, training batch 128,
and held-out evaluation batch 3,072, exactly matching the accepted multi5+permute P intervention.
No fresh Q task is defined or launched.

## Run/checkpoint and dataset identity

- Backbone:
  `artifacts/synbios_moe/checkpoints/synbios_moe_single_fsdp_4gpu/epoch_000540_step_000017280`.
- Dataset manifest SHA256:
  `144cf49ea607b4a502e5be277dbb63e0e9a08f296596e994cd19d3c6cfb11e25`.
- Probe-cache manifest SHA256:
  `9dcfd2cff38d6f3d29d7f10c2a3247b634f31395b3cda3755a755c8964ffaf5b`.
- Evaluation is the complete person-held-out probe split. These people participated in backbone
  pretraining, so the endpoint measures representation readout rather than unseen-person
  generalization.

## Primary launch metrics

| Item | Value |
|---|---:|
| Fresh heads | 5 P-whole |
| Rank | 2 |
| Steps/head | 4,000 |
| Training batch | 128 |
| Held-out batch | 3,072 |
| Source positions | 6 |
| GPU scheduler | dynamic, 4 × RTX 4090 |

## Supporting artifacts

- Original formal heads:
  `artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/training/`.
- Launch/result root:
  `artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/formal/diagnostics/`.
- Exact command, lifecycle, failures, and completion status: repository-root `HISTORY.md`.

## Interpretation

The matched configuration isolates the dataset/backbone condition: any difference from the
multi5+permute intervention cannot be attributed to P rank, optimizer-step budget, batch size,
label space, or validation protocol.

## Limitations or threats to validity

P turns every biography into six prefix-plus-`t1` examples, so 4,000 updates do not exhaust the
derived training set. The two backbones were trained on different biography augmentation
conditions; this experiment compares their resulting representations and does not isolate MoE
routing causally.

## Next decision/action

Completed: all five heads passed complete held-out validation. The published single-condition
figure contains ground-truth-t1 input consistency, original formal no-`t1` whole accuracy, and
fresh whole accuracy; the unnecessary delta heatmap is omitted. The result is reported in
`ground_truth_first_single_result.md`; no fresh Q experiment was added.
