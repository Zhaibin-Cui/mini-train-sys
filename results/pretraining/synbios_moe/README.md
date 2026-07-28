# SynBioS pretraining

The two formal runs use the same profile table, tokenizer, model, optimizer, and token budget. They
differ only in how each person's facts are rendered.

| Condition | Training data | Endpoint |
|---|---:|---:|
| `single` | 100,000 people × 1 fixed-order biography | epoch 540, step 17,280 |
| `multi5_permute` | 100,000 people × 5 rewritten, permuted biographies | epoch 108, step 17,388 |

The shared profile-table SHA-256 is
`7d239f046cb5e16ac3d8d7636b6901a2430f2ccb8dc1179063e4eaed92256da1`.

Prepare and train one condition:

```bash
python scripts/synbios_moe.py prepare \
  --output artifacts/synbios_moe/single \
  --variant single

NPROC=4 bash scripts/bash/synbios_moe.sh single fsdp
```

Replace `single` with `multi5_permute`; its preparation variant is `multi5+permute`.

The retained pretraining evidence is split into [`datasets/`](datasets/), [`runs/`](runs/),
[`checkpoints/`](checkpoints/), and [`preparation_logs/`](preparation_logs/). Downstream evidence is
kept with the stage that produced it:

- [cloze validation](../../cloze/synbios_moe/);
- [P/Q probes and mechanism diagnostics](../../probes/synbios_moe/);
- [cross-stage study index](../../catalog/studies/synbios_moe.json).
