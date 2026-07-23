# P/Q probe pilot comparison

## Question and hypothesis

Can the reproduction recover Allen-Zhu's distinction between fixed-order/local storage in
`bioS single` and direct name-linked storage after `multi5+permute` augmentation? The primary
test is first-token P/Q probing; whole-attribute probing is a secondary robustness check.

## Compared conditions and identity

| Condition | Backbone/checkpoint | Probe stage | Dataset/cache |
|---|---|---|---|
| `single` | `single_fsdp_4gpu`, epoch 540, step 17,280 | pilot, 3,000 steps | 100,000 people, 100,000 P examples, 100,000 Q examples |
| `multi5_permute` | `multi5_permute_fsdp_4gpu`, epoch 108, step 17,388 | pilot, 3,000 steps | 100,000 people, 500,000 P examples, 100,000 Q examples |

Both use the same person-level split: 49,882 training people and 50,118 held-out validation
people. All 22 tasks (11 label targets × P/Q) trained and checkpoint-reloaded validation
completed with zero failures. The machine-readable cross-condition summary is
`artifacts/synbios_moe/results/probe_pilot_comparison_20260723/summary.json`.

## Primary held-out first-token results

Values are validation accuracy (%). `own` is the P position immediately before the target
attribute in the fixed single biography order; `p0` is the earliest P position. Q is the
name-only first-token probe.

| Target | single P p0 | single P own | multi5 P p0 | multi5 P own | single Q | multi5 Q |
|---|---:|---:|---:|---:|---:|---:|
| birth date | 100.00 | 100.00 | 99.75 | 99.75 | 39.17 | 99.74 |
| birth city | 6.30 | 100.00 | 99.51 | 99.95 | 6.21 | 99.44 |
| university | 5.39 | 99.80 | 99.68 | 99.98 | 5.76 | 99.61 |
| major | 5.51 | 100.00 | 99.44 | 99.99 | 5.15 | 99.48 |
| company | 5.23 | 100.00 | 97.39 | 99.66 | 5.10 | 97.31 |
| company city | 11.15 | 100.00 | 97.11 | 99.61 | 13.13 | 97.20 |

Across the six first-token targets, the mean earliest-position P accuracy is 22.26% for
`single` versus 98.81% for `multi5_permute`; mean Q accuracy is 12.42% versus 98.80%.
These are held-out validation values, not training recall.

## Whole-attribute result and interpretation

Whole-attribute performance is heterogeneous: earliest-position P means are 3.11% (`single`)
and 31.47% (`multi5_permute`), while whole-Q means are 3.14% and 32.34%. This does not erase
the first-token result; it shows that exact multi-token classification is a harder secondary
task. We keep whole tasks for paper comparability and report first-token as the primary memory
mechanism result.

## Conclusion / reliability gate

The pilot supports the intended Allen-Zhu trend with a very large held-out effect: `single`
retains attributes in position-local form (low before the target, high at the target), while
`multi5+permute` makes attributes linearly available from the earliest position and directly
from the person's name. The result is not a universal-100% artifact: the single early P/Q
values remain near their class-dependent baselines, and the comparison uses an independent
person split.

Limitations: one seed, one small backbone checkpoint per condition, and a 3,000-step pilot;
these establish the direction, not final paper-scale accuracy. The formal run therefore keeps
the paper's probe protocol (P rank 2/batch 50; Q rank 16/batch 200; 30,000 steps) and uses
first-token as the primary endpoint, with whole-attribute as a secondary endpoint.

## Supporting evidence and next action

- `artifacts/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/pilot/`
- `artifacts/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/pilot/`
- `results/formal_runs/synbios_moe/results/single_fsdp_4gpu/probe_pipeline/pilot/`
- `results/formal_runs/synbios_moe/results/multi5_permute_fsdp_4gpu/probe_pipeline/pilot/`
- `HISTORY.md` entries dated 2026-07-23 for both pilot launches/completions

Before formal launch, update the console to show task, phase, and full-pipeline ETA, run the
required regression gates, and record the paper-faithful formal configuration in HISTORY.
