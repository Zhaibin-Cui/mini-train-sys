# Formal probe protocol decision

## Decision

Use the Allen-Zhu bioS probe task matrix for the formal reproduction, while allocating the
training budget from the pilot's convergence evidence. First-token probes are already stable;
whole-attribute probes need a longer budget. This changes only the amount of optimization time,
not the task definitions or held-out evaluation protocol.

| Probe | Rank | Train batch | Validation batch | Steps | Primary endpoint |
|---|---:|---:|---:|---:|---|
| P first | 2 | 128 | 512 | 4,000 | first-token accuracy by position |
| Q first | 16 | 768 | 6,144 | 4,000 | name-only first-token accuracy |
| P whole | 2 | 128 | 512 | 12,000 | secondary whole-attribute accuracy |
| Q whole | 16 | 768 | 6,144 | 12,000 | secondary whole-attribute accuracy |

Whole-attribute P/Q tasks remain in the same run as secondary endpoints. The birthday
whole-attribute task remains omitted because the class space is not defined for that paper
endpoint. The formal config represents these four budgets explicitly and the pipeline resolves
the correct step count per task.

## Why this is the correct trade-off

The capacity benchmark's selected batches are the fastest safe settings on this server, while
the rank and task definitions remain paper-faithful. We use pilot-derived steps to avoid spending
a long budget on already-saturated first probes. The longer whole budget is a sufficiency check,
not a claim that whole-attribute accuracy is the primary Allen-Zhu mechanism result. The exact
batch evidence is in `capacity.md` and its exported benchmark manifest.

## Evidence

See [pilot comparison](pilot_comparison.md), the cross-condition machine-readable summary at
`artifacts/synbios_moe/results/probe_pilot_comparison_20260723/summary.json`, and the batch
benchmark report at [capacity.md](capacity.md). Formal launch remains gated on the monitoring
ETA change and the required regression checks.
