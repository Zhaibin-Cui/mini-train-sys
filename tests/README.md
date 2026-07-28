# Tests

`tests/` contains automated correctness and regression checks. Benchmarks and walkthrough
notebooks live in [`benchmarks/`](../benchmarks/) and [`examples/`](../examples/).

Run the hardware-independent suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The current `unit/synbios_moe/` suite checks deterministic data generation, P/Q task definitions,
route metrics, artifact integrity, and report path resolution. CUDA and distributed checks will be
kept under explicit `integration/`, `regression/`, or `smoke/` entry points as they are migrated.
