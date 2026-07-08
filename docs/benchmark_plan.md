# Benchmark Plan

## Operator Benchmarks

- correctness: max absolute error, max relative error, gradient check where practical;
- speed: p50/p95 latency with CUDA synchronization;
- memory: peak allocated and reserved memory;
- shape sweep: hidden size, sequence length, vocab size, dtype.

## Training Benchmarks

- fixed model and fixed data order;
- compare `torch`, `torch.compile` if enabled, and `triton`;
- record tokens/sec, step time, peak VRAM, validation loss, and MFU estimate.

## Distributed Benchmarks

- fixed global batch when measuring scaling efficiency;
- fixed per-GPU batch when measuring throughput scaling;
- DDP knobs: bucket size, gradient_as_bucket_view, static graph;
- FSDP knobs: sharding strategy, mixed precision, prefetch behavior.

