"""Triton CrossEntropy kernel target.

Useful shapes: `[batch * seq, vocab]` with large vocab. Track numerical stability,
backward correctness, and memory traffic.
"""

