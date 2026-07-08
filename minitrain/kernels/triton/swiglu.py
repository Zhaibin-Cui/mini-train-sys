"""Triton SwiGLU kernel target.

Start with elementwise `silu(gate) * up`; later consider fusing with surrounding
linear projections only if the benchmark shows launch/memory overhead dominates.
"""

