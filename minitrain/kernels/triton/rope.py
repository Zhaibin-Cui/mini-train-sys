"""Triton RoPE kernel target.

Primary optimization: fuse Q and K rotary application into one launch and avoid
materializing intermediate rotated tensors.
"""

