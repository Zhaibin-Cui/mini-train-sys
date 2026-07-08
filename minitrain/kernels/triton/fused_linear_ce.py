"""Triton FusedLinearCrossEntropy kernel target.

This is the highest-value kernel for interviews because it avoids materializing
the full logits tensor and can use chunking for memory savings.
"""

