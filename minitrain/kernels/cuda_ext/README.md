# CUDA C++ Extensions

Use this area for kernels that need lower-level control than Triton provides.

Good first candidates:

- RMSNorm backward with warp-level reductions.
- FusedLinearCrossEntropy with custom tiling for large vocabularies.
- Experimental communication kernels or PyTorch custom ops.

Keep the Python-facing API compatible with `OpsBackend`.

