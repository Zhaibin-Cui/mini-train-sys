# ⚙️ Engineering reports

| Report | Focus | Headline |
|---|---|---|
| [Kernel engineering](kernels.md) | 8 operator families, Triton/native CUDA design, isolated benchmark | **1.22–6.51×** vs shape-matched Torch |
| [Distributed & end-to-end](distributed_training.md) | FSDP scaling, equal-work and equal-memory backend comparison | **92.24%** 4-GPU efficiency; **2.20×** at equal VRAM |

Machine-readable evidence lives under `results/benchmarks/`; historical root report paths are
short compatibility pages so there is only one authoritative copy of each conclusion.
