# Distributed Strategies

This folder is inspired by TorchTitan's distributed components, Megatron-LM's
model-parallel process-group boundaries, and DeepSpeed's runtime separation.

Start simple:

- `single.py`: timing baseline.
- `ddp.py`: first real distributed baseline.
- `fsdp.py`: memory-saving baseline.
- `custom_allreduce.py`: teaching implementation for communication analysis.

Later extensions can add tensor parallel, pipeline parallel, sequence parallel,
and ZeRO-style optimizer sharding without changing model code.

