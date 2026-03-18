# Trace Generator Module
This is the module that generates traces from distributes training runs.

## Structure of this folder

### File gpu_activity_exporter.py:
It implements the monitoring modules.

### File imagenet_vit_huge_FSDP_nccl.py:
It trains Vision Transformer huge over Imagenet with model parallelism enabled.
It generates a trace of operations on GPU rank 0.
