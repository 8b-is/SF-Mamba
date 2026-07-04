# Non-causal (normal) depthwise conv1d in CUDA with Batch folded tensor

Supported features:
- Support fp32, fp16, bf16.
- Kernel size 3.

Please provide position_indices as a tensor of shape [B, T]. Set the value to 1 for the first token corresponding to the data before batch folding, -1 for the last token, and 0 for all other tokens.

## Setup

```
pip install non_causal_conv1d --user
```

