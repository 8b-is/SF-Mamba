# Mamba with Batch Folding

Please provide position_indices as a tensor of shape [B, T]. Set the value to 1 for the first token corresponding to the data before batch folding, -1 for the last token, and 0 for all other tokens.

## Install
```
pip install mamba_bfold --user
```
