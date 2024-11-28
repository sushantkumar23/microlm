# MicroLM

This repository contains a PyTorch implementation of the MicroLM model. Improvement over the original nanogpt implementation by Andrej Karpathy are as follows:

- [x] Rotary embeddings: Using rotary embeddings instead of learned embeddings for positional encoding
- [x] RMS normalization: Using RMS normalization instead of LayerNorm
- [x] Coordinate descent tuning: Using coordinate descent tuning when updating the model weights
- [x] Bfloat16 precision: Using bfloat16 precision instead of fp16

## Setup

Run the following commands to setup the environment and run the training script.

```
git clone https://github.com/sushantkumar23/microlm
cd microlm
pip install -r requirements.txt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade
python cached_fineweb10B.py 10
torchrun --standalone --nproc_per_node=1 pretrain.py
```

## Loss Visualization

Jupyter notebook `Loss_Plot.ipynb` is provided to visualize the loss curves.
