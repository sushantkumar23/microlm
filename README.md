# MicroLM

This repository contains a PyTorch implementation of the MicroLM model. Improvement over the original nanogpt implementation by Andrej Karpathy are as follows:

1. Rotary embeddings: are used for the attention mechanism

## Setup

Run the following commands to setup the environment and run the training script.

```
git clone https://github.com/sushantkumar23/microlm
cd microlm
pip install -r requirements.txt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade
python cached_fineweb10B.py 10
torchrun --standalone --nproc_per_node=2 pretrain.py
```

## Loss Visualization

Jupyter notebook `Loss_Plot.ipynb` is provided to visualize the loss curves.
