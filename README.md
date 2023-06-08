# DVSOD-Baseline
This repository provides the source code of DVSOD baseline.

## Installation 

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

## Getting Started

First download the `DViSal` [dataset](https://github.com/DVSOD/DVSOD-DViSal). Then the model can be used in just a few adaptions to start training:

1. Set your `DViSal dataset path` and `ckpt path` in `train.py`
2. Start training, with ```python train.py```
