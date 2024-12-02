# Uncertainty-Aware Monocular Depth Estimation

## Overview
We train a monocular depth estimation model on NYU Depth v2 that is uncertainty-aware. We largely follow the uncertainty quantification methods from the 2021 Neurips paper ["What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"](https://proceedings.neurips.cc/paper_files/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf), but utilize a modern vision transformer backbone (Swin) combined with a custom multi-scale decoder inspired by UNet. Thanks to everyone for contributing!

## Prerequisites

Conda is required to setup the environment, but can be optional if you modify the setup scripts.
The dependencies can be found in setup.sh, but as an overview, we use Python 3.12, PyTorch 2.4, and Cuda 11.8 in our system.

## Installation

All scripts are written for SLURM, but can be easily modified (remove module load) to act as simple Bash scripts.

To install the conda environment:
```Bash 
sbatch setup.sh 
```

To download dataset, splits, and pre-training for Swin: 

```Bash 
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
wget http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat -O nyuv2_splits.mat
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.pth
```

Construct model directory and move the Swin checkpoint to it:
```Bash
mkdir models
mv upernet_swin_tiny_patch4_window7_512x512.pth models/
```

## Usage
Run sbatch train.sh to launch a training job:
```Bash
sbatch train.sh 
```

Then check the log folder for outputs.

To run the demo:
```Bash 
python demo.py last_model.pth khoury_test_image.jpg 
```

## Acknowledgments
[Swin Transformer for Semantic Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)

 ["What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"](https://proceedings.neurips.cc/paper_files/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf)