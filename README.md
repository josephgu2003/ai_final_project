anaconda3/2022.05
Python 3.12
Torch 2.4
Cuda 11.8

use sbatch setup.sh to install conda env
use sbatch train.sh to run training job
use ./srun_job.sh to get gpu in a bash environment

download dataset: wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
download splits: wget http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat -O nyuv2_splits.mat
download swint: wget https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.pth