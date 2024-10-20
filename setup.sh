#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=16GB

# Commands to execute
module load anaconda3/2019.10
# module load gcc/9.2.0
# module load cuda/11.3

echo 'btw, gpu:t4:1 seems to work if the v100-sxm2 is not available'

conda create -n cs4100 python=3.8 -y

conda activate cs4100

# conda install pytorch=1.11.0 torchvision -c pytorch

pip install -U openmim
mim install mmcv-full
pip install mmsegmentation

# source activate cs4100

# pip install torch torchvision torchaudio numpy --index-url https://download.pytorch.org/whl/cu118

# # Model deps

# # https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md
# pip install -U openmim
# mim install mmcv-full
# pip install mmsegmentation
# export CUDA_HOME=/usr/local/cuda-10.2

