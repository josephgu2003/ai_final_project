#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --partition=short
#SBATCH --cpus-per-task=24
#SBATCH --mem=16GB

# Commands to execute
module load anaconda3/2022.05
module load gcc/10.1.0
module load cuda/11.8

conda create -n cs4100 python=3.12

source activate cs4100

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
