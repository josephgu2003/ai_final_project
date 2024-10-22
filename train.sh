#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12GB

# Commands to execute
module load anaconda3/2022.05
module load gcc/10.1.0
module load cuda/11.8

export CUDA_VISIBLE_DEVICES=0

source activate cs4100
echo 'btw, gpu:t4:1 seems to work if the v100-sxm2 is not available'

python3 -c "import torch; print(torch.cuda.is_available()); x = torch.zeros(1).cuda(); print(x)"
python3 -u train.py