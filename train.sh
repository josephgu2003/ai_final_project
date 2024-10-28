#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12GB

# Commands to execute
module load anaconda3/2022.05
module load gcc/10.1.0
module load cuda/11.8

export CUDA_VISIBLE_DEVICES=0

source activate cs4100
echo 'btw, gpu:t4:1 seems to work if the v100-sxm2 is not available'

base_dir=./log
exp_name=exp_bs_4
folder=${base_dir}/${exp_name}
mkdir $folder

python -u train.py --base_dir $base_dir --exp_name $exp_name --bs 4 | tee $folder/log_stdout.txt # use this to log extra information not handled otherwise 
