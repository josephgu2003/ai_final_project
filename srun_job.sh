srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=12 --gres=gpu:v100-sxm2:1 --mem=12G --time=03:00:00 --pty /bin/bash
