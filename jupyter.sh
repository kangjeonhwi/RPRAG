#!/bin/bash

#SBATCH --job-name=jupyter_kernel
#SBATCH --nodelist=devbox
#SBATCH --partition=gpu
#SBATCH --output=logs/jupyter.out
#SBATCH --error=logs/jupyter.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Chage Directory to project folder
cd /mnt/raid5/kangjh/Research/ParametricReasoning/RPRAG
source .venv/bin/activate

echo "Python path: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
# Run the code

jupyter notebook --ip=0.0.0.0 --port=9799 --no-browser