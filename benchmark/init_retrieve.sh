#!/bin/bash

#SBATCH --job-name=pre_retrieve
#SBATCH --nodelist=devbox
#SBATCH --partition=gpu
#SBATCH --output=/mnt/raid5/kangjh/Research/ParametricReasoning/RPRAG/benchmark/logs/rt_test.out
#SBATCH --error=/mnt/raid5/kangjh/Research/ParametricReasoning/RPRAG/benchmark/logs/rt_test.err

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00


cd /mnt/raid5/kangjh/Research/ParametricReasoning/RPRAG
source .venv/bin/activate

cd ./benchmark
uv run python /mnt/raid5/kangjh/Research/ParametricReasoning/RPRAG/benchmark/init_retrieve.py
