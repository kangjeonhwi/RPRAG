#!/bin/bash

#SBATCH --job-name=bench_test
#SBATCH --nodelist=devbox
#SBATCH --partition=gpu
#SBATCH --output=/mnt/raid5/kangjh/Research/ParametricReasoning/RPRAG/benchmark/logs/bench_test.out
#SBATCH --error=/mnt/raid5/kangjh/Research/ParametricReasoning/RPRAG/benchmark/logs/bench_test.err

#SBATCH --gres=gpu:2

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00


cd /mnt/raid5/kangjh/Research/ParametricReasoning/RPRAG
source .venv/bin/activate

cd ./benchmark
uv run python /mnt/raid5/kangjh/Research/ParametricReasoning/RPRAG/benchmark/benchmark.py
