#!/bin/bash

#SBATCH --job-name=NaiveGen_inf
#SBATCH --nodelist=devbox
#SBATCH --partition=gpu
#SBATCH --output=/home/kangjh/Research/ParametricReasoning/RPRAG/logs/naive_bench_output.out
#SBATCH --error=/home/kangjh/Research/ParametricReasoning/RPRAG/logs/naive_bench_error.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source /home/kangjh/Research/ParametricReasoning/RPRAG/.venv/bin/activate
python /home/kangjh/Research/ParametricReasoning/RPRAG/benchmark/NaiveGeneration/inference.py