#!/bin/bash

#SBATCH --job-name=R3-inference
#SBATCH --nodelist=devbox
#SBATCH --partition=gpu
#SBATCH --output=logs/inference-output.out
#SBATCH --error=logs/inference-error.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00


split_path="check_model_name"
retrieve_host="http://10.0.12.120"
retrieve_port="8001"
split_host=""
split_port=""

qwen_stop_token_id=151645
num_docs=3
tp=4
model_path="/mnt/raid5/kangjh/downloads/R3-RAG-Qwen"
model_name="R3-RAG-Qwen-Bench"
num_search_one_attempt=5
log_dir=metrics/${model_name}
cd /mnt/raid5/kangjh/Research/ParametricReasoning/RPRAG
source .venv/bin/activate

mkdir -p ${log_dir}
python /mnt/raid5/kangjh/Research/ParametricReasoning/RPRAG/benchmark/R3-RAG/inference.py \
    --model_path=${model_path} \
    --log_dir=${log_dir} \
    --retrieve_url=http://${retrieve_host}:${retrieve_port}/search \
    --num_search_one_attempt=${num_search_one_attempt} \
    --stop_token_id $qwen_stop_token_id \
    --num_of_docs $num_docs \
    --tp $tp \
    --split_url=http://${split_host}:${split_port}/split_query 