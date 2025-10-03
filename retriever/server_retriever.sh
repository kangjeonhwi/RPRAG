#!/bin/bash

#SBATCH --job-name=retriever-server
#SBATCH --nodelist=devbox
#SBATCH --partition=gpu
#SBATCH --output=logs/retriever-server.out
#SBATCH --error=logs/retriever-server.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00

cd /mnt/raid5/kangjh/Research/ParametricReasoning/RPRAG
source .venv/bin/activate
cd /mnt/raid5/kangjh/Research/ParametricReasoning/RPRAG/retriever

retriever_port=8001

host=$(hostname -I | awk '{print $1}')
echo "host: $host"

while true; do
    uvicorn retriever_service:app --host $host --port $retriever_port
    echo 'retriever服务退出，3秒后重启...'
    sleep 3
done
" Enter