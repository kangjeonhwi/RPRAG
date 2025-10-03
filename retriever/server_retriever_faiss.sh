#!/bin/bash

#SBATCH --job-name=retriever-server-gpu
#SBATCH --nodelist=devbox
#SBATCH --partition=gpu
#SBATCH --output=logs/retriever-server-%j.out  # %j: Job ID를 로그 파일명에 추가
#SBATCH --error=logs/retriever-server-%j.err   # %j: Job ID를 로그 파일명에 추가
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00

# --- 설정 변수 ---
PROJECT_DIR="/mnt/raid5/kangjh/Research/ParametricReasoning/RPRAG"
VENV_PATH="$PROJECT_DIR/.venv/bin/activate"
SERVER_DIR="$PROJECT_DIR/retriever"
SCRIPT_NAME="retrieve_faiss.py"  # 이전 답변에서 생성한 Python 스크립트 파일명
APP_OBJECT_NAME="app"              # FastAPI 앱 객체 이름
RETRIEVER_PORT=8001
HOST_IP="0.0.0.0"                  # 모든 네트워크 인터페이스에서 접속 허용

echo "--- 리트리버 서버 SLURM 작업 시작 ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Host Node: $(hostname)"
echo "프로젝트 디렉토리: $PROJECT_DIR"

# 프로젝트 루트 디렉토리로 이동
cd "$PROJECT_DIR"
echo "현재 디렉토리: $(pwd)"

cd "$SERVER_DIR"
echo "서버 디렉토리로 이동: $(pwd)"

ACCESSIBLE_IP=$(hostname -I | awk '{print $1}')
echo "서버 바인딩 주소: $HOST_IP:$RETRIEVER_PORT"
echo "외부 접속 주소 (예상): http://$ACCESSIBLE_IP:$RETRIEVER_PORT"
echo "------------------------------------"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate faiss
while true; do
    echo "[$(date)] Uvicorn 서버를 시작합니다..."
    uvicorn ${SCRIPT_NAME%.py}:$APP_OBJECT_NAME --host $HOST_IP --port $RETRIEVER_PORT
    
    # 서버 프로세스가 종료되면 3초 후 재시작
    echo "[$(date)] 리트리버 서버가 종료되었습니다. 3초 후 재시작합니다..."
    sleep 3
done