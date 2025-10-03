import json
import os
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any


class Settings:
    # 경로 설정
    retrieval_model_path: str = '/mnt/raid5/kangjh/downloads/e5-base-v2'
    index_path: str = "/mnt/raid5/kangjh/downloads/Tevatron/wikipedia-nq-corpus-flashragformat/index/e5_ivfpq.index"
    corpus_path: str = "/mnt/raid5/kangjh/downloads/Tevatron/wikipedia-nq-corpus-flashragformat/processed_corpus.jsonl"

    # Retriever 성능 파라미터
    retrieval_topk: int = 5
    retrieval_batch_size: int = 128  # API 배치 검색 시 내부적으로 사용할 배치 크기
    retrieval_use_fp16: bool = True
    retrieval_nprobe: int = 2048

# --------------------------------------------------------------------------
# 2. FastAPI 애플리케이션 및 Pydantic 모델 정의
# --------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str

class BatchQueryRequest(BaseModel):
    queries: List[str]

# 설정 인스턴스 생성
settings = Settings()

# FastAPI 앱 생성
app = FastAPI(
    title="GPU-Accelerated Faiss Retriever Service",
    version="2.0",
    description="A high-performance retriever using SentenceTransformers and faiss-gpu."
)

# 전역 변수로 모델, 코퍼스, GPU 인덱스를 관리
# 서버 시작 시 한 번만 로드됩니다.
model: SentenceTransformer = None
corpus: List[str] = None
gpu_index: faiss.GpuIndex = None

# --------------------------------------------------------------------------
# 3. 서버 시작 시 모델 및 인덱스 로딩
# --------------------------------------------------------------------------
@app.on_event("startup")
def load_retriever_components():
    """
    FastAPI 서버가 시작될 때 모델, 코퍼스, Faiss 인덱스를 GPU에 로드합니다.
    """
    global model, corpus, gpu_index

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This server requires a GPU.")

    device = 'cuda'
    print(f"--- 🚀 Loading Retriever Components to {device.upper()} ---")

    # 1. SentenceTransformer 모델 로딩
    print(f"Loading embedding model from: {settings.retrieval_model_path}")
    model = SentenceTransformer(settings.retrieval_model_path, device=device)
    if settings.retrieval_use_fp16:
        model.half()
    print("✅ Embedding model loaded.")

    # 2. 코퍼스 로딩
    print(f"Loading corpus from: {settings.corpus_path}")
    try:
        corpus = []
        with open(settings.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                corpus.append(json.loads(line).get('contents', ''))
        print(f"✅ Corpus loaded with {len(corpus)} documents.")
    except Exception as e:
        raise RuntimeError(f"Failed to load corpus file: {e}")

    # 3. Faiss 인덱스 로딩 및 GPU로 이동
    print(f"Loading FAISS index from: {settings.index_path}")
    try:
        cpu_index = faiss.read_index(settings.index_path)
        cpu_index.nprobe = settings.retrieval_nprobe
        print(f"✅ Index loaded on CPU. nprobe set to {cpu_index.nprobe}.")
        
        print("Moving FAISS index to GPU...")
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = settings.retrieval_use_fp16
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
        print("✅ FAISS index is now on GPU and ready to serve.")
    except Exception as e:
        raise RuntimeError(f"Failed to load or move FAISS index to GPU: {e}")

# --------------------------------------------------------------------------
# 4. 검색 로직을 수행하는 내부 함수
# --------------------------------------------------------------------------
def _perform_search(queries: List[str]) -> List[List[Dict[str, Any]]]:
    """
    주어진 쿼리 리스트에 대해 GPU Faiss 검색을 수행하고 결과를 반환합니다.
    """
    try:
        prefixed_queries = [f"query: {q}" for q in queries]

        # 쿼리 임베딩
        query_embeddings = model.encode(
            prefixed_queries,
            batch_size=settings.retrieval_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Faiss 검색
        distances, ids = gpu_index.search(query_embeddings.astype(np.float32), settings.retrieval_topk)

        # 결과 포맷팅
        all_results = []
        for i, query in enumerate(queries):
            query_results = []
            for j in range(settings.retrieval_topk):
                doc_id = ids[i][j]
                if doc_id != -1:
                    query_results.append({
                        "id": int(doc_id),
                        "content": corpus[doc_id],
                        "score": float(distances[i][j])
                    })
            all_results.append(query_results)
        
        return all_results

    except Exception as e:
        print(f"Error during search: {e}")
        # 실제 운영 환경에서는 더 상세한 로깅이 필요합니다.
        raise HTTPException(status_code=500, detail=f"An internal error occurred during search: {e}")


@app.post("/search", response_model=List[Dict[str, Any]])
def search(request: QueryRequest):
    """
    단일 쿼리에 대한 검색을 수행합니다.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    results = _perform_search([request.query])
    return results[0]

@app.post("/search_batch", response_model=List[List[Dict[str, Any]]])
def search_batch(request: BatchQueryRequest):
    """
    여러 쿼리에 대한 배치 검색을 수행합니다.
    """
    if not request.queries:
        raise HTTPException(status_code=400, detail="Queries cannot be empty.")
    
    return _perform_search(request.queries)

if __name__ == "__main__":
    # uvicorn your_script_name:app --host 0.0.0.0 --port 8000
    # 예: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8001)