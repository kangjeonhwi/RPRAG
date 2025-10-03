from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from flashrag.retriever import DenseRetriever
from flashrag.config import Config
import uvicorn
from typing import List, Any

class QueryRequest(BaseModel):
    query: str
    instruction : str = "Retrieve relevant documents to assist in answering the question."

class BatchQueryRequest(BaseModel):
    queries: List[str]
    instruction : str = "Retrieve relevant documents to assist in answering the question."

app = FastAPI(title="Dense Retriever Service", version="1.0")
dense_retriever: DenseRetriever = None

@app.on_event("startup")
def load_retriever():
    global dense_retriever
    if dense_retriever is None:
        print("Loading Retriever...")
        retrieval_config ={
                'retrieval_method': 'e5',
                'retrieval_model_path': '/mnt/raid5/kangjh/downloads/e5-base-v2',
                'retrieval_query_max_length': 256,
                'retrieval_use_fp16': True,
                'retrieval_topk': 5,
                'retrieval_batch_size': 32,
                'index_path': "/mnt/raid5/kangjh/downloads/Tevatron/wikipedia-nq-corpus-flashragformat/index/e5_Flat.index/e5_Flat.index",
                'corpus_path': "/mnt/raid5/kangjh/downloads/Tevatron/wikipedia-nq-corpus-flashragformat/processed_corpus.jsonl",
                'save_retrieval_cache': False,
                'use_retrieval_cache': False,
                'retrieval_cache_path': None,
                'use_reranker': False,
                'faiss_gpu': False,
                'use_sentence_transformer': False,
                'retrieval_pooling_method': 'mean',
                'instruction' : "Retrieve relevant documents to assist in answering the question."
            }
        dense_retriever = DenseRetriever(retrieval_config)
        print("Retriever loaded successfully.")

@app.post("/search")
def search(query_request: QueryRequest):
    query = query_request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        retrieval_results = dense_retriever.search(query)
        return retrieval_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/search_batch")
def search_batch(query_request: BatchQueryRequest):
    queries = query_request.queries
    if not queries:
        raise HTTPException(status_code=400, detail="Queries cannot be empty.")
    try:
        retrieval_results = dense_retriever.batch_search(queries)
        return retrieval_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
