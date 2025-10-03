import json
import os
import time
import random
from typing import Dict, List, Any

import hydra
import tqdm
from omegaconf import DictConfig, OmegaConf
from flashrag.retriever import DenseRetriever

def run_preprocessing(cfg: DictConfig):
    """
    RAG를 위한 데이터 전처리 스크립트.
    1. DenseRetriever를 로컬에서 직접 초기화하여 네트워크 오버헤드 제거.
    2. GPU를 사용한 FAISS 인덱싱으로 검색 속도 가속화.
    3. 처리된 결과를 즉시 파일에 저장하여, 중단 시 해당 지점부터 이어하기 기능 구현.
    """
    input_file = cfg.input_file
    output_file = cfg.rag_params.preprocessed_file

    print("--- 🚀 Starting RAG Data Pre-processing ---")

    processed_questions = set()
    if os.path.exists(output_file):
        print(f"✅ Output file '{output_file}' found. Checking for completed records to resume.")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # 각 라인이 유효한 JSON인지 확인
                    if line.strip():
                        record = json.loads(line)
                        # 'problem' 또는 'question' 키를 사용하여 고유 질문 식별
                        question = record.get('problem', record.get('question'))
                        if question:
                            processed_questions.add(question)
            print(f"📈 Found {len(processed_questions)} already processed records. Will skip them.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"🤔 WARNING: Could not parse output file '{output_file}'. Starting from scratch. Error: {e}")
            # 문제가 있는 파일은 덮어쓰기 위해 processed_questions를 비움
            processed_questions.clear()


    # --- 데이터 로딩 및 처리할 데이터 선별 (2/3) ---
    print(f"Loading records from: {input_file}")
    all_records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_records.append(json.loads(line))
            
    # 아직 처리되지 않은 레코드만 필터링
    records_to_process = [
        rec for rec in all_records
        if rec.get('problem', rec.get('question', '')) not in processed_questions
    ]
    
    total_count = len(all_records)
    processed_count = len(processed_questions)
    to_process_count = len(records_to_process)

    print(f"📊 Total records: {total_count} | Already processed: {processed_count} | New records to process: {to_process_count}")

    if not records_to_process:
        print("🎉 No new records to process. Pre-processing is already complete.")
        return

    # 디버그 모드일 경우, 처리할 레코드 중에서 샘플링
    if cfg.debug:
        sample_size = min(8, len(records_to_process))
        records_to_process = random.sample(records_to_process, sample_size)
        print(f"🐞 Debug mode: Sampled {len(records_to_process)} new records for pre-processing.")

    print("\nInitializing DenseRetriever... (This may take a moment)")
    retriever_config = OmegaConf.to_container(cfg.retriever, resolve=True)
    dense_retriever = DenseRetriever(retriever_config)
    print("✅ Retriever initialized successfully.")

    all_questions = [rec.get('problem', rec.get('question', '')) for rec in records_to_process]
    batch_size = retriever_config.get('retrieval_batch_size', 32)


    with open(output_file, "a", encoding='utf-8') as f:
        # tqdm을 사용하여 배치 처리 진행 상황 시각화
        for i in tqdm.tqdm(range(0, len(all_questions), batch_size), desc="Retrieving documents"):
            batch_questions = all_questions[i:i + batch_size]
            batch_records = records_to_process[i:i + batch_size]
            
            try:
                # retriever의 batch_search 직접 호출
                retrieved_docs_batch = dense_retriever.batch_search(batch_questions)

                # 배치 단위로 결과 처리 및 파일 저장
                for rec, docs in zip(batch_records, retrieved_docs_batch):
                    doc_contents = [d.get('contents', '') for d in docs[:cfg.rag_params.num_of_docs]]
                    rec['doc'] = "\n".join(doc_contents)
                    
                    # JSON으로 변환하여 파일에 한 줄씩 쓰기 (즉시 저장)
                    json.dump(rec, f, ensure_ascii=False)
                    f.write('\n')

            except Exception as e:
                print(f"\n❌ CRITICAL ERROR during retrieval batch: {e}")
                print("🛑 Aborting pre-processing. You can restart the script to resume from the last saved point.")
                return

    print(f"\n💾 RAG pre-processing complete. {len(records_to_process)} new records saved to: {output_file}")


@hydra.main(version_base=None, config_path="conf", config_name="retrieve_config")
def main(cfg: DictConfig):
    run_preprocessing(cfg)

if __name__ == "__main__":
    main()