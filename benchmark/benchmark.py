import json
import os
import random
import re
import time
from typing import Dict, List, Any, Optional

import hydra
import requests
import tqdm
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams

# =====================================================================================
# SECTION 1: SHARED UTILITY FUNCTIONS (모든 모드에서 공유)
# =====================================================================================

def mystrip(one_str: str) -> str:
    """공백, 개행, # 문자 제거"""
    return one_str.strip().strip("\\n").strip("#")

def extract_substring(text: str, start_str: str, stop_strs: List[str]) -> Optional[str]:
    """특정 시작 문자열과 종료 문자열 리스트 사이의 텍스트 추출"""
    start_index = text.find(start_str)
    if start_index == -1:
        return None
    start = start_index + len(start_str)
    
    end = len(text)
    for stop_str in stop_strs:
        temp_index = text.find(stop_str, start)
        if temp_index != -1 and temp_index < end:
            end = temp_index
            
    return mystrip(text[start:end]) if start < end else None

def split_response(response: str) -> Dict[str, Any]:
    """LLM 응답을 analysis, query, answer로 파싱"""
    mydict = {"original": response, "analysis": None, "query": None, "answer": None}
    
    str_analysis = "The problem analysis:"
    str_query = "The retrieval query:"
    str_answer = "The final answer:"
    
    common_stop_strs = [str_analysis, str_query, str_answer, "The retrieval documents:", "###", "####"]
    query_stop_strs = common_stop_strs + ["\nStep", "?"]
    answer_stop_strs = common_stop_strs + ["\nStep"]

    analysis = extract_substring(response, str_analysis, common_stop_strs)
    if analysis is None:
        return mydict
    mydict["analysis"] = analysis

    # 분석 내용 이후부터 query와 answer 검색
    analysis_end_idx = response.find(str_analysis) + len(str_analysis)
    search_area = response[analysis_end_idx:]

    query_idx = search_area.find(str_query)
    answer_idx = search_area.find(str_answer)

    # query와 answer가 모두 존재하며, query가 먼저 나온 경우
    if query_idx != -1 and (answer_idx == -1 or query_idx < answer_idx):
        mydict["query"] = extract_substring(search_area, str_query, query_stop_strs)
    # answer가 존재하고, query가 없거나 answer가 먼저 나온 경우
    elif answer_idx != -1:
         mydict["answer"] = extract_substring(search_area, str_answer, answer_stop_strs)

    return mydict

def get_retrieval(retrieve_url: str, queries: List[str], batch_size: int = 64, retry: int = 3) -> List[Any]:
    """배치 처리 및 재시도 로직이 포함된 문서 검색 API 호출"""
    results = []
    for i in tqdm.tqdm(range(0, len(queries), batch_size), desc="Retrieving documents"):
        subset = queries[i:i + batch_size]
        for attempt in range(retry):
            try:
                response = requests.post(retrieve_url, json={"queries": subset}, headers={"Content-Type": "application/json"}, timeout=30)
                if response.status_code == 200 and response.json():
                    results.extend(response.json())
                    break
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}, attempt {attempt + 1}/{retry}...")
                time.sleep(2)
        else:
            raise ValueError(f"Failed to retrieve queries after {retry} attempts: {subset}")
    return results

# =====================================================================================
# SECTION 2: MODE-SPECIFIC PROMPT GENERATORS
# =====================================================================================

# --- Naive Mode Prompts ---
def generate_naive_cot_prompt(question: str) -> List[Dict[str, str]]:
    system_message = """You are a helpful assistant that thinks through problems step by step before providing a final answer based on your own knowledge.

For any question, please structure your response in this format:
The problem analysis: [Provide detailed step-by-step reasoning]
The final answer: [Provide the concise final answer]"""
    user_message = f"The question: {question}"
    return [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

def generate_naive_direct_prompt(question: str) -> List[Dict[str, str]]:
    system_message = "Answer the question based on your own knowledge. Only give me the answer and do not output any other words."
    user_message = f"The question: {question}"
    return [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

# --- RAG Mode Prompts ---
def generate_rag_cot_prompt(question: str, doc: str) -> List[Dict[str, str]]:
    system_message = f"""You are a helpful assistant that answers questions based on document retrieval with step-by-step reasoning.
Your answer must be fully based on the document content.

The following are the provided documents:
{doc}

Please structure your response in this format:
The problem analysis: [Provide detailed step-by-step reasoning based on the documents]
The final answer: [Provide the concise final answer based on the documents]"""
    user_message = f"The question: {question}"
    return [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

def generate_rag_direct_prompt(question: str, doc: str) -> List[Dict[str, str]]:
    system_message = f"Answer the question based on the given document. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n{doc}"
    user_message = f"The question: {question}"
    return [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

# --- RL-Iterative Mode Utilities & Prompts ---
def split_query_remote(split_url: str, queries: List[str], batch_size: int = 64, retry: int = 3) -> List[List[str]]:
    """질의 분할 API 호출"""
    results = []
    for i in tqdm.tqdm(range(0, len(queries), batch_size), desc="Splitting queries"):
        subset = queries[i:i + batch_size]
        for attempt in range(retry):
            try:
                response = requests.post(split_url, json={"queries": subset}, headers={"Content-Type": "application/json"}, timeout=30)
                if response.status_code == 200 and response.json().get("response"):
                    results.extend(response.json()["response"])
                    break
            except requests.exceptions.RequestException as e:
                print(f"Query split request failed: {e}, attempt {attempt + 1}/{retry}...")
                time.sleep(2)
        else:
            print(f"Failed to split queries, returning original: {subset}")
            results.extend([[q] for q in subset]) # 실패 시 원본 쿼리 반환
    return results
    
def generate_rl_step_prompt(context: str) -> List[Dict[str, str]]:
    """RL 모드의 각 스텝에 사용될 프롬프트 생성"""
    # RL 모델은 주로 instruction-tuned 되어 있으므로 간단한 프롬프트 구조 사용
    system_message = "You are a helpful assistant. Please follow the user's instructions carefully."
    user_message = context
    return [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

def generate_rl_direct_prompt(question: str, doc: str) -> List[Dict[str, str]]:
    """RL 모드의 fallback용 직접 답변 프롬프트"""
    system_message = "You are a helpful assistant."
    user_message = (f"Please directly and briefly give the final answer of the following question only one step based on "
                    f"the given documents. Don't give too much analysis. Please give the final answer.\n\n"
                    f"The retrieval documents:\n{doc}\n\nThe question: {question}")
    return [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]


# =====================================================================================
# SECTION 3: CORE LOGIC FOR EACH MODE
# =====================================================================================

def solve_naive(cfg: DictConfig, ckpt: LLM, records: List[Dict]):
    """Naive Generation 모드 실행"""
    print("Running in 'naive' mode...")
    sampling_params = SamplingParams(temperature=cfg.params.temperature, max_tokens=cfg.params.max_tokens)
    
    # 1. Main Pass (CoT)
    print("--- Naive Mode: Main Pass (CoT) ---")
    messages = [generate_naive_cot_prompt(rec['question']) for rec in records]
    outputs = [o.outputs[0].text for o in ckpt.chat(messages, sampling_params)]
    
    for i, (rec, out) in enumerate(zip(records, outputs)):
        rec['history'] = rec.get('history', [])
        rec['history'].append({'pass': 'cot', 'output': out})
        val = split_response(out)
        if val['answer']:
            rec['answer'] = val['answer']
            rec['state'] = 'done'
        else:
            rec['state'] = 'failed_cot'
    
    # 2. Fallback Pass (Direct)
    print("--- Naive Mode: Fallback Pass (Direct) ---")
    remain_idxs = [i for i, rec in enumerate(records) if rec['state'] != 'done']
    if not remain_idxs:
        print("All records solved in the main pass.")
        return

    remain_messages = [generate_naive_direct_prompt(records[i]['question']) for i in remain_idxs]
    remain_outputs = [o.outputs[0].text for o in ckpt.chat(remain_messages, sampling_params)]
    
    for idx, out in zip(remain_idxs, remain_outputs):
        records[idx]['history'].append({'pass': 'direct', 'output': out})
        records[idx]['answer'] = mystrip(out) # 직접 답변이므로 strip만 수행
        records[idx]['state'] = 'done_direct'

def solve_rag(cfg: DictConfig, ckpt: LLM, records: List[Dict]):
    print("Running in 'rag' mode...")
    print("--- RAG Mode: Main Pass (RAG-CoT) ---")
    sampling_params = SamplingParams(temperature=cfg.params.temperature, max_tokens=cfg.params.max_tokens)
    messages = [generate_rag_cot_prompt(rec['question'], rec.get('doc', '')) for rec in records]
    outputs = [o.outputs[0].text for o in ckpt.chat(messages, sampling_params)]
    
    for rec, out in zip(records, outputs):
        rec['history'] = rec.get('history', [])
        rec['history'].append({'pass': 'rag_cot', 'output': out})
        val = split_response(out)
        if val['answer']:
            rec['answer'] = val['answer']
            rec['state'] = 'done'
        else:
            rec['state'] = 'failed_rag_cot'
    
    print("--- RAG Mode: Fallback Pass (Direct RAG) ---")
    remain_idxs = [i for i, rec in enumerate(records) if rec['state'] != 'done']
    if not remain_idxs:
        print("All records solved in the main pass.")
        return

    remain_messages = [generate_rag_direct_prompt(records[i]['question'], records[i].get('doc', '')) for i in remain_idxs]
    remain_outputs = [o.outputs[0].text for o in ckpt.chat(remain_messages, sampling_params)]
    
    for idx, out in zip(remain_idxs, remain_outputs):
        records[idx]['history'].append({'pass': 'rag_direct', 'output': out})
        records[idx]['answer'] = mystrip(out)
        records[idx]['state'] = 'done_direct'

def solve_rl_iterative(cfg: DictConfig, ckpt: LLM, records: List[Dict]):
    """RL-based Iterative Search 모드 실행"""
    print("Running in 'rl_iterative' mode...")
    sampling_params = SamplingParams(
        temperature=cfg.params.temperature, 
        max_tokens=cfg.params.max_tokens, 
        stop_token_ids=[cfg.model.stop_token_id]
    )

    # 초기 context 설정
    for rec in records:
        rec['context'] = f"The question: {rec['question']}"
        rec['state'] = 'pending'
        rec['docs_retrieved'] = []
        rec['history'] = []

    # 1. Iterative Search
    for turn in range(cfg.rl_params.num_search_one_attempt):
        print(f"\n--- RL Mode: Turn {turn + 1}/{cfg.rl_params.num_search_one_attempt} ---")
        
        pending_idxs = [i for i, rec in enumerate(records) if rec['state'] == 'pending']
        if not pending_idxs:
            print("No more pending records. Finishing iterative search.")
            break
        print(f"Processing {len(pending_idxs)} records...")

        messages = [generate_rl_step_prompt(records[i]['context']) for i in pending_idxs]
        outputs = [o.outputs[0].text for o in ckpt.chat(messages, sampling_params)]
        vals = [split_response(out) for out in outputs]

        queries_to_retrieve = []
        for i, val in zip(pending_idxs, vals):
            records[i]['history'].append({f'turn_{turn+1}_output': val['original']})
            if val['answer']:
                records[i]['answer'] = val['answer']
                records[i]['state'] = 'done'
            elif val['query']:
                queries_to_retrieve.append({'record_idx': i, 'query': val['query']})
                step_log = f"Step {turn + 1}:\nThe problem analysis: {val['analysis']}\nThe retrieval query: {val['query']}"
                records[i]['context'] += f"\n{step_log}"
            else: # 잘못된 포맷
                records[i]['state'] = 'failed_format'

        if not queries_to_retrieve:
            continue
        
        # 문서 검색
        retrieved_docs = get_retrieval(cfg.urls.retrieve_url, [q['query'] for q in queries_to_retrieve])
        
        for item, docs in zip(queries_to_retrieve, retrieved_docs):
            idx = item['record_idx']
            doc_contents = [d['contents'] for d in docs[:cfg.rl_params.num_passages_one_retrieval]]
            doc_str = "\n".join(doc_contents)
            records[idx]['docs_retrieved'].append(doc_contents)
            records[idx]['context'] += f"\nThe retrieval documents: {doc_str}"
            # 디버깅을 위해 첫번째 레코드의 context 출력
            if idx == pending_idxs[0]:
                print("\n--- Context Update Example ---")
                print(records[idx]['context'])
                print("----------------------------\n")
    
    # 마지막 턴 이후에도 pending 상태인 경우 fail 처리
    for rec in records:
        if rec['state'] == 'pending':
            rec['state'] = 'failed_iterative'

    # 2. Fallback Pass (Direct with retrieved docs)
    print("\n--- RL Mode: Fallback Pass (Direct) ---")
    remain_idxs = [i for i, rec in enumerate(records) if 'answer' not in rec]
    if not remain_idxs:
        print("All records solved in the iterative search.")
        return

    # Fallback을 위해 문서가 없는 경우, 원본 질문으로 검색
    queries_for_fallback_retrieval = [
        (i, records[i]['question']) for i in remain_idxs if not records[i]['docs_retrieved']
    ]
    if queries_for_fallback_retrieval:
        print(f"Retrieving documents for {len(queries_for_fallback_retrieval)} records for fallback...")
        fallback_docs = get_retrieval(cfg.urls.retrieve_url, [q for _, q in queries_for_fallback_retrieval])
        for (idx, _), docs in zip(queries_for_fallback_retrieval, fallback_docs):
            doc_contents = [d['contents'] for d in docs[:cfg.rl_params.num_passages_one_retrieval]]
            records[idx]['docs_retrieved'].append(doc_contents)

    remain_messages = []
    for i in remain_idxs:
        all_docs = [doc for turn_docs in records[i]['docs_retrieved'] for doc in turn_docs]
        doc_str = "\n".join(all_docs)
        remain_messages.append(generate_rl_direct_prompt(records[i]['question'], doc_str))

    remain_outputs = [o.outputs[0].text for o in ckpt.chat(remain_messages, sampling_params)]

    for idx, out in zip(remain_idxs, remain_outputs):
        records[idx]['history'].append({'pass': 'direct_fallback', 'output': out})
        val = split_response(out)
        if val['answer']:
            records[idx]['answer'] = val['answer']
            records[idx]['state'] = 'done_direct'
        else:
            records[idx]['answer'] = mystrip(out) # 파싱 실패시 전체 출력 저장
            records[idx]['state'] = 'failed_fallback'


# =====================================================================================
# SECTION 4: MAIN ORCHESTRATOR
# =====================================================================================

def solve(cfg: DictConfig, input_filepath: str):
    """추론 모드를 선택하고 전체 프로세스를 실행"""
    # output_file 이름은 main 함수에서 설정하므로 여기서는 사용만 함
    output_file_name = cfg.output_file 
    
    ckpt, records = solve_init(cfg, input_filepath)
    
    mode_solvers = {
        "naive": solve_naive,
        "rag": solve_rag,
        "rl_iterative": solve_rl_iterative
    }
    
    solver = mode_solvers.get(cfg.mode)
    
    if solver:
        solver(cfg, ckpt, records)
    else:
        raise ValueError(f"Invalid mode '{cfg.mode}'. Choose from {list(mode_solvers.keys())}")

    done_count = sum(1 for r in records if 'answer' in r and r['answer'])
    print("\n--- Benchmark Summary ---")
    print(f"Total records processed: {len(records)}")
    print(f"Successfully answered: {done_count} ({done_count/len(records):.2%})")
    print(f"Failed to answer: {len(records) - done_count}")
    
    # Hydra가 생성한 디렉토리 내에 결과 저장
    hydra_output_dir = os.getcwd()
    output_file_path = os.path.join(hydra_output_dir, output_file_name)
    print(f"💾 Saving records to {output_file_path}")
    with open(output_file_path, "w", encoding='utf-8') as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
    print("✅ Done.")
    # 모델 객체 정리
    del ckpt

# NEW FUNCTION: RAG 데이터 전처리 및 저장
def preprocess_rag_data(cfg: DictConfig):
    """RAG 모드를 위한 데이터 전처리: 문서 검색 후 파일로 저장"""
    input_file = cfg.input_file
    output_file = cfg.rag_params.preprocessed_file
    
    print(f"--- Pre-processing for RAG mode ---")
    print(f"Loading records from: {input_file}")
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
            
    if cfg.debug:
        records = random.sample(records, min(8, len(records)))
        print(f"🐞 Debug mode: Sampled {len(records)} records for pre-processing.")

    all_questions = [rec.get('problem', rec.get('question')) for rec in records]
    retrieved_docs = get_retrieval(cfg.urls.retrieve_url, all_questions)
    
    for rec, docs in zip(records, retrieved_docs):
        doc_contents = [d['contents'] for d in docs[:cfg.rag_params.num_of_docs]]
        rec['doc'] = "\n".join(doc_contents)
    
    print(f"💾 Saving pre-processed RAG data to: {output_file}")
    with open(output_file, "w", encoding='utf-8') as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
    print("✅ RAG pre-processing complete.")
    return output_file


# MODIFIED: main 함수가 전체 실험을 관리하는 Orchestrator 역할 수행
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("--- 🚀 Unified Benchmarking Script Started ---")
    
    # RAG 모드가 실험에 포함된 경우, 먼저 데이터 전처리 수행
    rag_input_file = None
    if "rag" in cfg.benchmark_settings.modes:
        preprocessed_file_path = cfg.rag_params.preprocessed_file
        if not os.path.exists(preprocessed_file_path):
            rag_input_file = preprocess_rag_data(cfg)
        else:
            print(f"🔎 Found existing pre-processed RAG data: {preprocessed_file_path}. Skipping pre-processing.")
            rag_input_file = preprocessed_file_path

    # 정의된 모든 실험 조합에 대해 루프 실행
    for mode in cfg.benchmark_settings.modes:
        
        # 현재 모드에 맞는 입력 파일 선택
        current_input_file = rag_input_file if mode == "rag" else cfg.input_file

        for model_name, model_config in cfg.benchmark_settings.models.items():
            print(f"\n\n{'='*25} RUNNING BENCHMARK {'='*25}")
            print(f"📊 Mode: {mode}, 모델: {model_name}")
            print(f"{'='*72}")
            
            # 현재 실행에 맞는 설정으로 cfg 객체를 동적으로 수정
            run_cfg = cfg.copy()
            run_cfg.mode = mode
            
            # 기본 모델 설정을 특정 모델 설정으로 덮어쓰기
            OmegaConf.update(run_cfg, "model", model_config, merge=True)
            
            # 결과 파일 이름 동적 생성
            run_cfg.output_file = f"results_{mode}_{model_name}.jsonl"
            
            # 수정된 설정으로 벤치마크 실행
            solve(run_cfg, input_filepath=current_input_file)

    print("\n--- 🎉 All benchmark runs completed! ---")


if __name__ == "__main__":
    main()
