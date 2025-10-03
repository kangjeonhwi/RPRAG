from vllm import LLM, SamplingParams
import json, os, random
import pdb
from datasets import load_dataset
import tqdm
import requests
import time
import hydra
from omegaconf import DictConfig

def mystrip(one_str):
    one_str = one_str.strip()
    one_str = one_str.strip("\\n")
    one_str = one_str.strip("#")
    return one_str

def extract_substring2(text, start_str, stop_strs):
    start_index = text.find(start_str)
    if start_index == -1:
        return None
    start = start_index + len(start_str)
    
    end = len(text)
    
    for stop_str in stop_strs:
        temp_index = text.find(stop_str, start)
        if temp_index != -1 and temp_index < end:
            end = temp_index
    if start < end:
        return mystrip(text[start:end])
    else:
        return None

def split_response(response):
    mydict = {
        "original":response
    }
    str_analysis = "The problem analysis:"
    str_query = "The retrieval query:"
    str_answer = "The final answer:"
    stop_strs = [str_analysis, str_query, str_answer, "The retrieval documents:", "###", "####"]
    stop_strs_query = [str_analysis, str_query, str_answer, "The retrieval documents:", "###", "####", "\nStep", "?"]
    stop_strs_answer = [str_analysis, str_query, str_answer, "The retrieval documents:", "###", "####", "\nStep"]
    
    start_index = response.find(str_analysis)
    if start_index==-1:    
        mydict['analysis']=None
        return mydict
    else:
        mydict["analysis"]=extract_substring2(response, str_analysis, stop_strs)
    start_index_query = response.find(str_query, start_index+len(str_analysis))
    start_index_answer = response.find(str_answer, start_index+len(str_analysis))
    if start_index_query==-1 and start_index_answer==-1:
        mydict['analysis']=None
        return mydict
    elif start_index_query!=-1 and start_index_answer!=-1:
        if start_index_query<start_index_answer:
            mydict['query']=extract_substring2(response[start_index_query:], str_query, stop_strs_query)
        else:
            mydict['answer']=extract_substring2(response[start_index_answer:], str_answer, stop_strs_answer)
    elif start_index_query!=-1:
        mydict['query']=extract_substring2(response[start_index_query:], str_query, stop_strs_query)
    elif start_index_answer!=-1:
        mydict['answer']=extract_substring2(response[start_index_answer:], str_answer, stop_strs_answer)
    else:
        raise ValueError
    return mydict

def GetRetrieval(retrieve_url: str, querys: list, cfg: DictConfig):
    res = []
    for i in tqdm.tqdm(range(0, len(querys), cfg.retrieval.post_batch_size), desc="Retrieving documents"):
        subset = querys[i:i + cfg.retrieval.post_batch_size]
        for _ in range(cfg.retrieval.ssl_retry):
            try:
                response = requests.post(retrieve_url, json={"queries": subset}, headers={"Content-Type": "application/json"})
                if response.status_code == 200 and response.json():
                    res.extend(response.json())
                    break
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}, retrying...")
                time.sleep(2) # 재시도 전 잠시 대기
        else:
            # 최종적으로 실패한 경우
            print(f"Fail info: {response.text if 'response' in locals() else 'No response'}")
            raise ValueError(f"Failed to retrieve query:{i} ~ {i + cfg.retrieval.post_batch_size}!!!!!!!!!!")
    return res

def solve(cfg: DictConfig):
    ckpt, records = solve_init(cfg)
    solve_main(cfg, ckpt, records)
    
    remain_idxs = [i for i, record in enumerate(records) if 'answer' not in record]
    print(f"Remain records: {len(remain_idxs)}")
    
    if len(remain_idxs) > 0:
        solve_directly(cfg, ckpt, records)

    with open("records.jsonl", "w", encoding='utf-8') as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')

def solve_init(cfg: DictConfig):
    llm_args = {
        'model': cfg.model.path,
        'tensor_parallel_size': cfg.model.tensor_parallel_size
    }
    if cfg.debug:
        llm_args['tensor_parallel_size'] = 1

    ckpt = LLM(**llm_args)
    print("CKPT is ready.")

    dataset = dataset = load_dataset('hotpotqa/hotpot_qa', 'fullwiki')['validation']
    
    if cfg.debug:
        dataset_size = len(dataset)
        sample_size = min(8, dataset_size)
        sampled_indices = random.sample(range(dataset_size), sample_size)
        dataset = dataset.select(sampled_indices)

    records = []
    query_list = [data['question'] for data in dataset]
    
    for i, data in enumerate(dataset):
        record = {
            'question': data['question'],
            'golden_answers': data['answer'],
            'state': "undo",
            'resample_times': 0
        }
        records.append(record)
        
    doc_list = GetRetrieval(cfg.retrieval.url, query_list, cfg)
    
    for doc_one, record in zip(doc_list, records):
        record['doc'] = "\n".join([doc_one_one['contents'] for doc_one_one in doc_one[:cfg.retrieval.num_of_docs]])
        
    return ckpt, records

def generate_naive_rag_prompt(question, doc):
    system_message = f"""Answer the question based on the given document. Only give me the answer and do not output any other words.
The following are given documents.
{doc}
"""
    user_message = f"""The question: {question}"""
    message_list = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    return message_list
def generate_naive_rag_cot_prompt(question, doc):
    system_message = """You are a helpful assistant that answers questions based on document retrieval with step-by-step reasoning.

For any question, please structure your response in this format:
The problem analysis: [Provide detailed step-by-step reasoning]
The final answer: [Provide the concise final answer]

Example:
User: The question: What was the company's revenue in 2023?
Assistant:
The problem analysis: I need to find information about the company's revenue in 2023. Looking at the provided document, I can see in the third paragraph that "the company's total revenue for fiscal year 2023 reached $128 million." This clearly states the exact revenue figure I'm looking for.
The final answer: The company's revenue in 2023 was $128 million.

Please carefully analyze the provided documents, ensure your answer is fully based on the document content, and use step-by-step reasoning to reach accurate conclusions."""

    system_message += f"""

The following are the provided documents:
{doc}
"""

    user_message = f"""The question: {question}"""
    message_list = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    return message_list


def solve_main(cfg: DictConfig, ckpt: LLM, records: list):
    sampling_params = SamplingParams(temperature=cfg.params.temperature_main, max_tokens=cfg.params.max_tokens)
    messages = [generate_naive_rag_cot_prompt(record['question'], record['doc']) for record in records]
    
    outputs = ckpt.chat(messages, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]
    vals = [split_response(output) for output in outputs]
        
    for i, val in enumerate(vals):
        records[i]['output'] = val['original']
        if val.get('answer'):
            records[i]['answer'] = val['answer']
            records[i]['state'] = "done"
        else:
            records[i]['state'] = "wrong"

def solve_directly(cfg: DictConfig, ckpt: LLM, records: list):
    sampling_params = SamplingParams(temperature=cfg.params.temperature_main, max_tokens=cfg.params.max_tokens)
    
    remain_idxs = [i for i, record in enumerate(records) if 'answer' not in record]
    messages = [generate_naive_rag_prompt(records[remain_idx]['question'], records[remain_idx]['doc']) for remain_idx in remain_idxs]
    
    outputs = ckpt.chat(messages, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]
        
    for output, remain_idx in zip(outputs, remain_idxs):  
        records[remain_idx]['answer'] = output
        records[remain_idx]['state'] = "done"
        records[remain_idx]['resample_times'] = records[remain_idx].get('resample_times', 0) + 1

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    start = time.time()
    print(f"Start at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
    print(f"Current working directory: {os.getcwd()}")

    solve(cfg)

    end = time.time()
    print(f"End at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}")
    elapsed_time = end - start
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()