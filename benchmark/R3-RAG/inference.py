import tqdm
import requests, re, random
from vllm import LLM, SamplingParams
import json, os
from datasets import load_dataset
import hydra
from omegaconf import DictConfig, OmegaConf
import time
import pdb

# --- 기존 유틸리티 함수 (변경 없음) ---
POST_BATCH_SIZE = 2048
SSL_RETRY = 8

def split_by_question_mark(text):
    if not isinstance(text, str):
        return None
    if not text:
        return ['']
    result = re.findall(r'[^?]*\?', text)
    return result if result else [text]

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

def split_answer(response_text):
    result = {}
    lines = response_text.strip().split('\n')
    for line in lines:
        if line.startswith("Correctness analysis:"):
            result['Correctness analysis'] = line.replace("Correctness analysis:", "").strip()
        elif line.startswith("Final answer:"):
            result['Final answer'] = line.replace("", "").strip()
    return result

# --- API 호출 함수 (args -> cfg 로 변경) ---
def split_query_remote(split_url, querys : list):
    res = []
    for i in tqdm.tqdm(range(0, len(querys), POST_BATCH_SIZE)):
        subset = querys[i:i+POST_BATCH_SIZE]
        for _ in range(SSL_RETRY):
            response = requests.post(split_url, json={"querys":subset}, headers={"Content-Type": "application/json"})
            if response.status_code == 200 and response.json()["response"]:
                res.extend(response.json()["response"])
                break
        else:
            res.extend([[query] for query in subset])
            print(f"Fail info: {response.text}")
            print(f"Failed to split query:{i} ~ {i + POST_BATCH_SIZE}!!!!!!!!!!")
    return res

def GetRetrieval(retrieve_url, querys):
    res = []
    for i in tqdm.tqdm(range(0, len(querys), POST_BATCH_SIZE)):
        subset = querys[i:i+POST_BATCH_SIZE]
        for _ in range(SSL_RETRY):
            response = requests.post(retrieve_url, json={"querys": subset}, headers={"Content-Type": "application/json"})
            if response.status_code == 200 and response.json():
                res.extend(response.json())
                break
        else:
            print(f"Fail info: {response.text}")
            raise ValueError(f"Failed to retrieve query:{i} ~ {i + POST_BATCH_SIZE}!!!!!!!!!!")
    return res

# --- 핵심 로직 함수 (args -> cfg 로 변경) ---
def solve_init(cfg: DictConfig):
    ckpt = LLM(
        model=cfg.paths.model_path, 
        tensor_parallel_size=cfg.model.tensor_parallel_size,
        max_seq_len_to_capture=cfg.model.max_seq_len_to_capture
    )
    print("ckpt is ready.")

    records = []
    for dataset_name in cfg.experiment.datasets:
        data_path = os.path.join(cfg.paths.dataset_dir, dataset_name, 'dev.jsonl')
        datas = load_dataset('json', data_files=data_path)['train'].to_dict()
        datas = [
            {k : datas[k][i] for k in datas.keys()}
            for i in range(len(datas[list(datas.keys())[0]])) 
        ]
        if cfg.debug:
            datas = random.sample(datas, 8)
        
        for split in [False]: # This part can be further configured in YAML if needed
            for num_passages_one_retrieval in [cfg.experiment.num_of_docs]:
                for num_passages_one_split_retrieval in ([6, 8] if split == True else [num_passages_one_retrieval]):
                    records.extend([
                        {
                            'dataset' : dataset_name,
                            'problem' : data['question'],
                            'golden_answers' : data['golden_answers'],
                            'num_passages_one_split_retrieval' : num_passages_one_split_retrieval,
                            'num_passages_one_retrieval' : num_passages_one_retrieval,
                            'split' : split,
                            'context' : f"The question: {data['question']}",
                            'split_querys' : [],
                            "docs":[],
                            "state": "undo",
                        }
                        for data in datas
                    ])

    # Hydra's output directory is the new log_dir. We save directly to it.
    output_filename = 'records_init.jsonl'
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"✅ Successfully saved {len(records)} initial records to '{os.path.join(os.getcwd(), output_filename)}'")
    except IOError as e:
        print(f"🔥 Error saving file: {e}")
    
    return ckpt, records

def solve_main(cfg: DictConfig, ckpt, records, temperature=0):
    sampling_params = SamplingParams(temperature=temperature, stop_token_ids=[cfg.model.stop_token_id], max_tokens=512)
    for turn in range(cfg.experiment.num_search_one_attempt):
        remain_idxs = [i for i, record in enumerate(records) if record['state'] == 'undo']

        print(f"Turn {turn + 1}**************************************************")
        print(f"Remain records: {len(remain_idxs)}")
        if len(remain_idxs) == 0:
            break
        
        messages = [
            [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": records[i]['context']}
            ] for i in remain_idxs
        ]
        outputs = ckpt.chat(messages, sampling_params)
        outputs = [output.outputs[0].text for output in outputs]
        vals = [split_response(output) for output in outputs]

        # ... (rest of the logic is the same, just replacing args with cfg)
        print(f"Turn{turn+ 1} : Split querys*********************************")
        split_querys_list = []
        for val, i in zip(vals, remain_idxs):
            if val.get('query'):
                if records[i]['split']:
                    if val.get('query') and val.get('query').count('?') >= 2:
                        split_querys = split_by_question_mark(val.get('query'))
                        records[i]['split_querys'].append(split_querys)
                    else:
                        split_querys_list.append((i, val['query']))
                else:
                    records[i]['split_querys'].append([val['query']])
        
        split_outputs = split_query_remote(cfg.urls.split_url, [query for _, query in split_querys_list])
        for output, (i, query) in zip(split_outputs, split_querys_list):
            records[i]['split_querys'].append(output + [query])

        print(f"Turn{turn+ 1} : Retrieve Docs*********************************")
        retrive_list = []
        for val, i in zip(vals, remain_idxs):
            if val.get('query'):
                for query in records[i]['split_querys'][-1]:
                    retrive_list.append((i, query))
        
        doc_list = GetRetrieval(cfg.urls.retrieve_url, [query for _, query in retrive_list])
        id_set = {}
        doc_dict = {}
        for doc, (i, query) in zip(doc_list, retrive_list):
            id_set[i] = id_set.get(i, set()) | set(d['id'] for d in doc[:records[i]['num_passages_one_retrieval']])
            for d in doc[:records[i]['num_passages_one_retrieval']]:
                doc_dict[d['id']] = d['contents']
        
        id_set = {i: random.sample(list(ids), min(len(ids), records[i]['num_passages_one_split_retrieval'])) for i, ids in id_set.items()}
        for i, ids in id_set.items():
            records[i]['docs'].append([doc_dict[id] for id in ids])
        
        print(f"Turn{turn+ 1} : Update context*********************************")
        for val, i in zip(vals, remain_idxs):
            if 'analysis' not in val:
                val['analysis'] = ''
                print("No analysis!!!")
                records[i]['state'] = 'wrong'

            if val.get('answer'):
                answer = val.get('answer')
                step = f"Step {turn + 1}:\nThe problem analysis: {val['analysis']}\nThe final answer: {val['answer']}"
                records[i]['context'] += "\n" + step
                records[i]['answer'] = answer
                records[i]['turn'] = turn
                records[i]['state'] = 'done'
            elif val.get('query'):
                step = f"Step {turn + 1}:\nThe problem analysis: {val['analysis']}\nThe retrieval query: {val['query']}"
                doc_str = "\n".join(records[i]['docs'][-1])
                records[i]['context'] += "\n" + step + "\n" + f"The retrieval documents: {doc_str}"
                if turn == (cfg.experiment.num_search_one_attempt - 1):
                    records[i]['state'] = 'fail'
            else:
                print("Format error!!!")
                records[i]['state'] = 'wrong'
                continue
            
            if i == remain_idxs[0]:
                print(records[i]['context'])


def re_init_no_answer_records(records):
    # This function is fine, no args/cfg dependency
    for record in records:
        if 'answer' not in record:
            dict_tmp = {
                'state': 'undo',
                'dataset': record['dataset'],
                'resample_times': record.get('resample_times', 0) + 1,
                'problem': record['problem'],
                'golden_answers': record['golden_answers'],
                'num_passages_one_split_retrieval': record['num_passages_one_split_retrieval'],
                'num_passages_one_retrieval': record['num_passages_one_retrieval'],
                'split': record['split'],
                'context': f"The question: {record['problem']}",
                'split_querys': [],
                "docs": [],
            }
            record.clear()
            record.update(dict_tmp)

def get_answer_directly(cfg: DictConfig, ckpt, records, temperature=0):
    remain_idxs = [i for i, record in enumerate(records) if 'answer' not in record]
    print(f"Remain records: {len(remain_idxs)}")
    if len(remain_idxs) == 0:
        return
    
    remain_no_doc_idxs_and_query = [(i, records[i]['problem']) for i in remain_idxs if not records[i]['docs']]
    doc_list = GetRetrieval(cfg.urls.retrieve_url, [query for _, query in remain_no_doc_idxs_and_query])
    for doc, (i, query) in zip(doc_list, remain_no_doc_idxs_and_query):
        records[i]['docs'].append([d['contents'] for d in doc[:records[i]['num_passages_one_retrieval']]])

    messags_list = [
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": f"Please directly and briefly give the final answer of the following question only one step based on the given documents. Don't give too much analysis. Please give the final answer. The retrieval documents: {records[i]['docs'][0:10]}. The question: {records[i]['problem']}"}
        ] for i in remain_idxs
    ]
    
    sampling_params = SamplingParams(temperature=temperature, stop_token_ids=[cfg.model.stop_token_id], max_tokens=512)
    outputs = ckpt.chat(messags_list, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]
    vals = [split_response(output) for output in outputs]
    final_remain_idxs = []
    
    for val, i in zip(vals, remain_idxs):
        records[i]['resample_times'] = records[i].get('resample_times', 0) + 1
        if 'answer' in val:
            records[i]['answer'] = val['answer']
            records[i]['turn'] = 1
            records[i]['state'] = "done"
        else:
            records[i]['state'] = "unsolvable"
            final_remain_idxs.append(i)
    
    print(f"Final remain problems: {len(final_remain_idxs)}")

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    
    # solve 함수를 main 함수처럼 사용
    try:
        ckpt, records = solve_init(cfg)
        solve_main(cfg, ckpt, records, temperature=0)
        with open("records_final.jsonl", "w", encoding='utf-8') as f:
            for record in records:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        print(f"✅ Final records saved to '{os.path.join(os.getcwd(), 'records_final.jsonl')}'")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

if __name__ == "__main__":
    start = time.time()
    print(f"Start at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
    main()
    end = time.time()
    print(f"End at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}")
    print(f"Total execution time: {end - start:.2f} seconds")