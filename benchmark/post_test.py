import requests
import tqdm
import time
from typing import List
from flask import Flask, request, jsonify
import threading
import os
import signal

# ----------------------------------------------------
# 1. 🔥 에러 로깅이 강화된 함수
# ----------------------------------------------------
def get_retrieval(retrieve_url: str, queries: List[str], batch_size: int = 3, retry: int = 3) -> List[str]:
    """배치 처리 및 재시도 로직이 포함된 문서 검색 API 호출 (상세 에러 로깅 추가)"""
    results = []
    for i in tqdm.tqdm(range(0, len(queries), batch_size), desc="Retrieving documents"):
        subset = queries[i:i + batch_size]
        for attempt in range(retry):
            try:
                response = requests.post(
                    retrieve_url,
                    json={"queries": subset},
                    headers={"Content-Type": "application/json"},
                    timeout=10 # 타임아웃을 짧게 조정하여 테스트 용이하게 함
                )
                
                # 🔥 200번대 상태 코드가 아니면 여기서 즉시 HTTPError 발생!
                response.raise_for_status() 
                
                # 성공 시 결과 추가
                results.extend(response.json())
                print(f"Subset {i//batch_size+1} retrieved successfully.")
                break # 성공했으므로 재시도 중단

            except requests.exceptions.HTTPError as e:
                # 🔥 서버가 에러 코드를 응답한 경우 (가장 흔한 실패 원인)
                print(f"\n[ERROR] HTTP Error occurred on attempt {attempt + 1}/{retry}.")
                print(f"  - Status Code: {e.response.status_code}")
                print(f"  - Response Body: {e.response.text}")
                time.sleep(2)
            
            except requests.exceptions.RequestException as e:
                # 🔥 연결 실패, 타임아웃 등 네트워크 레벨 에러
                print(f"\n[ERROR] Request Exception occurred on attempt {attempt + 1}/{retry}.")
                print(f"  - Error Type: {type(e).__name__}")
                print(f"  - Error Details: {e}")
                time.sleep(2)

        else: # for 루프가 break 없이 모두 실행 (모든 재시도 실패)
            print(f"\n[FATAL] Failed to retrieve subset after {retry} attempts: {subset}")

    return results

# ----------------------------------------------------
# 2. 🐛 에러를 시뮬레이션하는 Mock 서버
# ----------------------------------------------------
app = Flask(__name__)

@app.route('/retrieve', methods=['POST'])
def mock_retrieve():
    queries = request.get_json().get("queries", [])
    
    # 🔥 쿼리에 'error'가 포함되어 있으면 500 에러를 강제로 발생시킴
    if any("error" in q for q in queries):
        print(f"\n[Mock Server] Simulating 500 error for queries: {queries}")
        error_response = {"detail": "An intentional error occurred on the server."}
        return jsonify(error_response), 500

    mock_results = [f"Retrieved content for query: '{q}'" for q in queries]
    print(f"\n[Mock Server] Successfully processed queries: {queries}")
    return jsonify(mock_results)

def run_mock_server():
    app.run(host='127.0.0.1', port=5000)

# ----------------------------------------------------
# 3. 실제 테스트 실행 부분
# ----------------------------------------------------
if __name__ == "__main__":
    server_thread = threading.Thread(target=run_mock_server)
    server_thread.daemon = True
    server_thread.start()
    time.sleep(1)

    test_url = "http://10.0.12.120:8001/search_batch"
    sample_queries = [
        "What is Large Language Model?",
        "This query will cause an error.", # 🐛 에러 유발 쿼리
        "How does transformer architecture work?",
        "What are the latest trends in AI?",
        "Who developed the Qwen model?"
    ]

    print("\n--- Starting retrieval test with enhanced error logging ---")
    try:
        retrieved_results = get_retrieval(test_url, sample_queries, batch_size=2) # 배치 사이즈 조정
        print("\n--- Retrieval test finished ---\n[Final Results]:", retrieved_results)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        os.kill(os.getpid(), signal.SIGINT)