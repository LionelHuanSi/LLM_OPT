import asyncio
import json
import os
import time
import aiohttp

# Competition Constants
F_TTFT = 100.0   # ms
C_TTFT = 1500.0  # ms
F_TPOT = 20.0    # ms
C_TPOT = 45.0    # ms
GAMMA = 2.0
W = 0.5

def calculate_score_component(val, floor, ceiling, gamma=GAMMA):
    if val <= floor:
        return 1.0
    elif val >= ceiling:
        return 0.0
    else:
        norm = (val - floor) / (ceiling - floor)
        return (1.0 - norm) ** gamma

def compute_ers(ttft_list, tpot_list):
    if not ttft_list or not tpot_list:
        return 0.0, 0.0, 0.0
    
    s_ttft = [calculate_score_component(ttft, F_TTFT, C_TTFT) for ttft in ttft_list]
    s_tpot = [calculate_score_component(tpot, F_TPOT, C_TPOT) for tpot in tpot_list]
    
    avg_s_ttft = sum(s_ttft) / len(s_ttft)
    avg_s_tpot = sum(s_tpot) / len(s_tpot)
    
    ers = W * avg_s_ttft + (1 - W) * avg_s_tpot
    return ers, avg_s_ttft, avg_s_tpot

async def send_request(session, url, request_data, arrival_offset_ms, start_time_base):
    req_id = request_data.get("request_id")
    body = request_data.get("body", {})
    
    # Target delay based on arrival timestamp
    target_time = start_time_base + (arrival_offset_ms / 1000.0)
    now = time.perf_counter()
    sleep_dur = target_time - now
    if sleep_dur > 0:
        await asyncio.sleep(sleep_dur)
        
    req_start_time = time.perf_counter()
    first_token_time = None
    last_token_time = None
    token_count = 0

    headers = {"Content-Type": "application/json"}
    # Force streaming to accurately capture TTFT and TPOT
    payload = dict(body)
    if "VLLM_MODEL_OVERRIDE" in os.environ:
        payload["model"] = os.environ["VLLM_MODEL_OVERRIDE"]
    payload["stream"] = True
    
    try:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                err_text = await response.text()
                print(f"Req {req_id} failed with status {response.status}: {err_text[:100]}")
                return req_id, None, None, False

            async for chunk in response.content:
                chunk_str = chunk.decode("utf-8")
                # Parse server-sent events (SSE)
                lines = chunk_str.split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data_json = json.loads(line[6:])
                            choices = data_json.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    t_curr = time.perf_counter()
                                    if first_token_time is None:
                                        first_token_time = t_curr
                                    last_token_time = t_curr
                                    token_count += 1
                        except Exception:
                            pass
                            
        if first_token_time is None:
            # Fallback if content was empty or non-chunked
            first_token_time = time.perf_counter()
            last_token_time = first_token_time
            token_count = 1

        ttft_ms = (first_token_time - req_start_time) * 1000.0
        if token_count > 1 and last_token_time > first_token_time:
            tpot_ms = ((last_token_time - first_token_time) * 1000.0) / (token_count - 1)
        else:
            tpot_ms = 0.0
            
        return req_id, ttft_ms, tpot_ms, True
        
    except Exception as e:
        print(f"Req {req_id} exception: {e}")
        return req_id, None, None, False

async def run_benchmark(trace_file, endpoint_url):
    with open(trace_file, "r", encoding="utf-8") as f:
        requests = [json.loads(line) for line in f if line.strip()]

    limit = int(os.environ.get("VLLM_LIMIT_REQUESTS", -1))
    if limit > 0:
        requests = requests[:limit]
        print(f"Limiting workload to the first {limit} requests.")

    print(f"Loaded {len(requests)} requests from {trace_file}")
    print(f"Target Endpoint: {endpoint_url}")
    print("Replaying workload with streaming metric capture...")

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        start_time_base = time.perf_counter()
        tasks = []
        for req in requests:
            arrival_offset_ms = req.get("timestamp_ms", 0)
            tasks.append(send_request(session, endpoint_url, req, arrival_offset_ms, start_time_base))
            
        results = await asyncio.gather(*tasks)

    ttfts = []
    tpots = []
    successful = 0

    for req_id, ttft, tpot, ok in results:
        if ok and ttft is not None:
            successful += 1
            ttfts.append(ttft)
            tpots.append(tpot)

    print(f"\n--- BENCHMARK RESULTS ({successful}/{len(requests)} Successful) ---")
    if ttfts:
        avg_ttft = sum(ttfts) / len(ttfts)
        avg_tpot = sum(tpots) / len(tpots)
        ers, s_ttft, s_tpot = compute_ers(ttfts, tpots)
        
        print(f"Average TTFT: {avg_ttft:.2f} ms")
        print(f"Average TPOT: {avg_tpot:.2f} ms")
        print(f"TTFT Sub-Score (S_ttft): {s_ttft:.4f}")
        print(f"TPOT Sub-Score (S_tpot): {s_tpot:.4f}")
        print(f"Overall ERS Score       : {ers:.4f} (Raw Score: {ers * 100:.2f})")
    else:
        print("No successful requests recorded.")

if __name__ == "__main__":
    trace_path = os.path.join(os.path.dirname(__file__), "input", "trace-round1.jsonl")
    url = os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
    asyncio.run(run_benchmark(trace_path, url))
