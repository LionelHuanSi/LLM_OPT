import json
import os
import sys

def analyze_trace(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    num_requests = len(lines)
    print(f"Total requests in trace: {num_requests}")

    timestamps = []
    prompt_chars = []
    max_tokens_requested = []
    system_prompts = set()

    for i, line in enumerate(lines):
        req = json.loads(line)
        req_id = req.get("request_id", i)
        ts = req.get("timestamp_ms", 0)
        timestamps.append(ts)

        body = req.get("body", {})
        max_tok = body.get("max_tokens", None)
        if max_tok is not None:
            max_tokens_requested.append(max_tok)

        messages = body.get("messages", [])
        full_text = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_prompts.add(content)
            full_text += f"{role}: {content}\n"
        
        prompt_chars.append(len(full_text))

    # Concurrency analysis based on arrival timestamps
    timestamps.sort()
    min_ts, max_ts = timestamps[0], timestamps[-1]
    duration_s = (max_ts - min_ts) / 1000.0 if max_ts > min_ts else 1.0

    print("\n--- TIMING & BURSTINESS ---")
    print(f"Trace Start Time: {min_ts} ms")
    print(f"Trace End Time  : {max_ts} ms")
    print(f"Trace Duration  : {duration_s:.2f} s")
    print(f"Request Rate    : {num_requests / duration_s:.2f} req/s")

    print("\n--- PROMPT LENGTH ESTIMATION ---")
    min_len = min(prompt_chars)
    max_len = max(prompt_chars)
    avg_len = sum(prompt_chars) / len(prompt_chars)
    print(f"Min Prompt Chars: {min_len}")
    print(f"Max Prompt Chars: {max_len} (~{max_len // 4} tokens)")
    print(f"Avg Prompt Chars: {avg_len:.1f} (~{avg_len / 4:.1f} tokens)")

    if max_tokens_requested:
        print(f"Max Completion Tokens Specified: {max(max_tokens_requested)}")

    print("\n--- PREFIX CACHING ANALYSIS ---")
    print(f"Unique System Prompts: {len(system_prompts)}")
    if len(system_prompts) > 0 and len(system_prompts) <= 5:
        for idx, sys_p in enumerate(system_prompts):
            print(f"  System Prompt #{idx+1} length: {len(sys_p)} chars (~{len(sys_p)//4} tokens)")

    print("\n--- RECOMMENDATIONS FOR vLLM FLAGS ---")
    estimated_max_prompt_tokens = max_len // 3  # Safe upper bound for token count ratio
    suggested_max_model_len = max(2048, ((estimated_max_prompt_tokens + 1024) // 512 + 1) * 512)
    print(f"Suggested --max-model-len: {suggested_max_model_len} (instead of 262144 baseline)")

if __name__ == "__main__":
    trace_file = os.path.join(os.path.dirname(__file__), "input", "trace-round1.jsonl")
    analyze_trace(trace_file)
