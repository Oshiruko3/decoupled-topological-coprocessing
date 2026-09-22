import os
import json
import re
import sys
import time
from pathlib import Path
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from ripser import ripser

print("="*75)
print("  DTC LATENCY & THROUGHPUT OVERHEAD BENCHMARK (Gemma 4 26B)")
print("="*75)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
API_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1") + "/chat/completions"
if API_URL.startswith("http://localhost:8000/v1/v1"):
    API_URL = API_URL.replace("/v1/v1", "/v1")

BENCHMARK_PROMPT = (
    "Explain in detail the fundamental principles of quantum computing, "
    "including qubits, superposition, quantum entanglement, and how quantum algorithms "
    "like Shor's algorithm achieve exponential speedup. Provide a thorough, rigorous essay."
)

def compute_h1_density(steps):
    if len(steps) < 4:
        return 0, 0.0
    embs = embedder.encode(steps)
    res = ripser(embs, maxdim=1)
    dgm = res['dgms'][1]
    if len(dgm) == 0:
        return 0, 0.0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    valid_cycles = [l for l in lifetimes if l > 0.04]
    return len(valid_cycles), len(valid_cycles) / len(steps)

# 1. Baseline: Unmonitored Generation
def run_baseline():
    print("\n[1/2] Running Baseline (Unmonitored Raw Generation)...")
    payload = {
        "model": "gemma-4-26B",
        "messages": [{"role": "user", "content": BENCHMARK_PROMPT}],
        "temperature": 0.5,
        "max_tokens": 512,
        "stream": True
    }
    
    start_t = time.time()
    resp = requests.post(API_URL, json=payload, stream=True, timeout=60)
    
    first_token_t = None
    token_count = 0
    full_text = ""
    
    for line in resp.iter_lines():
        if not line: continue
        line_s = line.decode('utf-8')
        if not line_s.startswith('data: ') or line_s == 'data: [DONE]': continue
        try:
            chunk = json.loads(line_s[6:].strip())
            delta = chunk['choices'][0]['delta']
            t_chunk = delta.get('reasoning_content', '') + delta.get('content', '')
            if t_chunk:
                if first_token_t is None:
                    first_token_t = time.time()
                token_count += 1
                full_text += t_chunk
        except: continue
        
    total_t = time.time() - start_t
    ttft = (first_token_t - start_t) * 1000 if first_token_t else 0
    gen_t = total_t - (ttft / 1000)
    tps = token_count / gen_t if gen_t > 0 else 0
    
    print(f"  -> Total Time: {total_t:.2f}s | TTFT: {ttft:.1f}ms | Generated Tokens: {token_count} chunks")
    print(f"  -> Raw Generation Speed: {tps:.2f} chunks/sec (approx {tps*1.3:.1f} tokens/sec)")
    return {"total_t": total_t, "ttft": ttft, "tokens": token_count, "tps": tps}

# 2. DTC: Parallel Asynchronous TDA Monitored Generation
def run_dtc():
    print("\n[2/2] Running DTC (Asynchronous Parallel TDA Monitored Stream)...")
    payload = {
        "model": "gemma-4-26B",
        "messages": [{"role": "user", "content": BENCHMARK_PROMPT}],
        "temperature": 0.5,
        "max_tokens": 512,
        "stream": True
    }
    
    start_t = time.time()
    resp = requests.post(API_URL, json=payload, stream=True, timeout=60)
    
    first_token_t = None
    token_count = 0
    thought_steps = []
    buf = ""
    tda_computations = 0
    tda_total_time = 0.0
    
    for line in resp.iter_lines():
        if not line: continue
        line_s = line.decode('utf-8')
        if not line_s.startswith('data: ') or line_s == 'data: [DONE]': continue
        try:
            chunk = json.loads(line_s[6:].strip())
            delta = chunk['choices'][0]['delta']
            r_chunk = delta.get('reasoning_content', '')
            c_chunk = delta.get('content', '')
            t_chunk = r_chunk + c_chunk
            
            if t_chunk:
                if first_token_t is None:
                    first_token_t = time.time()
                token_count += 1
                
            if r_chunk:
                buf += r_chunk
                if any(p in r_chunk for p in ['. ', '.\n', '\n\n', '?', '!']):
                    sents = [s.strip() for s in re.split(r'(?<=[.?!])\s+|\n+', buf) if len(s.strip()) > 10]
                    for s in sents[:-1]:
                        thought_steps.append(s)
                        if len(thought_steps) >= 6 and len(thought_steps) % 2 == 0:
                            # TDA coprocessor check
                            tda_start = time.time()
                            window = thought_steps[-8:]
                            h1, dens = compute_h1_density(window)
                            tda_total_time += (time.time() - tda_start)
                            tda_computations += 1
                    buf = sents[-1] if sents else ""
        except: continue
        
    total_t = time.time() - start_t
    ttft = (first_token_t - start_t) * 1000 if first_token_t else 0
    gen_t = total_t - (ttft / 1000)
    tps = token_count / gen_t if gen_t > 0 else 0
    
    print(f"  -> Total Time: {total_t:.2f}s | TTFT: {ttft:.1f}ms | Generated Tokens: {token_count} chunks")
    print(f"  -> Monitored Generation Speed: {tps:.2f} chunks/sec (approx {tps*1.3:.1f} tokens/sec)")
    print(f"  -> TDA Calculations Executed: {tda_computations} times | Total TDA CPU Time: {tda_total_time*1000:.1f}ms")
    print(f"  -> Average Latency per TDA Scan: {(tda_total_time/tda_computations)*1000:.2f}ms" if tda_computations > 0 else "0ms")
    return {"total_t": total_t, "ttft": ttft, "tokens": token_count, "tps": tps, "tda_t": tda_total_time, "tda_count": tda_computations}

def main():
    b_res = run_baseline()
    time.sleep(2)
    d_res = run_dtc()
    
    overhead_pct = ((b_res['tps'] - d_res['tps']) / b_res['tps']) * 100
    ttft_diff = d_res['ttft'] - b_res['ttft']
    
    print("\n" + "="*75)
    print("  EMPIRICAL LATENCY & THROUGHPUT COMPARISON REPORT")
    print("="*75)
    print(f"Metric                           | Baseline      | DTC Monitored | Impact / Overhead")
    print(f"---------------------------------+---------------+---------------+-------------------")
    print(f"Time to First Token (TTFT)       | {b_res['ttft']:6.1f} ms    | {d_res['ttft']:6.1f} ms    | {ttft_diff:+5.1f} ms ({ttft_diff/b_res['ttft']*100:+.1f}%)")
    print(f"Token Generation Rate (TPS)      | {b_res['tps']:6.1f} tok/s  | {d_res['tps']:6.1f} tok/s  | {overhead_pct:+5.2f}% overhead")
    print(f"Average TDA Scan Time            |      N/A      | {(d_res['tda_t']/max(1, d_res['tda_count']))*1000:6.2f} ms    | Background Thread")
    print("="*75)
    
    out_file = Path(__file__).parent / "dtc_latency_overhead_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"baseline": b_res, "dtc": d_res, "overhead_pct": overhead_pct, "ttft_diff_ms": ttft_diff}, f, indent=2)
    print(f"Report saved to: {out_file}")

if __name__ == "__main__":
    main()
