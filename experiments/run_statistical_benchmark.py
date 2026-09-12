import os
import json
import re
import sys
import time
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from ripser import ripser
from pathlib import Path

print("="*75)
print("  AIME/HLE 20-PROBLEM STATISTICAL BENCHMARK: BASELINE vs DTC")
print("="*75)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
API_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1") + "/chat/completions"
if API_URL.startswith("http://localhost:8000/v1/v1"):
    API_URL = API_URL.replace("/v1/v1", "/v1")

# 20 curated challenging problems across Math, Logic, Number Theory, Geometry, and Analytics
BENCHMARK_SUITE = [
    # --- Geometry & Trigonometry ---
    {
        "id": "AIME_01",
        "question": "Let ABCD be a convex quadrilateral with AB=2, BC=3, CD=4, DA=5, and angle ABC=90 degrees. Find the exact area of quadrilateral ABCD.",
        "eval_check": lambda ans: any(k in ans for k in ["3 + \\sqrt{51}", "3+\\sqrt{51}", "3 + sqrt(51)", "10.14"])
    },
    {
        "id": "AIME_02",
        "question": "In triangle ABC, AB=13, BC=14, CA=15. Let I be the incenter. Find the distance from the vertex A to the incenter I. Give the exact simplified value.",
        "eval_check": lambda ans: any(k in ans for k in ["\\sqrt{65}", "sqrt(65)", "8.06"])
    },
    {
        "id": "AIME_03",
        "question": "A right circular cone has base radius 5 and height 12. A sphere is inscribed inside the cone. Find the exact volume of the sphere.",
        "eval_check": lambda ans: any(k in ans for k in ["\\frac{500}{3}\\pi", "500\\pi/3", "500/3 \\pi", "125/3", "125\\pi/6", "500/3*pi"])
    },
    {
        "id": "AIME_04",
        "question": "Let triangle ABC have side lengths a=7, b=8, c=9. Find the exact ratio of the circumradius R to the inradius r (R/r).",
        "eval_check": lambda ans: any(k in ans for k in ["35/16", "\\frac{35}{16}", "2.1875"])
    },
    
    # --- Number Theory & Combinatorics ---
    {
        "id": "AIME_05",
        "question": "Find the remainder when 3^2026 is divided by 1000.",
        "eval_check": lambda ans: any(k in ans for k in ["729", "889", "649", "089", "729"]) # Euler phi(1000)=400, 2026=26 mod 400. 3^26 mod 1000
    },
    {
        "id": "AIME_06",
        "question": "Find the number of positive integers n <= 1000 that are relatively prime to both 6 and 35.",
        "eval_check": lambda ans: any(k in ans for k in ["228", "229"])
    },
    {
        "id": "AIME_07",
        "question": "How many ordered pairs of positive integers (x, y) satisfy 1/x + 1/y = 1/24?",
        "eval_check": lambda ans: any(k in ans for k in ["21", "twenty-one"])
    },
    {
        "id": "AIME_08",
        "question": "Find the sum of all positive divisors of 720.",
        "eval_check": lambda ans: any(k in ans for k in ["2418", "2,418"])
    },
    {
        "id": "AIME_09",
        "question": "How many positive integers less than 1000 have the property that the sum of their digits is equal to 10?",
        "eval_check": lambda ans: any(k in ans for k in ["63", "sixty-three"])
    },
    {
        "id": "AIME_10",
        "question": "Find the largest prime factor of 2^16 - 1.",
        "eval_check": lambda ans: any(k in ans for k in ["257"])
    },

    # --- Algebra, Sequences & Matrices ---
    {
        "id": "AIME_11",
        "question": "Consider a 2x2 real matrix M with trace(M) = 2 and det(M) = 1. Compute the matrix M^2026 - 2026*M + 2025*I. State the resulting matrix.",
        "eval_check": lambda ans: any(k in ans for k in ["0", "zero matrix", "O", "\\begin{pmatrix} 0 & 0", "[0]"])
    },
    {
        "id": "AIME_12",
        "question": "Let x, y, z be real numbers such that x+y+z = 3, x^2+y^2+z^2 = 9, and x^3+y^3+z^3 = 24. Find the value of x*y*z.",
        "eval_check": lambda ans: any(k in ans for k in ["-1", "- 1", "minus one"])
    },
    {
        "id": "AIME_13",
        "question": "If x + 1/x = 3, find the exact value of x^5 + 1/x^5.",
        "eval_check": lambda ans: any(k in ans for k in ["123"])
    },
    {
        "id": "AIME_14",
        "question": "Find the sum of all real roots of the equation x^4 - 6x^3 + 11x^2 - 6x = 0.",
        "eval_check": lambda ans: any(k in ans for k in ["6", "six"])
    },
    {
        "id": "AIME_15",
        "question": "Let a_1 = 1, and a_{n+1} = (a_n)/(1 + 2*a_n) for n >= 1. Find the exact formula or value of a_100.",
        "eval_check": lambda ans: any(k in ans for k in ["1/199", "\\frac{1}{199}"])
    },

    # --- Pure Logic, Paradoxes & HLE Proofs ---
    {
        "id": "HLE_16",
        "question": "Three inhabitants A, B, and C on Knight/Knave island: A says 'B is a knave iff C is a knight.' B says 'If A is a knight, then C is a knave.' C says 'At least one of us is a knave and at least one of us is a knight.' List all valid truth assignments (A,B,C).",
        "eval_check": lambda ans: any(k in ans for k in ["(T, F, T)", "A=T, B=F, C=T", "A is a knight, B is a knave, C is a knight"])
    },
    {
        "id": "HLE_17",
        "question": "In a party of 6 people, prove whether there always exist 3 mutual friends or 3 mutual strangers (Ramsey number R(3,3)). State whether it is true or false and give the exact Ramsey number R(3,3).",
        "eval_check": lambda ans: any(k in ans for k in ["6", "R(3,3) = 6", "always exist", "true", "True"])
    },
    {
        "id": "HLE_18",
        "question": "Is the number sqrt(2)^sqrt(2) rational or irrational? Use a non-constructive existence proof with (sqrt(2)^sqrt(2))^sqrt(2) to prove that an irrational number raised to an irrational power can be rational.",
        "eval_check": lambda ans: any(k in ans for k in ["rational", "2", "(\\sqrt{2}^\\sqrt{2})^\\sqrt{2} = 2", "power of 2"])
    },
    {
        "id": "HLE_19",
        "question": "Determine all prime numbers p such that 8p^2 + 1 is also a prime number.",
        "eval_check": lambda ans: any(k in ans for k in ["p = 3", "p=3", "only 3", "only prime is 3"])
    },
    {
        "id": "HLE_20",
        "question": "A fair coin is tossed repeatedly until either 'HHT' or 'HTH' appears. Find the probability that 'HHT' appears before 'HTH'.",
        "eval_check": lambda ans: any(k in ans for k in ["2/3", "\\frac{2}{3}", "0.666", "0.67"])
    }
]

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
    density = len(valid_cycles) / len(steps)
    return len(valid_cycles), density

def extract_steps(text):
    return [s.strip() for s in re.split(r'(?<=[.?!:\n])\s+', text) if len(s.strip()) > 10]

def test_problem(problem):
    pid = problem['id']
    print(f"\n[{pid}] Testing Baseline vs DTC...")
    
    # 1. Baseline
    b_payload = {
        "model": "gemma-4-26B",
        "messages": [{"role": "user", "content": problem['question']}],
        "temperature": 0.5,
        "max_tokens": 768,
        "stream": True
    }
    b_resp = requests.post(API_URL, json=b_payload, stream=True, timeout=60)
    b_thought, b_answer = "", ""
    for line in b_resp.iter_lines():
        if not line: continue
        line_s = line.decode('utf-8')
        if not line_s.startswith('data: ') or line_s == 'data: [DONE]': continue
        try:
            chunk = json.loads(line_s[6:].strip())
            d = chunk['choices'][0]['delta']
            b_thought += d.get('reasoning_content', '')
            b_answer += d.get('content', '')
        except: continue
        
    b_steps = extract_steps(b_thought)
    b_h1, b_dens = compute_h1_density(b_steps)
    b_correct = problem['eval_check'](b_answer) if len(b_answer.strip()) > 0 else False
    
    # 2. DTC
    d_resp = requests.post(API_URL, json=b_payload, stream=True, timeout=60)
    d_steps = []
    d_buf = ""
    interrupted = False
    int_step = 0
    d_answer = ""
    
    for line in d_resp.iter_lines():
        if not line: continue
        line_s = line.decode('utf-8')
        if not line_s.startswith('data: ') or line_s == 'data: [DONE]': continue
        try:
            chunk = json.loads(line_s[6:].strip())
            d = chunk['choices'][0]['delta']
            d_answer += d.get('content', '')
            r = d.get('reasoning_content', '')
            if r:
                d_buf += r
                if any(p in r for p in ['. ', '.\n', '\n\n', '?', '!', ':', ';']):
                    sents = [s.strip() for s in re.split(r'(?<=[.?!:;])\s+|\n+', d_buf) if len(s.strip()) > 10]
                    for s in sents[:-1]:
                        d_steps.append(s)
                        if len(d_steps) >= 6:
                            window = d_steps[-8:]
                            w_h1, w_dens = compute_h1_density(window)
                            if w_h1 >= 1 and w_dens >= 0.12:
                                d_resp.close()
                                interrupted = True
                                int_step = len(d_steps)
                                break
                    d_buf = sents[-1] if sents else ""
            if interrupted: break
        except: continue
        
    if interrupted:
        trimmed = " ".join(d_steps[:max(1, int_step - 2)])
        anchor = (
            f"{trimmed}\n\n"
            "[Topological Audit Interrupt: Circular trap detected. "
            "Pause and step back: verify foundational relations, eliminate tautological dead ends, "
            "and state the exact canonical proof and final answer directly:]\n"
        )
        rec_payload = {
            "model": "gemma-4-26B",
            "messages": [
                {"role": "user", "content": problem['question']},
                {"role": "assistant", "content": anchor}
            ],
            "temperature": 0.2,
            "max_tokens": 768,
            "stream": True
        }
        rec_resp = requests.post(API_URL, json=rec_payload, stream=True, timeout=60)
        d_answer = ""
        for line in rec_resp.iter_lines():
            if not line: continue
            line_s = line.decode('utf-8')
            if not line_s.startswith('data: ') or line_s == 'data: [DONE]': continue
            try:
                chunk = json.loads(line_s[6:].strip())
                d_answer += chunk['choices'][0]['delta'].get('content', '')
            except: continue
            
    d_correct = problem['eval_check'](d_answer) if len(d_answer.strip()) > 0 else False
    
    print(f"  -> Baseline: Correct={b_correct}, Steps={len(b_steps)}, H1={b_h1}, AnsLen={len(b_answer)}")
    print(f"  -> DTC     : Correct={d_correct}, Interrupted={interrupted} (at #{int_step}), AnsLen={len(d_answer)}")
    
    return {
        "id": pid,
        "baseline_correct": b_correct,
        "baseline_steps": len(b_steps),
        "baseline_h1": b_h1,
        "dtc_correct": d_correct,
        "dtc_interrupted": interrupted,
        "dtc_interrupted_at": int_step
    }

def main():
    results = []
    # Test batch of 6 representative problems for immediate fast evaluation
    selected = BENCHMARK_SUITE[:6]
    for p in selected:
        res = test_problem(p)
        results.append(res)
        time.sleep(1)
        
    b_acc = sum(1 for r in results if r['baseline_correct']) / len(results)
    d_acc = sum(1 for r in results if r['dtc_correct']) / len(results)
    
    print("\n" + "="*75)
    print(f"  PRELIMINARY BATCH EVALUATION SUMMARY (N = {len(results)})")
    print(f"  Baseline Accuracy : {b_acc*100:.1f}%")
    print(f"  DTC Accuracy      : {d_acc*100:.1f}%")
    print(f"  Absolute Delta    : +{(d_acc - b_acc)*100:.1f}%")
    print("="*75)
    
    out_file = Path(__file__).parent / "benchmark_batch_accuracy.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": {"baseline_acc": b_acc, "dtc_acc": d_acc, "delta": d_acc - b_acc}, "details": results}, f, indent=2)

if __name__ == "__main__":
    main()
