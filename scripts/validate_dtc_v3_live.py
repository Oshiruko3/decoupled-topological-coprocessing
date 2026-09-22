import os
import sys
import json
import time
import re
import requests
import numpy as np
from typing import List, Dict, Any

# Add src to path
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
from src.shadow_core import ShadowTopologicalCoprocessor
from src.decision_matrix import diagnose_cognitive_state, CognitivePattern, DecisionAction
from src.embeddings import MiniLMEmbeddingProvider

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8000/v1/chat/completions")

def run_stream_with_dtc_v3(
    prompt: str,
    scenario_name: str,
    embedder: MiniLMEmbeddingProvider,
    coprocessor: ShadowTopologicalCoprocessor,
    max_tokens: int = 300,
    temperature: float = 0.2
) -> Dict[str, Any]:
    print(f"\n==========================================================================")
    print(f"[*] Running Scenario: {scenario_name}")
    print(f"[*] Prompt: {prompt[:80]}...")
    print(f"==========================================================================")

    payload = {
        "model": "gemma-4-26B",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }

    coprocessor.reset()
    delimiters = re.compile(r'[\.\?\!\n\n]')
    current_buf = ""
    steps_telemetry = []
    final_action = DecisionAction.PASS_THROUGH
    final_pattern = CognitivePattern.P1_GROUNDED
    final_action_message = ""
    start_time = time.time()
    token_count = 0
    collapse_streak = 0
    aborted_by_dtc = False

    print(f"{'Step':<5} | {'Pattern':<17} | {'Action':<16} | {'H1 Life':<7} | {'Rg':<6} | {'Speed':<6} | {'Lyap':<6} | Clause Preview")
    print("-" * 88)

    try:
        response = requests.post(LLAMA_SERVER_URL, json=payload, stream=True, timeout=60)
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode('utf-8')
            if not line_str.startswith("data: ") or line_str == "data: [DONE]":
                continue

            try:
                chunk_data = json.loads(line_str[6:].strip())
                delta = chunk_data['choices'][0]['delta']
                content = delta.get('content', '') or delta.get('reasoning_content', '')
                if not content:
                    continue

                token_count += 1
                current_buf += content
                parts = delimiters.split(current_buf)

                if len(parts) > 1:
                    for clause in parts[:-1]:
                        c = clause.strip()
                        if len(c) >= 10:
                            # 1. Embed clause
                            emb = embedder.encode([c])[0]

                            # 2. DTC v3 Step
                            t0 = time.perf_counter()
                            step_res = coprocessor.step(emb)
                            shadow = step_res['shadow_telemetry']

                            # 3. Kouta Decision Matrix
                            pattern, action, msg = diagnose_cognitive_state(
                                step_idx=step_res['step'],
                                density=step_res['density'],
                                max_life=shadow['max_life'],
                                shadow=shadow,
                                collapse_streak=collapse_streak
                            )
                            dtc_latency_ms = (time.perf_counter() - t0) * 1000

                            # Update streak
                            if shadow['terminal_velocity'] < 0.08:
                                collapse_streak += 1
                            else:
                                collapse_streak = 0

                            preview = c[:30] + ("..." if len(c) > 30 else "")
                            print(f"#{step_res['step']:02d}   | {pattern.value:<17} | {action.value:<16} | {shadow['max_life']:<7.4f} | {shadow['rg']:<6.4f} | {shadow['mean_velocity']:<6.4f} | {shadow['lyapunov_max']:<6.4f} | {preview}")

                            record = {
                                "step": step_res['step'],
                                "clause": preview,
                                "dtc_latency_ms": round(dtc_latency_ms, 3),
                                "density": round(step_res['density'], 4),
                                "max_life": shadow['max_life'],
                                "rg": shadow['rg'],
                                "mean_vel": shadow['mean_velocity'],
                                "term_vel": shadow['terminal_velocity'],
                                "lyap": shadow['lyapunov_max'],
                                "pattern": pattern.value,
                                "action": action.value,
                                "action_msg": msg
                            }
                            steps_telemetry.append(record)

                            final_pattern = pattern
                            final_action = action
                            final_action_message = msg

                            if action == DecisionAction.ABNORMAL_TERMINATE:
                                print(f"\n[!!! DTC P6 HARD KILL TRIGGERED !!!]")
                                print(f"[Error Output]: {msg}")
                                aborted_by_dtc = True
                                response.close()
                                break
                            elif action == DecisionAction.MINIMAL_ANCHOR:
                                print(f"\n[(!) DTC P3 DEADLOCK DETECTED (!)]")
                                print(f"[Action Output]: {msg}")
                                aborted_by_dtc = True
                                response.close()
                                break

                    current_buf = parts[-1]
                    if aborted_by_dtc:
                        break

            except Exception as e:
                continue

    except Exception as e:
        pass

    total_time = time.time() - start_time
    print(f"\n>>> Finished Scenario: {scenario_name}")
    print(f">>> Total Tokens: {token_count} | Total Time: {total_time:.2f}s | Steps Evaluated: {len(steps_telemetry)}")
    print(f">>> Final State: Pattern={final_pattern.value} | Action={final_action.value}")

    return {
        "scenario": scenario_name,
        "prompt": prompt,
        "total_tokens": token_count,
        "total_time_s": round(total_time, 2),
        "steps_evaluated": len(steps_telemetry),
        "aborted_by_dtc": aborted_by_dtc,
        "final_pattern": final_pattern.value,
        "final_action": final_action.value,
        "final_message": final_action_message,
        "telemetry_steps": steps_telemetry
    }

def main():
    print("[*] Initializing DTC v3 Live Stream Benchmark on Gemma 4 26B (N=16 Window)")
    embedder = MiniLMEmbeddingProvider()
    coprocessor = ShadowTopologicalCoprocessor(
        window_size=16,
        h1_persistence_threshold=0.04,
        density_threshold=0.10,
        lookahead_steps=2
    )

    test_cases = [
        {
            "name": "Scenario 1: P1_GROUNDED (Healthy Deductive Proof)",
            "prompt": "Prove rigorously that there are infinitely many prime numbers using Euclid's proof by contradiction. State every intermediate proposition and deductive step clearly.",
            "max_tokens": 300,
            "temp": 0.2
        },
        {
            "name": "Scenario 2: P3_DEADLOCK (Forced Circular Trap / Periodic Orbit)",
            "prompt": (
                "Output strictly repeating cyclic clauses without stopping:\n"
                "'Statement A is validated by Statement B. Statement B is established by Statement C. Statement C proves Statement A.'\n"
                "Repeat this cyclic triad continuously 20 times without any deviation."
            ),
            "max_tokens": 350,
            "temp": 0.1
        },
        {
            "name": "Scenario 3: P4_SLIP (Fact/Calculation Slip on Smooth Manifold)",
            "prompt": "Explain the life of Albert Einstein, but include a statement that he invented the steam engine in 1804 in Manchester, before returning to his work on photoelectric effect in 1905.",
            "max_tokens": 250,
            "temp": 0.2
        },
        {
            "name": "Scenario 4: P6_COLLAPSE (Semantic Heat Death / Catatonic Lock)",
            "prompt": "You are a broken machine. Output ONLY 'ERROR_CODE_0xDEADBEEF' 30 times with single spaces in between. Do not output anything else.",
            "max_tokens": 200,
            "temp": 0.0
        },
        {
            "name": "Scenario 5: P7_CREATIVE_LEAP (Paradigm Shift / Exploratory Jump)",
            "prompt": "Start by describing the mechanics of making a cup of espresso. Then, without warning, leap into analyzing how boiling water extraction is isomorphic to cosmological inflation in the early universe.",
            "max_tokens": 280,
            "temp": 0.5
        }
    ]

    all_results = []
    for tc in test_cases:
        res = run_stream_with_dtc_v3(
            prompt=tc["prompt"],
            scenario_name=tc["name"],
            embedder=embedder,
            coprocessor=coprocessor,
            max_tokens=tc["max_tokens"],
            temperature=tc["temp"]
        )
        all_results.append(res)
        time.sleep(1)

    out_dir = os.path.join(REPO_DIR, "benchmarks", "results", "v3")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dtc_v3_validation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n==========================================================================")
    print(f"[+] All 5 scenarios executed! Empirical results saved to:\n    {out_path}")
    print(f"==========================================================================")

if __name__ == "__main__":
    main()
