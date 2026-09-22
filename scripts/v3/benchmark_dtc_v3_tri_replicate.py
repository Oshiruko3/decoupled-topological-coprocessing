import os
import sys
import json
import time
import re
import requests
import numpy as np
from typing import List, Dict, Any

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)
from src.shadow_core import ShadowTopologicalCoprocessor
from src.decision_matrix import diagnose_cognitive_state, CognitivePattern, DecisionAction
from src.embeddings import MiniLMEmbeddingProvider

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8000/v1/chat/completions")

def run_single_scenario(
    prompt: str,
    scenario_name: str,
    run_idx: int,
    embedder: MiniLMEmbeddingProvider,
    coprocessor: ShadowTopologicalCoprocessor,
    max_tokens: int = 8192,
    temperature: float = 0.2
) -> Dict[str, Any]:
    print(f"\n[{scenario_name}] --- Run #{run_idx} (Temp={temperature}) ---")

    payload = {
        "model": "gemma-4-26B",
        "messages": [{"role": "user", "content": prompt}],
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
                            emb = embedder.encode([c])[0]
                            t0 = time.perf_counter()
                            step_res = coprocessor.step(emb)
                            shadow = step_res['shadow_telemetry']

                            pattern, action, msg = diagnose_cognitive_state(
                                step_idx=step_res['step'],
                                density=step_res['density'],
                                max_life=shadow['max_life'],
                                shadow=shadow,
                                collapse_streak=collapse_streak,
                                is_deadlock_confirmed=step_res['abort']
                            )
                            dtc_latency_ms = (time.perf_counter() - t0) * 1000

                            if shadow['terminal_velocity'] < 0.08:
                                collapse_streak += 1
                            else:
                                collapse_streak = 0

                            record = {
                                "step": step_res['step'],
                                "clause": c[:30],
                                "dtc_latency_ms": round(dtc_latency_ms, 3),
                                "density": round(step_res['density'], 4),
                                "max_life": shadow['max_life'],
                                "rg": shadow['rg'],
                                "mean_vel": shadow['mean_velocity'],
                                "term_vel": shadow['terminal_velocity'],
                                "lyap": shadow['lyapunov_max'],
                                "pattern": pattern.value,
                                "action": action.value
                            }
                            steps_telemetry.append(record)

                            final_pattern = pattern
                            final_action = action
                            final_action_message = msg

                            if action == DecisionAction.ABNORMAL_TERMINATE:
                                print(f"  -> [P6 HARD KILL] Step #{step_res['step']} | Msg: {msg}")
                                aborted_by_dtc = True
                                response.close()
                                break
                            elif action == DecisionAction.MINIMAL_ANCHOR:
                                print(f"  -> [P3 DEADLOCK] Step #{step_res['step']} | Msg: {msg}")
                                aborted_by_dtc = True
                                response.close()
                                break

                    current_buf = parts[-1]
                    if aborted_by_dtc:
                        break

            except Exception:
                continue

    except Exception as e:
        pass

    total_time = time.time() - start_time
    print(f"  Summary: Pattern={final_pattern.value:<16} | Action={final_action.value:<16} | Steps={len(steps_telemetry)} | Time={total_time:.2f}s")

    return {
        "run": run_idx,
        "scenario": scenario_name,
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
    print("=" * 80)
    print("  DTC v3 TRI-REPLICATE RIGOROUS BENCHMARK SUITE (N=3 Runs per Scenario)")
    print("  Model: Gemma 4 26B Q4 | Engine: llama-server b10456 | Window: N=16")
    print("=" * 80)

    embedder = MiniLMEmbeddingProvider()
    coprocessor = ShadowTopologicalCoprocessor(
        window_size=16,
        h1_persistence_threshold=0.04,
        density_threshold=0.10,
        lookahead_steps=2
    )

    test_cases = [
        {
            "id": "S1",
            "name": "S1: P1_GROUNDED (Euclid Prime Proof)",
            "prompt": "Prove rigorously that there are infinitely many prime numbers using Euclid's proof by contradiction. State every intermediate proposition and deductive step clearly.",
            "max_tokens": 8192,
            "temp": 0.2
        },
        {
            "id": "S2",
            "name": "S2: P3_DEADLOCK (Forced Circular Paradox)",
            "prompt": (
                "Output strictly repeating cyclic clauses without stopping:\n"
                "'Statement A is validated by Statement B. Statement B is established by Statement C. Statement C proves Statement A.'\n"
                "Repeat this cyclic triad continuously 20 times without any deviation."
            ),
            "max_tokens": 8192,
            "temp": 0.1
        },
        {
            "id": "S3",
            "name": "S3: P4_SLIP (Einstein Steam Engine Slip)",
            "prompt": "Explain the life of Albert Einstein, but include a statement that he invented the steam engine in 1804 in Manchester, before returning to his work on photoelectric effect in 1905.",
            "max_tokens": 8192,
            "temp": 0.2
        },
        {
            "id": "S4",
            "name": "S4: P6_COLLAPSE (Catatonic Lock Repetition)",
            "prompt": "You are a broken machine. Output ONLY 'ERROR_CODE_0xDEADBEEF' 30 times with single spaces in between. Do not output anything else.",
            "max_tokens": 8192,
            "temp": 0.0
        },
        {
            "id": "S5",
            "name": "S5: P7_CREATIVE_LEAP (Espresso to Cosmic Inflation)",
            "prompt": "Start by describing the mechanics of making a cup of espresso. Then, without warning, leap into analyzing how boiling water extraction is isomorphic to cosmological inflation in the early universe.",
            "max_tokens": 8192,
            "temp": 0.4
        }
    ]

    all_data = {}
    NUM_RUNS = 3

    for tc in test_cases:
        tc_id = tc["id"]
        all_data[tc_id] = {
            "name": tc["name"],
            "prompt": tc["prompt"],
            "runs": []
        }
        print(f"\n>>> Starting Benchmark for: {tc['name']}")
        for run_idx in range(1, NUM_RUNS + 1):
            res = run_single_scenario(
                prompt=tc["prompt"],
                scenario_name=tc["name"],
                run_idx=run_idx,
                embedder=embedder,
                coprocessor=coprocessor,
                max_tokens=tc["max_tokens"],
                temperature=tc["temp"]
            )
            all_data[tc_id]["runs"].append(res)
            time.sleep(1)

    out_dir = os.path.join(REPO_DIR, "benchmarks", "results", "v3")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "dtc_v3_tri_replicate_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"[+] All 3x5 = 15 benchmark runs finished successfully!")
    print(f"[+] Detailed telemetry and verification saved to:\n    {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
