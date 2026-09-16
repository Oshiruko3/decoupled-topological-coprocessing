"""
DTC v2.0 Benchmark Runner (Standalone Reproducibility Script)
Author: Kouta Matsumoto (Independent Researcher)

This script provides an OpenAI-compatible streaming client coupled with
Decoupled Topological Coprocessing (DTC) for evaluating LLMs on reasoning
loop traps and mathematical deduction tasks.
"""

import argparse
import json
import os
import re
import sys
import time
from typing import List, Dict, Any, Optional

try:
    import requests
    from sentence_transformers import SentenceTransformer
    import ripser
    import numpy as np
except ImportError:
    pass  # Dependencies checked in main()


class TopologicalCoprocessor:
    """Non-invasive real-time persistent homology coprocessor for token streams."""
    
    def __init__(
        self,
        embed_model_name: str = "all-MiniLM-L6-v2",
        window_size: int = 8,
        min_char_len: int = 12,
        density_threshold: float = 0.12,
        noise_threshold: float = 0.04,
        max_lifetime_threshold: float = 0.08
    ):
        self.embedder = SentenceTransformer(embed_model_name)
        self.window_size = window_size
        self.min_char_len = min_char_len
        self.density_threshold = density_threshold
        self.noise_threshold = noise_threshold
        self.max_lifetime_threshold = max_lifetime_threshold
        
        self.buffer = ""
        self.propositions: List[str] = []
        self.delimiters = re.compile(r'[\.\?\!\n]')

    def reset(self):
        self.buffer = ""
        self.propositions = []

    def feed_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Feed token chunk. Returns anomaly dict if H1 cycle detected, else None."""
        self.buffer += token
        parts = self.delimiters.split(self.buffer)
        if len(parts) > 1:
            for clause in parts[:-1]:
                c = clause.strip()
                if len(c) >= self.min_char_len:
                    self.propositions.append(c)
            self.buffer = parts[-1]

        if len(self.propositions) < self.window_size:
            return None

        # Analyze current sliding window
        window = self.propositions[-self.window_size:]
        embeddings = self.embedder.encode(window, normalize_embeddings=True)
        
        # Compute Vietoris-Rips persistent homology
        res = ripser.ripser(embeddings, maxdim=1)
        h1_diagram = res['dgms'][1]

        if len(h1_diagram) == 0:
            return None

        lifetimes = h1_diagram[:, 1] - h1_diagram[:, 0]
        valid_lifetimes = lifetimes[lifetimes > self.noise_threshold]
        valid_count = len(valid_lifetimes)
        density = valid_count / self.window_size
        max_life = float(np.max(lifetimes)) if len(lifetimes) > 0 else 0.0

        if density >= self.density_threshold or max_life > self.max_lifetime_threshold:
            return {
                "triggered": True,
                "density": density,
                "max_life": max_life,
                "valid_count": valid_count,
                "propositions_count": len(self.propositions),
                "trigger_window": window
            }
        return None


def run_single_eval(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    mode: str,
    coprocessor: Optional[TopologicalCoprocessor]
) -> Dict[str, Any]:
    """Execute a single generation request (Baseline or DTC)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }

    if coprocessor:
        coprocessor.reset()

    generated_text = ""
    start_time = time.time()
    anomaly_event = None
    token_count = 0

    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode('utf-8')
            if line_str.startswith("data: "):
                data_str = line_str[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk_json = json.loads(data_str)
                    delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        generated_text += content
                        token_count += 1
                        
                        if mode == "dtc" and coprocessor:
                            alert = coprocessor.feed_token(content)
                            if alert:
                                anomaly_event = alert
                                response.close()  # Low-Latency Abort (Transport-Layer Close)
                                break
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        if not anomaly_event:
            return {"error": str(e), "generated_text": generated_text, "tokens": token_count}

    elapsed = time.time() - start_time

    # If DTC triggered, execute rollback & anchor injection
    if anomaly_event and mode == "dtc":
        # Rollback k steps
        k = 2
        safe_point = max(1, anomaly_event["propositions_count"] - k)
        safe_prefix = " ".join(coprocessor.propositions[:safe_point])
        anchor = "\n\n[Topological Audit Interrupt: Immediately terminate circular verification. Directly address problem symmetries, paradox structure, or the non-existence of assumed entities.]\n\n"
        steered_prompt = safe_prefix + anchor

        steered_payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": steered_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }

        steered_text = ""
        try:
            r2 = requests.post(url, headers=headers, json=steered_payload, stream=True, timeout=120)
            for line in r2.iter_lines():
                if not line:
                    continue
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    d_str = line_str[6:].strip()
                    if d_str == "[DONE]":
                        break
                    try:
                        c_json = json.loads(d_str)
                        delta = c_json.get("choices", [{}])[0].get("delta", {})
                        c_text = delta.get("content", "")
                        if c_text:
                            steered_text += c_text
                            token_count += 1
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

        return {
            "mode": "dtc",
            "triggered": True,
            "anomaly": anomaly_event,
            "tokens": token_count,
            "elapsed_sec": round(elapsed, 2),
            "generated_text": generated_text,
            "steered_continuation": steered_text
        }

    return {
        "mode": "baseline" if mode == "baseline" else "dtc_pass",
        "triggered": False,
        "tokens": token_count,
        "elapsed_sec": round(elapsed, 2),
        "generated_text": generated_text
    }


def main():
    parser = argparse.ArgumentParser(description="DTC v2.0 Benchmark Evaluation Runner")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1", help="Base URL of inference API")
    parser.add_argument("--api-key", type=str, default="", help="Optional API key")
    parser.add_argument("--model", type=str, required=True, help="Model name / identifier")
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation dataset JSON")
    parser.add_argument("--mode", type=str, choices=["baseline", "dtc"], default="dtc", help="Evaluation mode")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Context generation token limit")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--window-size", type=int, default=8, help="Sliding window size W")
    parser.add_argument("--threshold", type=float, default=0.12, help="H1 density threshold tau")
    parser.add_argument("--output", type=str, default="results.json", help="Path to save evaluation results")
    args = parser.parse_args()

    # Verify dependencies
    for mod in ["requests", "sentence_transformers", "ripser", "numpy"]:
        if mod not in sys.modules:
            print(f"Error: Missing required dependency '{mod}'. Install via pip:")
            print("  pip install requests sentence-transformers ripser numpy")
            sys.exit(1)

    coprocessor = None
    if args.mode == "dtc":
        print(f"Initializing Topological Coprocessor (W={args.window_size}, tau={args.threshold})...")
        coprocessor = TopologicalCoprocessor(
            window_size=args.window_size,
            density_threshold=args.threshold
        )

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} problems from {args.dataset}. Starting evaluation ({args.mode.upper()})...")
    results = []

    for idx, item in enumerate(data, 1):
        q_id = item.get("id") or item.get("problem_index") or f"Q_{idx}"
        prompt = item.get("prompt") or item.get("problem_text") or item.get("question")
        print(f"[{idx}/{len(data)}] Running {q_id}...", end=" ", flush=True)

        res = run_single_eval(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            mode=args.mode,
            coprocessor=coprocessor
        )
        res["id"] = q_id
        results.append(res)
        print(f"Done! ({res.get('tokens', 0)} tok, Triggered: {res.get('triggered', False)})")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll evaluations completed! Results saved to: {args.output}")


if __name__ == "__main__":
    main()
