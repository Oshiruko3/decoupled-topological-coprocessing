import os
import json
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILE_26B = os.path.join(REPO_DIR, "benchmarks", "results", "v3", "dtc_v3_tri_replicate_results.json")
FILE_E2B = os.path.join(REPO_DIR, "benchmarks", "results", "v3", "dtc_v3_e2b_benchmark_results.json")

def analyze_results(filepath: str, model_name: str):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_latencies = []
    scenarios_summary = {}

    for s_id, s_data in data.items():
        s_name = s_data["name"]
        scenarios_summary[s_id] = {
            "name": s_name,
            "tokens_list": [],
            "times_list": [],
            "steps_list": [],
            "actions": [],
            "aborted_count": 0,
            "tokens_saved_estimate": []
        }

        for run in s_data["runs"]:
            tokens = run["total_tokens"]
            time_s = run["total_time_s"]
            steps = run["steps_evaluated"]
            aborted = run["aborted_by_dtc"]
            action = run["final_action"]

            scenarios_summary[s_id]["tokens_list"].append(tokens)
            scenarios_summary[s_id]["times_list"].append(time_s)
            scenarios_summary[s_id]["steps_list"].append(steps)
            scenarios_summary[s_id]["actions"].append(action)
            if aborted:
                scenarios_summary[s_id]["aborted_count"] += 1
                # If aborted, estimated savings against 8192 max_tokens budget
                scenarios_summary[s_id]["tokens_saved_estimate"].append(8192 - tokens)

            for st in run.get("telemetry_steps", []):
                lat = st.get("dtc_latency_ms", 0.0)
                if lat > 0.05: # filter warm-up steps
                    all_latencies.append(lat)

    all_latencies = np.array(all_latencies)

    print(f"\n{'=' * 80}")
    print(f"  QUANTITATIVE ANALYSIS REPORT: {model_name}")
    print(f"{'=' * 80}")

    print("\n--- 1. Real-Time Latency Statistics (Coprocessor Overhead across all steps) ---")
    print(f"  Total Evaluated Steps   : {len(all_latencies)}")
    print(f"  Mean Latency            : {np.mean(all_latencies):.4f} ms")
    print(f"  Median Latency (P50)    : {np.median(all_latencies):.4f} ms")
    print(f"  95th Percentile (P95)   : {np.percentile(all_latencies, 95):.4f} ms")
    print(f"  99th Percentile (P99)   : {np.percentile(all_latencies, 99):.4f} ms")
    print(f"  Max Latency             : {np.max(all_latencies):.4f} ms")
    print(f"  Standard Deviation      : {np.std(all_latencies):.4f} ms")

    print("\n--- 2. Scenario-by-Scenario Resource Consumption & Token ROI ---")
    print(f"{'ID':<4} | {'Scenario Name':<38} | {'Avg Steps':<9} | {'Avg Time':<8} | {'Avg Tokens':<10} | {'Aborts':<6} | {'Token ROI (Saved)'}")
    print("-" * 105)

    for s_id, sm in scenarios_summary.items():
        avg_steps = np.mean(sm["steps_list"])
        avg_time = np.mean(sm["times_list"])
        avg_tok = np.mean(sm["tokens_list"])
        aborts = f"{sm['aborted_count']}/3"
        if sm["tokens_saved_estimate"]:
            avg_saved = np.mean(sm["tokens_saved_estimate"])
            pct_saved = (avg_saved / 8192.0) * 100.0
            roi_str = f"~{avg_saved:.0f} tok ({pct_saved:.1f}%)"
        else:
            roi_str = "N/A (Healthy Passthrough)"

        print(f"{s_id:<4} | {sm['name'][:38]:<38} | {avg_steps:<9.1f} | {avg_time:<7.2f}s | {avg_tok:<10.1f} | {aborts:<6} | {roi_str}")

    return {
        "model": model_name,
        "latency_stats": {
            "steps": len(all_latencies),
            "mean_ms": round(float(np.mean(all_latencies)), 4),
            "median_ms": round(float(np.median(all_latencies)), 4),
            "p95_ms": round(float(np.percentile(all_latencies, 95)), 4),
            "p99_ms": round(float(np.percentile(all_latencies, 99)), 4),
            "max_ms": round(float(np.max(all_latencies)), 4),
            "std_ms": round(float(np.std(all_latencies)), 4)
        },
        "scenarios": scenarios_summary
    }

def main():
    res_26b = analyze_results(FILE_26B, "Gemma 4 26B Q4 (Edge Host / 18.9GB VRAM)")
    res_e2b = analyze_results(FILE_E2B, "Gemma 4 E2B FP16 (Main Server / Unquantized)")

    combined = {
        "26B_Q4": res_26b,
        "E2B_FP16": res_e2b
    }

    out_json = os.path.join(REPO_DIR, "benchmarks", "results", "v3", "dtc_v3_empirical_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Full statistical summary saved to: {out_json}")

if __name__ == "__main__":
    main()
