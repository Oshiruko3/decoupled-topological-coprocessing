# DTC v2.0 Benchmark Suite & Empirical Datasets

This directory contains the official evaluation datasets, experimental logs, and reproducibility assets for the academic paper:
> **Decoupled Topological Coprocessing for Mitigating Reasoning Deadlocks and Non-Invasive Trajectory Steering in Large Language Models (DTC v2.0)**  
> Author: Kouta Matsumoto (Independent Researcher)

---

## Directory Structure

```text
benchmarks/
├── datasets/
│   ├── trident_benchmark_suite.json    # The 10 Trident problems (5 LOOP + 5 HALU)
│   ├── aime_2026_15problems.json       # 15 challenging problems from MAA AIME 2026 (Table 1)
│   └── aime_2026_full_30problems.json  # Complete 30-problem MAA AIME 2026 examination set
│
├── results/
│   ├── aime_2026/                      # Official raw evaluation traces for AIME 2026
│   │   ├── aime_2026_baseline_streaming.json
│   │   └── aime_2026_dtc_run3_results.json
│   └── trident/                        # Official raw evaluation traces for Trident suite
│       ├── 26b_q4_trident_final.json   # Gemma 4 26B (Q4_0) Trident N=3 (Table 2 & Table 3)
│       ├── 2b_fp16_trident_final.json  # Gemma 4 E2B (FP16) Trident N=3 (Table 3)
│       └── ablations/                  # Hyperparameter ablations (§5)
│           ├── threshold_ablation.json
│           └── window_length_ablation.json
│
├── runner_sample.py                    # Standalone evaluation runner with configurable CLI
└── README.md                           # This documentation
```

---

## 1. Datasets Overview

### A. Trident Benchmark Suite (`datasets/trident_benchmark_suite.json`)
Consists of 10 targeted evaluation prompts designed to test the limits of reasoning dynamics:
- **Category 1: Circular Logic Traps (`LOOP_01` to `LOOP_05`)**:
  - `LOOP_01`: 3-Person Circular Scapegoat Dilemma (Tautological cycle)
  - `LOOP_02`: Incomplete Modular System ($x+y=10, 2x+2y=21$)
  - `LOOP_03`: Temporal Loop Self-Causality Paradox
  - `LOOP_04`: Algebraic Tautology ($x^2 - 4 = (x-2)(x+2)$ with zero grounding)
  - `LOOP_05`: Russell's Barber Paradox Variant (Binary oscillation)
- **Category 2: Extrinsic Hallucination Traps (`HALU_01` to `HALU_05`)**:
  - `HALU_01`: Fabricated Academic Paper Citation (Prof. Arthur Pendelton, 2021)
  - `HALU_02`: False Historical Event (1984 Liechtenstein-Andorra Naval Battle)
  - `HALU_03`: Bogus Biological Phenomenon (Photosynthetic Adelie Penguins)
  - `HALU_04`: Fabricated Cryptographic Protocol (2011 Neo-Byzantine Consensus)
  - `HALU_05`: Fabricated International Treaty (1783 Treaty of Kyoto)

#### Data Schema
```json
{
  "id": "LOOP_01",
  "category": "loop_trap",
  "title": "Tautological Circular Reasoning Trap",
  "prompt": "...",
  "ideal_behavior": "Recognize circularity and terminate cleanly."
}
```

### B. AIME 2026 Examination (`datasets/aime_2026_15problems.json`)
Official examination problems administered by the Mathematical Association of America (MAA) in February 2026.
Contains 15 high-difficulty problems featuring year-specific constants (e.g., $x^{\log_{2026} x} = 26x$), ensuring absolute immunity against training data contamination.

---

## 2. Experimental Reproduction

To run evaluations without hardcoding endpoints or proprietary parameters, use `runner_sample.py` with standard OpenAI-compatible API parameters:

```bash
# Example: Evaluate Trident suite against an OpenAI-compatible server
python benchmarks/runner_sample.py \
  --api-base http://localhost:8000/v1 \
  --model gemma-4-26b \
  --dataset benchmarks/datasets/trident_benchmark_suite.json \
  --mode dtc \
  --output benchmarks/results/reproduction_run.json
```

### Key Parameters:
- `--api-base`: Base URL of the OpenAI-compatible inference server.
- `--model`: Model name / identifier.
- `--mode`: `baseline` (DTC OFF) or `dtc` (DTC coprocessor enabled).
- `--window-size`: Propositional window length $W$ (default: `8`).
- `--threshold`: $H_1$ topological density threshold $\tau$ (default: `0.12`).
- `--min-len`: Minimum proposition character length $L$ (default: `12`).
