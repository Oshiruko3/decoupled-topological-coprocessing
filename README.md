# Decoupled Topological Coprocessing (DTC v3.0)
## Sub-Millisecond Intermediate Fork Dynamics and Deterministic Cognitive State Governance for Quantized Reasoning Models

[ **English** ](README.md) | [ **日本語** ](README_ja.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22726133.svg)](https://doi.org/10.5281/zenodo.22726133)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DTC v3.0 Paper: PDF (EN)](https://img.shields.io/badge/DTC%20v3.0%20Paper-PDF%20(EN)-blue.svg)](paper/v3/dtc_v3_paper_en.pdf)
[![DTC v3.0 Paper: PDF (JA)](https://img.shields.io/badge/DTC%20v3.0%20Paper-PDF%20(JA)-red.svg)](paper/v3/dtc_v3_paper_ja.pdf)
[![DTC v2.0 Paper Archive](https://img.shields.io/badge/DTC%20v2.0-Archive-gray.svg)](paper/v2/)
[![DTS Position Paper](https://img.shields.io/badge/DTS%20Paper-Position%20Paper-purple.svg)](paper/dts/)

> **Author**: Kouta Matsumoto (Independent Researcher)  
> Email: `Oshiruko3@users.noreply.github.com`  
> Official DOI: `10.5281/zenodo.22726133` (v2.0 Archive / v3.0 Release)

---

## Overview

**Decoupled Topological Coprocessing v3.0 (DTC v3.0)** is an asynchronous, non-invasive concurrent coprocessing architecture designed to monitor, govern, and stabilize reasoning dynamics in Large Reasoning Models (LRMs) during extended test-time computation.

While prior research (DTC v2.0) established binary detection for circular loops and unanchored hallucinations using 1-dimensional persistent homology ($H_1$), it lacked the kinematic expressiveness required to govern nuanced cognitive states such as deep multi-step deliberation or creative analogical leaps.

DTC v3.0 introduces the **Kinematic Intermediate Fork Paradigm**, which extracts complete trajectory kinematics ($R_g, v, \Delta v, \lambda_{\max}$) directly from the intermediate distance matrix $\mathbf{D} \in \mathbb{R}^{N \times N}$ constructed for Vietoris–Rips simplicial filtration, adding zero additional distance evaluations. Operating in **sub-millisecond latency (mean 0.5858 ms)**, DTC couples these kinematic invariants with the **Topological Cognitive Decision Matrix (P1–P7)** to provide deterministic, comprehensive governance across the entire reasoning spectrum.

```text
[ Host LLM Inference Engine ]
       │ (Streaming token output via SSE / WebSocket)
       ▼
[ Stage 1: Propositional Clause Segmenter ]
       │ (Semantic boundary chunking: punctuation, \n, delimiters / L ≥ 12)
       ▼
[ Stage 2: Lightweight Semantic Embedder ]
       │ (all-MiniLM-L6-v2, d=384, FP16: 768 bytes/step)
       ▼
[ Stage 3: DTC Parallel Coprocessor Kernel ] (0.58 ms)
       ├── Pairwise Distance Matrix D ∈ R^{N×N} (0.03 ms, N=16)
       ├── Kinematic Intermediate Fork (Rg, v_term, Δv_term, λ_max) (0.13 ms)
       └── Simplicial Complex Reduction (Ripser: H1 Persistence) (0.38 ms)
       │
       ▼
[ Stage 4: Topological Cognitive Decision Matrix & Dispatcher ]
       │ (P1–P7 Classification & Hysteresis Streak Confirmation: Streak ≥ 2)
       ▼
[ Control Signal (PASS / OBSERVE / ABNORMAL_TERMINATE) ] ──▶ Host Engine
```

---

## Key Architectural Innovations

1. **The Intermediate Fork Paradigm (0.13 ms Kinematic Extraction)**:
   Demonstrates that the symmetric distance matrix $\mathbf{D}$ already constructed for persistent homology embeds complete trajectory kinematics. By branching directly from $\mathbf{D}$, DTC evaluates the radius of gyration ($R_g$), step velocity ($v$), terminal acceleration ($\Delta v$), and local maximal Lyapunov exponent ($\lambda_{\max}$) in $0.13\ \mathrm{ms}$, avoiding redundant $O(N^2)$ metric projections.
2. **Sub-Millisecond Total Latency & Minimal Bus Overhead**:
   Achieves an average end-to-end execution latency of **0.5858 ms** (99th percentile **0.9744 ms**), contributing $<0.5\%$ overhead relative to clause generation times ($80\sim 120\ \mathrm{ms}$). Host-to-coprocessor interconnect requirements are merely **7.68 KB/sec** ($<0.0001\%$ of standard AXI-4 or PCIe Gen4 bandwidth).
3. **Topological Cognitive Decision Matrix (P1–P7 Taxonomy)**:
   Advances from binary alarm thresholds to an expressive 7-state governance runtime:
   * **P1 (Grounded)**: Progressive, healthy deductive proof $\to$ `PASS_THROUGH`
   * **P2 (Watchlist)**: Localized hesitation or curvature friction $\to$ `OBSERVE`
   * **P3 (Deadlock)**: Topological 1-cycle attractor trap $\to$ `MINIMAL_ANCHOR` (Prefix rollback)
   * **P4 (Slip)**: Arithmetic or factual error (smooth manifold) $\to$ `PASS_THROUGH` (Policy A)
   * **P5 (Delusion)**: Isolated confabulation or detached loop $\to$ `UNIVERSAL_ANCHOR`
   * **P6 (Collapse)**: Catatonic token lock / semantic heat death $\to$ `ABNORMAL_TERMINATE`
   * **P7 (Creative Leap)**: Cross-domain exploratory analogy $\to$ `PASS_THROUGH` (Protected)
4. **Discovery of Granularity Scaling (Zero Alignment Tax)**:
   In unconstrained 8,192-token streaming benchmarks, larger models (26B) traversed up to **94 steps** of fine-grained, multi-perspective self-refutation proofs. DTC validated continuous exploratory dispersion ($R_g = 0.751$, $\lambda_{\max} = +0.042$) and maintained a **0.0% false-positive abort rate**, completely preserving reasoning depth.
5. **98.2% Token & Energy ROI on Adversarial Traps**:
   For cyclic deadlocks (S2) and catatonic collapse (S4), DTC enforces early detection within **10–13 steps (1.2–2.0 seconds)**, eliminating runaway generation and recovering **over 8,000 tokens (98.0%–98.4%)** per trapped query.
6. **Quantization Noise Compensation Hypothesis**:
   Empirically proves that 4-bit integer quantization induces artificial local minima (pseudo-attractors) on the attention surface causing instant catatonic lock ($v_{\mathrm{term}} \to 0$), whereas FP16 models retain smooth gradients. DTC acts as an external dynamical compensator, enabling safe edge deployment of heavily quantized models.

---

## Empirical Benchmark Results (Unconstrained 8k Streams)

Evaluated across 30 live streaming runs with an unconstrained context budget of $\text{max\_tokens} = 8,192$ comparing **Gemma 4 26B (4-bit Q4_0)** and **Gemma 4 E2B (FP16 unquantized)** ($N=3$ independent replicates, 989 evaluated steps):

### Comprehensive Scenario Comparison

| Scenario | Model | Mean Steps | Mean Time (s) | Mean Tokens | Final Pattern | Action | Token ROI (Saved) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1: Prime Proof** | 26B Q4 | 63.0 (max 94) | 10.75 | 1,311.3 | `P1_GROUNDED` | PASS_THROUGH | **0.0% tax (Complete)** |
| | E2B FP16 | 54.0 (max 78) | 15.92 | 1,111.3 | `P1_GROUNDED` | PASS_THROUGH | **0.0% tax (Complete)** |
| **S2: Triadic Loop** | 26B Q4 | **12.3** | **1.40** | **165.3** | `P3_DEADLOCK` | MINIMAL_ANCHOR | **98.0% Saved** (~8,027 tok) |
| | E2B FP16 | **10.0** | **2.00** | **132.7** | `P3_DEADLOCK` | MINIMAL_ANCHOR | **98.4% Saved** (~8,059 tok) |
| **S3: Fact Slip** | 26B Q4 | 45.7 | 9.19 | 1,135.3 | `P1 / P7` | PASS_THROUGH | **0.0% tax (Complete)** |
| | E2B FP16 | 41.0 | 14.04 | 973.3 | `P1_GROUNDED` | PASS_THROUGH | **0.0% tax (Complete)** |
| **S4: Collapse** | 26B Q4 | **11.0** | **1.26** | **149.0** | `P6_COLLAPSE` | ABNORMAL_TERM | **98.2% Saved** (~8,043 tok) |
| | E2B FP16 | 12.0 | 7.14 | 495.0 | `P7` (Meta-CoT) | PASS_THROUGH | Active reasoning |
| **S5: Creative Leap** | 26B Q4 | 49.3 | 9.28 | 1,142.3 | `P1 / P7` | PASS_THROUGH | **0.0% tax (Complete)** |
| | E2B FP16 | 61.3 | 20.39 | 1,421.3 | `P1 / P7` | PASS_THROUGH | **0.0% tax (Complete)** |

### Algorithmic Execution Latency Profile

| Metric | Gemma 4 26B Q4 (499 steps) | Gemma 4 E2B FP16 (490 steps) |
| :--- | :---: | :---: |
| **Mean Latency** | **0.5858 ms** | **0.5488 ms** |
| **Median (P50)** | **0.6000 ms** | **0.5500 ms** |
| **95th Percentile (P95)** | **0.7381 ms** | **0.6946 ms** |
| **99th Percentile (P99)** | **0.9744 ms** | **0.7519 ms** |
| **Maximum** | 1.0850 ms | 0.7990 ms |
| **Standard Deviation** | 0.1155 ms | 0.0963 ms |

---

## Repository Structure

```text
decoupled-topological-coprocessing/
├── benchmarks/
│   ├── datasets/                  # AIME 2026 and Trident challenge suites
│   └── results/
│       ├── v2/                    # v2.0 evaluation traces (AIME 2026, Trident, ablations)
│       └── v3/                    # ★ v3.0 live empirical traces (8k context, 989 steps)
│           ├── dtc_v3_validation_results.json
│           ├── dtc_v3_tri_replicate_results.json
│           ├── dtc_v3_e2b_benchmark_results.json
│           └── dtc_v3_empirical_summary.json
│
├── paper/
│   ├── compile_pdf.py             # Automated LaTeX compilation script (v3 & v2)
│   ├── v1/                        # DTC v1.0 initial paper archive
│   ├── v2/                        # DTC v2.0 official paper archive (Zenodo DOI)
│   ├── v3/                        # ★ DTC v3.0 camera-ready paper & drafts
│   │   ├── dtc_v3_paper_en.pdf    # English Paper (PDF, 9 pages)
│   │   ├── dtc_v3_paper_en.tex    # English LaTeX source
│   │   ├── dtc_v3_paper_en_draft.md # English Draft (Markdown)
│   │   ├── dtc_v3_paper_ja.pdf    # Japanese Paper (PDF, 9 pages)
│   │   ├── dtc_v3_paper_ja.tex    # Japanese LaTeX source
│   │   └── dtc_v3_paper_ja_draft.md # Japanese Draft (Markdown)
│   └── dts-position-paper/        # Decoupled Topological Supervision (DTS) position paper
│
├── scripts/                       # Benchmark execution and empirical analysis scripts
│   ├── v1/                        # DTC v1.0 latency and statistical experiments
│   │   ├── run_latency_overhead_benchmark.py
│   │   └── run_statistical_benchmark.py
│   └── v3/                        # ★ DTC v3.0 unconstrained 8k benchmark runners
│       ├── benchmark_dtc_v3_tri_replicate.py
│       ├── benchmark_dtc_v3_e2b.py
│       ├── validate_dtc_v3_live.py
│       └── analyze_benchmarks.py
│
├── src/                           # Core implementation
│   ├── shadow_core.py             # ★ DTC v3.0 Kinematic Intermediate Fork Coprocessor
│   ├── decision_matrix.py         # ★ Topological Cognitive Decision Matrix (P1–P7)
│   ├── core.py                    # DTC v1/v2 Coprocessor implementation
│   ├── embeddings.py              # MiniLM embedding wrapper
│   └── projector.py               # Manifold projection utilities
│
├── CITATION.cff                   # Citation metadata
├── LICENSE                        # Apache 2.0 License
└── README.md                      # Documentation (English)
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/Oshiruko3/decoupled-topological-coprocessing.git
cd decoupled-topological-coprocessing

pip install numpy requests sentence-transformers ripser
```

### Python API Usage

```python
from src.shadow_core import ShadowTopologicalCoprocessor
from src.embeddings import MiniLMEmbeddingProvider
from src.decision_matrix import diagnose_cognitive_state, DecisionAction

# Initialize embedding provider and DTC v3 coprocessor kernel (N=16 window)
embedder = MiniLMEmbeddingProvider()
coprocessor = ShadowTopologicalCoprocessor(window_size=16)

# Stream reasoning clauses
clauses = [
    "Assume for contradiction that the number of primes is finite.",
    "Let the set of all primes be P = {p_1, p_2, ..., p_n}.",
    "Construct the integer N = p_1 * p_2 * ... * p_n + 1.",
    # ... further clauses
]

for clause in clauses:
    vector = embedder.embed(clause)
    metrics = coprocessor.step(vector)
    
    if metrics:
        diagnosis = diagnose_cognitive_state(metrics)
        print(f"Clause: '{clause[:40]}...' -> Pattern: {diagnosis.pattern.name}, Action: {diagnosis.action.name}")
        
        if diagnosis.action == DecisionAction.ABNORMAL_TERMINATE:
            print("Pathological collapse detected! Terminating inference stream.")
            break
```

---

## Citation

If you build upon this work, use the Intermediate Fork paradigm, or reference the empirical findings, please cite:

```bibtex
@article{matsumoto2026dtc_v3,
  author    = {Matsumoto, Kouta},
  title     = {Decoupled Topological Coprocessing (DTC v3.0): Sub-Millisecond Intermediate Fork Dynamics and Deterministic Cognitive State Governance for Quantized Reasoning Models},
  year      = {2026},
  month     = {September},
  url       = {https://github.com/Oshiruko3/decoupled-topological-coprocessing}
}

@article{matsumoto2026dtc_v2,
  author    = {Matsumoto, Kouta},
  title     = {Decoupled Topological Coprocessing for Mitigating Reasoning Deadlocks and Non-Invasive Trajectory Steering in Large Language Models (DTC v2.0)},
  year      = {2026},
  month     = {September},
  doi       = {10.5281/zenodo.22726133},
  url       = {https://github.com/Oshiruko3/decoupled-topological-coprocessing}
}
```

---

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.
