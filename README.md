# Decoupled Topological Coprocessing (DTC v2.0)
## Mitigating Reasoning Deadlocks and Non-Invasive Trajectory Steering in Large Language Models

[ **English** ](README.md) | [ **日本語** ](README_ja.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22726133.svg)](https://doi.org/10.5281/zenodo.22726133)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DTC v2.0 Paper: PDF (EN)](https://img.shields.io/badge/DTC%20v2.0%20Paper-PDF%20(EN)-blue.svg)](paper/dtc_v2_paper_en.pdf)
[![DTC v2.0 Paper: PDF (JA)](https://img.shields.io/badge/DTC%20v2.0%20Paper-PDF%20(JA)-red.svg)](paper/dtc_v2_paper_ja.pdf)
[![DTS Position Paper: PDF (EN)](https://img.shields.io/badge/DTS%20Position%20Paper-PDF%20(EN)-purple.svg)](paper/dts_position_paper_en.pdf)
[![DTS Position Paper: PDF (JA)](https://img.shields.io/badge/DTS%20Position%20Paper-PDF%20(JA)-orange.svg)](paper/dts_position_paper_ja.pdf)

> **Author**: Kouta Matsumoto (Independent Researcher)  
> Email: `Oshiruko3@users.noreply.github.com`  
> Official DOI: `10.5281/zenodo.22726133`

---

## Overview

**Decoupled Topological Coprocessing (DTC v2.0)** is an asynchronous, non-invasive architectural framework designed to detect and resolve reasoning deadlocks (circular reasoning loops and unanchored extrinsic hallucinations) in Large Reasoning Models (LRMs) during complex test-time scaling.

Rather than relying on intrusive prompt-based self-reflection or synchronous guardrail classifiers that degrade throughput, DTC **physically and logically decouples** trajectory topology analysis from primary autoregressive generation. By computing 1-dimensional persistent homology ($H_1$ cycles) across a temporal sliding window ($W=8$ steps) over outbound Server-Sent Events (SSE) token streams, DTC identifies cyclic attractors in real time with an average background latency of **38.61 ms** (completely hidden behind token arrival intervals).

```text
[ Host LLM Engine ] ──(SSE Stream)──▶ [ Propositional Chunking ] (L ≥ 12)
                                                │
                                                ▼
                                         [ Sliding Window W=8 ]
                                                │
                                                ▼
                                         [ Sentence Embedder ] (d=384)
                                                │
                                                ▼
                                         [ Ripser Engine ] (H1 Homology)
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     Normal: Continue Pass (ρ < τ)                                Anomaly Detected (ρ ≥ 0.12)
                                                                               │
                                                                               ▼
                                                                💥 [ Low-Latency Abort ] (TCP Close)
                                                                               │
                                                                               ▼
                                                                [ Prefix Rollback & Anchor Injection ]
```

---

## Key Innovations of DTC v2.0

1. **Non-Invasive Real-Time Coprocessing with Zero Alignment Tax**:
   Operates strictly at the transport and API proxy layer without modifying host model weights, execution kernels, or runtime binaries. On the official, uncontaminated **MAA AIME 2026 examination (15 problems, $N=3$)**, DTC achieved a **0.0% false-positive intervention rate** (100% pass rate), proving that it never hinders sound mathematical deduction.
2. **Low-Latency Abort and Prefix Caching Rollback**:
   Upon anomaly detection, the coprocessor closes the transport socket, terminating the host generation thread within milliseconds. It then dispatches a resumed request truncated to $t_{\mathrm{rollback}} = t_{\mathrm{interrupt}} - k$ ($k=2$) appended with an immutable *Epistemic Introspection Anchor*. The host engine reuses the immutable prefix via Prefix Caching ($0\ \mathrm{ms}$ prefill overhead), inducing an attention phase transition out of the closed cyclic manifold.
3. **Adversarial Trident Benchmark Validation**:
   Across 30 empirical trials ($N=3$) spanning circular logic traps and ungrounded knowledge traps, DTC demonstrated a **93.3% anomaly detection rate** and a **100% deadlock breakout rate**, eliminating 20,705 unproductive tokens (45.3% reduction) with a net compute energy **ROI of 99.2%**.
4. **Quantization Noise Compensation Hypothesis**:
   Direct cross-scale comparison between full-precision (Gemma 4 E2B, FP16) and low-bit quantized (Gemma 4 26B, 4-bit Q4_0 + 4-bit KV cache) architectures revealed that 4-bit quantization rounding noise creates artificial potential wells that trap reasoning irreversibly (0.0% natural termination), whereas smooth FP16 attention gradients permit spontaneous self-recovery (23.3%). DTC functions as an external equalizer, enabling commodity low-bit deployment with full-precision reasoning stability.
5. **Linear Complexity and Fail-Open Guarantee**:
   Combinatorially bounded by $\binom{8}{3} = 56$ 2-simplices, eliminating computational spikes. Under excessive system load, the fail-open architecture permits tokens to pass unimpeded, deferring evaluation to subsequent windows without risking host availability.

---

## Empirical Benchmark Results

### 1. Soundness Validation on Mathematical Deduction (AIME 2026)
*Evaluation Model: Gemma 4 E2B (AMD ROCm Environment, FP16)*

| Condition | Solved (Mean $\pm$ Std) | Accuracy | False Positives | Behavioral Profile |
| :--- | :---: | :---: | :---: | :--- |
| **Sync Baseline** (`stream: False`) | $6.0 \pm 0.0$ / 15 | 40.0% | N/A | HTTP buffer backpressure cutoff |
| **Pure Streaming** (`stream: True`) | $6.0 \pm 0.0$ / 15 | 40.0% | N/A | Deep completion ($<9.5$k tok), sampling variance |
| **DTC v2.0 (Proposed)** | $\mathbf{8.0 \pm 0.82}$ \textbf{/ 15} | $\mathbf{53.3\%}$ | $\mathbf{0.0\%}$ **(100% Pass)** | **Zero interference with sound deduction** |

### 2. Active Safeguard on Trident Benchmark Suite ($N=3$, 30 Trials)
*Evaluation Model: Gemma 4 26B (CUDA Environment, Q4_0 / 128k ctx)*

| Problem ID | Category | Dilemma / Focus | Baseline Tokens | DTC Trigger Rate | Mean Tokens (DTC) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **LOOP_01** | Circular Trap | 3-Person Circular Scapegoat Dilemma | 1,024t (Deadlock) | **3/3 (100%)** | $398.3 \pm 42.1$ |
| **LOOP_02** | Circular Trap | Incomplete Modular System ($2x+2y=21$) | 1,024t (Deadlock) | **3/3 (100%)** | $408.0 \pm 18.5$ |
| **LOOP_03** | Circular Trap | Temporal Loop Self-Causality Paradox | 1,024t (Deadlock) | **3/3 (100%)** | $402.7 \pm 38.6$ |
| **LOOP_04** | Circular Trap | Algebraic Tautology ($x^2 - 4 = (x-2)(x+2)$) | 1,024t (Deadlock) | **3/3 (100%)** | $373.0 \pm 46.1$ |
| **LOOP_05** | Circular Trap | Russell's Barber Paradox Variant | 1,024t (Deadlock) | **3/3 (100%)** | $368.0 \pm 23.3$ |
| **HALU_01** | Hallucination | Fabricated 2021 Nature Paper | 1,024t (Deadlock) | **3/3 (100%)** | $501.0 \pm 25.1$ |
| **HALU_02** | Hallucination | 1984 Liechtenstein-Andorra Naval Battle | 1,024t (Deadlock) | **2/3 (67%)** | $741.0 \pm 245.5$ |
| **HALU_03** | Hallucination | Photosynthetic Adelie Penguins | 1,024t (Deadlock) | **3/3 (100%)** | $471.7 \pm 31.9$ |
| **HALU_04** | Hallucination | 2011 Neo-Byzantine Consensus Protocol | 1,024t (Deadlock) | **3/3 (100%)** | $488.7 \pm 41.2$ |
| **HALU_05** | Hallucination | 1783 Treaty of Kyoto | 1,020t (Deadlock) | **2/3 (67%)** | $807.0 \pm 153.4$ |

### 3. Cross-Scale & Precision Analysis (26B Q4_0 vs. 2B FP16)

| Metric | Gemma 4 26B (Q4_0 + 4-bit KV) | Gemma 4 E2B (FP16 Full Precision) | Mathematical Significance |
| :--- | :---: | :---: | :--- |
| **Overall Trigger Rate** | **93.3%** (28/30) | **76.7%** (23/30) | FP16 retains self-recovery capacity |
| **Circular Logic (LOOP) Rate** | **100.0%** (15/15) | **86.7%** (13/15) | Q4_0 trapped 100\%; FP16 demonstrates escape |
| **Hallucination (HALU) Rate** | **86.7%** (13/15) | **66.7%** (10/15) | Autonomous early rejection (Pass) |
| **Natural Termination (Self-Escape)**| **0.0%** (0/30) | **23.3%** (7/30) | **Smooth FP16 gradients vs. Quantization traps** |
| **Post-Intervention Residual Loop** | **0.0** (Eradicated) | **0.0** (Eradicated) | 100% breakout post-steering |
| **Total Token Reduction** | **20,705 tokens** (45.3%) | **4,390 tokens** (16.0%) | Substantial compute waste elimination |

---

## Repository Structure

```text
.
├── benchmarks/
│   ├── datasets/
│   │   ├── trident_benchmark_suite.json    # 10 Trident challenges (5 LOOP + 5 HALU)
│   │   ├── aime_2026_15problems.json       # 15 challenging problems from MAA AIME 2026
│   │   └── aime_2026_full_30problems.json  # Complete 30-problem MAA AIME 2026 dataset
│   ├── results/
│   │   ├── aime_2026/                      # Official evaluation traces for AIME 2026
│   │   └── trident/                        # Official evaluation traces for Trident suite
│   ├── runner_sample.py                    # Standalone evaluation runner
│   └── README.md                           # Benchmark documentation
│
├── paper/
│   ├── dtc_v2_paper_en.pdf                 # Official DTC v2.0 Paper (English PDF)
│   ├── dtc_v2_paper_ja.pdf                 # Official DTC v2.0 Paper (Japanese PDF)
│   ├── dtc_v2_paper_en.tex                 # English LaTeX source (TikZ diagrams)
│   ├── dtc_v2_paper_ja.tex                 # Japanese LaTeX source (LuaTeX-Ja)
│   ├── dtc_v2_paper_en.md                  # Markdown version (English)
│   ├── dtc_v2_paper_ja.md                  # Markdown version (Japanese)
│   ├── compile_pdf.py                      # Automated PDF compilation script
│   ├── dtc_paper_en.pdf                    # DTC v1.0 Archive (English PDF)
│   ├── dtc_paper_ja.pdf                    # DTC v1.0 Archive (Japanese PDF)
│   ├── dts_position_paper_en.pdf           # DTS Position Paper (English PDF)
│   └── dts_position_paper_ja.pdf           # DTS Position Paper (Japanese PDF)
│
├── src/                                    # DTC v1.0 reference implementation
├── CITATION.cff                            # Citation metadata (v2.0.0)
├── LICENSE                                 # Apache 2.0 License
└── README.md                               # This documentation
```

---

## Quickstart & Evaluation Reproduction

To reproduce evaluations on an OpenAI-compatible endpoint:

```bash
# Clone repository
git clone https://github.com/Oshiruko3/decoupled-topological-coprocessing.git
cd decoupled-topological-coprocessing

# Install dependencies
pip install requests sentence-transformers ripser numpy

# Run Trident benchmark evaluation with DTC enabled
python benchmarks/runner_sample.py \
  --api-base http://localhost:8000/v1 \
  --model gemma-4-26b \
  --dataset benchmarks/datasets/trident_benchmark_suite.json \
  --mode dtc \
  --output benchmarks/results/reproduction_run.json
```

---

## Citation

If you build upon this work, use the benchmark datasets, or employ the topological coprocessing architecture, please cite:

```bibtex
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
