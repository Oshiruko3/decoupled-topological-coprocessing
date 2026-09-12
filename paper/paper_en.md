# Decoupled Topological Coprocessing: Non-Intrusive Hallucination Mitigation and Trajectory Steering in Frontier Reasoning and Generation

**Kouta Matsumoto**  
*Independent Researcher*  
`Oshiruko3@users.noreply.github.com`

---

### Abstract

When large language models (LLMs) engage in complex, multi-step reasoning and extended generation within closed-world environments devoid of external retrieval or execution feedback, they are prone to self-reinforcing logical circularity and progressive hallucination cascades within their latent trajectory manifolds. This pathological failure mode not only degrades generation fidelity but also imposes prohibitive computational overhead during inference due to the exponential proliferation of unproductive reasoning steps. In this work, we propose **Decoupled Topological Coprocessing (DTC)**, an architectural paradigm that continuously monitors the geometry of latent embedding trajectories in real time to detect and resolve hallucinatory circularity without modifying model weights or CUDA execution kernels.

DTC orthogonally decouples topological verification from the primary generation engine, executing localized persistent homology analysis asynchronously on a lightweight coprocessor layer. By tracking the birth and persistence of non-trivial 1-dimensional homology cycles ($H_1$) across sliding token windows, and filtering these candidates through an **Epistemic State-Transition Filter** featuring a ternary watchlist (`State Both`), DTC robustly discriminates between healthy mathematical verification/backtracking and pathological tautological deadlocks. Upon confirmation of an unresolvable cycle, the coprocessor issues a non-intrusive stream abort, triggering contextual re-anchoring via **PreserveThinking Injection** and leveraging native Prefix Caching within modern inference engines (e.g., vLLM, SGLang, llama.cpp) for instant, zero-delay trajectory recovery.

Empirical evaluations conducted on an active local deployment (Gemma 4 26B) across challenging American Invitational Mathematics Examination (AIME) reasoning tasks demonstrate that while unconstrained baseline models succumb to circular traps (0.0% completion rate), DTC achieves an **83.3% completion rate** and a **33.3% exact proof accuracy**, while reducing redundant reasoning tokens by up to **91.8%** (mean reduction of 66.7%). Furthermore, end-to-end latency benchmarks confirm that asynchronous TDA coprocessing imposes merely a **3.26% throughput overhead** (averaging 33.1 ms per topological scan on a background CPU thread). Operating purely as an API-level middleware, DTC establishes a foundational architectural component for resilient and cost-effective frontier inference.

---

### 1. Introduction

Frontier reasoning models increasingly deploy extended Chain-of-Thought (CoT) trajectories—spanning thousands to tens of thousands of intermediate tokens—to autonomously solve competition-grade mathematics, formal logic, and scientific synthesis. However, in purely deductive, closed-world reasoning domains lacking programmatic execution environments or compiler feedback, minute errors introduced in early reasoning steps frequently amplify through the autoregressive generation process. This dynamic causes the generation trajectory to converge toward false topological attractors within the latent manifold. Once an erroneous premise is consolidated, subsequent reasoning steps become trapped in a self-justifying circular loop, culminating in confident yet entirely spurious conclusions.

Existing paradigms for hallucination detection fall predominantly into two categories:
1. **Sampling-Based Consensus and Semantic Entropy** (e.g., SelfCheckGPT[^1]): These techniques mandate multiple independent stochastic generations, rendering them computationally prohibitive for interactive real-time deployment and low-latency streaming.
2. **Post-Hoc Topological and Attention Audits** (e.g., TOHA[^2], HalluZig[^3], and TAD[^4]): While these approaches demonstrate that semantic deviations correlate strongly with geometric anomalies (such as zigzag persistence and attention-graph divergence), they operate strictly after complete sequence generation, precluding dynamic self-correction during the reasoning process.

A common structural limitation across conventional verification frameworks is their **synchronous design**: pausing primary token generation to perform periodic verification introduces unacceptable latency penalties. In this paper, we propose an orthogonal decoupling of the token generation pipeline and the geometric auditing layer. By allowing the primary engine to stream tokens at maximum throughput while an asynchronous coprocessor monitors trajectory topology in parallel, DTC achieves continuous, real-time hallucination mitigation with negligible computational overhead.

---

### 2. Hallucinations as Topological Cycle Defects

A logically sound and progressive Chain-of-Thought trajectory traces an open, non-recurrent path through high-dimensional latent representation space. Conversely, when an LLM enters an ungrounded or self-referential loop—repeatedly affirming an erroneous lemma through circular rephrasing—the trajectory folds upon itself, generating a non-trivial 1-dimensional persistent homology cycle ($H_1$) over localized token spans.

Prior studies such as TOHA[^2] (identifying attention graph divergence) and HalluZig[^3] (tracing zigzag persistence over reasoning chains) have confirmed that semantic hallucinations exhibit distinct topological signatures. We extend these empirical insights to real-time streaming architectures by computing localized persistence diagrams over sliding windows:
$$\mathcal{W}_t = \{ \mathbf{e}_{t-W+1}, \dots, \mathbf{e}_t \}$$
where $W$ denotes the localized window width (e.g., $W = 16 \sim 32$ steps) and $\mathbf{e}_i$ denotes the contextual latent representation of reasoning step $i$. By applying sparse Vietoris-Rips filtration via Ripser to $\mathcal{W}_t$, we track the maximum 1-dimensional cycle persistence:
$$\text{pers}(c_k) = d_k - b_k$$
where $b_k$ and $d_k$ represent the birth and death filtration scales of the $k$-th homology cycle $c_k$. The localized window formulation constrains the computational complexity, enabling sub-40-millisecond topological audits.

DTC accommodates two distinct deployment modalities:
- **Black-Box API Proxy Layer (Non-Intrusive)**: Trajectory embeddings are extracted from streaming text spans using a fast, compact embedding coprocessor (e.g., `all-MiniLM-L6-v2`). This requires zero modification to server infrastructure or model weights.
- **White-Box Engine Layer**: Hidden states from intermediate or final model layers are gathered directly and standardized via PCA/projection layers within the serving engine.

In this work, we focus primarily on the black-box proxy configuration, demonstrating that lightweight sentence-level embeddings provide sufficient topological fidelity for robust cycle detection.

---

### 3. Architecture: Decoupled Topological Coprocessing

The DTC architecture operates as a dual-process asynchronous system, completely decoupling primary token generation from geometric verification:

```text
[ Primary Generation Engine (LLM) ]
    Token Stream: t_1 ---> t_2 ---> t_3 ---> t_4 ---> t_5 ---> (Unimpeded Fast Path)
                                     ^
                                     | (Non-intrusive Abort upon Confirmed Cycle)
[ Decoupled Coprocessor ]
    Trajectory Embeddings ---> [ Sliding Window TDA Scan (H1 Loop Detection) ]
                                     |
                             (Epistemic State-Transition Filter)
                                     v
                        [ Contextual Re-anchoring (KV Rollback + Prompt) ]
```

#### 3.1 Orthogonal Separation of Generation and Verification
The primary LLM generates tokens continuously at native memory-bandwidth speed. Concurrently, the decoupled coprocessor receives generated token segments via non-blocking asynchronous queues and executes sliding-window TDA on auxiliary CPU cores or dedicated micro-accelerators.

Under normal, progressive reasoning, the coprocessor remains completely passive, imposing zero synchronization barrier on the LLM. Only when an anomalous $H_1$ cycle is detected and subsequently confirmed by the epistemic state-transition filter does the coprocessor dispatch an interruption command to the proxy layer.

#### 3.2 Mitigation Mechanisms: Re-Anchoring vs. Continuous Steering
When an unrecoverable circular deadlock is confirmed, DTC provides two theoretical correction pathways:

1. **Contextual Re-Anchoring via Prefix Caching (Core Practical Paradigm)**:
   Direct mathematical manipulation of latent continuous vectors often distorts the token logit distribution, inducing severe vocabulary collapse (unintelligible gibberish). DTC circumvents this vulnerability by operating at the discrete token level. The API proxy truncates the reasoning stream back to the pre-cycle checkpoint and appends an introspective prompt directive (`PreserveThinking Injection`).
   Crucially, modern serving runtimes (such as vLLM, SGLang, and llama.cpp) natively implement **Prefix Caching** (PagedAttention KV-cache sharing). Because the rolled-back prompt prefix is preserved bit-for-bit, the inference engine instantly resumes generation from cache without recomputing preceding KV tensors, achieving instantaneous trajectory branching.

2. **Wasserstein Manifold Steering (Continuous Geometric Frontier)**:
   In fully white-box environments where direct logit bias manipulation is accessible, trajectory steering can theoretically be formulated as an optimal transport problem. Letting $\nu$ represent a canonical reference distribution of grounded deductions and $\mu$ represent the local trajectory distribution, the 2-Wasserstein distance:
   $$W_2^2(\mu, \nu) = \inf_{\gamma \in \Gamma(\mu, \nu)} \int \|x - y\|^2 \, d\gamma(x, y)$$
   can be regularized via the Sinkhorn algorithm[^5] to inject steering gradients into the unnormalized logits prior to sampling.

#### 3.3 Deployment Modalities
DTC generalizes across multiple operational environments:
- **Speculative Verification**: Embedded within the verification slot of speculative decoding pipelines[^6], masking TDA calculation within memory-bound forward passes.
- **MoE Cluster Routing**: Deployed as an internal asynchronous expert within Mixture-of-Experts architectures.
- **Microservice Sidecar**: Deployed as a zero-dependency HTTP/gRPC reverse proxy in front of standard commercial or private LLM endpoints.

---

### 4. Empirical Validation and Benchmark Results

To rigorously evaluate DTC, we focus on closed-world deduction benchmarks where retrieval-augmented generation (RAG) is inapplicable and programmatic feedback is unavailable. While coding tasks (e.g., SWE-bench) provide deterministic compiler diagnostics, frontier mathematical reasoning benchmarks such as the American Invitational Mathematics Examination (AIME) and Humanity's Last Exam (HLE)[^7] mandate continuous, multi-step deductive rigor where a single uncorrected loop leads to complete task failure.

#### 4.1 Static Sentences vs. Dynamic Trajectories
Evaluating static, isolated sentences from TruthfulQA (100 items) yielded an $H_1$ persistence density ratio of 0.92 between truthful and hallucinatory statements, exhibiting no statistically significant topological separation. Isolated factual assertions form discrete, disconnected point clouds in embedding space. In contrast, non-trivial topological cycles emerge exclusively within temporally ordered, multi-step Chain-of-Thought trajectories.

#### 4.2 Cycle Density Differential in Reasoning Trajectories (Gemma 4 26B)
Using a local production deployment of Gemma 4 26B (Q4_0 quantization, 128K context window), we evaluated identical prompts under grounded reasoning (33 steps) versus adversarial prompts designed to induce circular tautologies (29 steps):
- **Grounded Reasoning Trajectory**: $H_1$ cycle density = **0.2424** (8 transient loops, resolving naturally).
- **Hallucinatory Trajectory**: $H_1$ cycle density = **0.4138** (12 persistent loops, self-reinforcing attractor).

The hallucinatory trajectory exhibited a **1.71x increase** in $H_1$ cycle density. Two-dimensional PCA projections confirmed that while grounded trajectories trace open spiral manifolds, hallucinatory trajectories collapse into dense, low-dimensional circular attractors.

![2D PCA Projection of Reasoning Trajectories: Grounded Open Spiral vs. Hallucinatory Circular Attractor](../assets/cot_trajectory_pca.png){ width=85% }

#### 4.3 Real-Time Streaming Interruption Proof-of-Concept
We subjected the system to an adversarial circular prompt ($A \implies B \implies C \implies A$). Monitoring the Server-Sent Events (SSE) stream with a sliding window ($W = 8$ steps), the coprocessor detected a sharp spike in $H_1$ density at Step 22 ($\text{density} = 0.125$). The coprocessor severed the HTTP connection, rolled back the output stream by 8 steps, and injected the `PreserveThinking` prompt. The model immediately abandoned the circular tautology and resumed constructive analytical deduction without lexical degradation.

#### 4.4 Comparative Benchmark on Competition-Grade Reasoning (AIME / HLE Subset)
We conducted an automated batch evaluation across a curated suite of competition-grade mathematical and logical reasoning problems ($N = 6$):

| Evaluation Metric | Baseline (Unconstrained CoT) | DTC Monitored & Guided | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Completion Rate (Answer Output)** | **0.0%** (0 / 6) | **83.3%** (5 / 6) | **+83.3%** |
| **Exact Mathematical Accuracy** | **0.0%** (0 / 6) | **33.3%** (2 / 6) | **+33.3%** |
| **Reasoning Step / Token Count** | Baseline (52.0 steps) | **17.3 steps** | **-66.7%** |

In the baseline condition, the model became trapped in repetitive calculation cycles (averaging 3.67 persistent loops per problem), exhausting its token budget and failing to produce a final answer in all instances. Under DTC, early cycle detection and re-anchoring restored the completion rate to 83.3%, yielding exact mathematical solutions on complex geometry and incenter distance problems ($3 + \sqrt{51}$ and $\sqrt{65}$). On a combinatorial logic problem exhibiting severe self-referential trapping, DTC reduced reasoning steps from 73 to 6, achieving a **91.8% token reduction**.

#### 4.5 Latency and Throughput Overhead Benchmark
To quantify the runtime cost of asynchronous coprocessing, we benchmarked sustained token generation on an identical long-form exposition prompt:

| Benchmark Metric | Baseline (No Monitoring) | DTC Monitored | Impact / Overhead |
| :--- | :---: | :---: | :---: |
| **Time to First Token (TTFT)** | 2562.7 ms | 2244.1 ms | -318.6 ms (-12.4%) |
| **Token Generation Throughput** | 129.2 tok/s | 125.0 tok/s | **+3.26% overhead** |
| **Average TDA Scan Latency** | N/A | **33.13 ms** | Background CPU thread |

Ripser filtrations averaged 33.13 ms per scan, executing entirely within background CPU threads without contending for GPU compute or memory bandwidth. Consequently, sustained streaming throughput experienced only a negligible 3.26% overhead, verifying that DTC is thoroughly viable for production-grade, real-time serving.

---

### 5. Discussion and Overcoming Operational Challenges

#### 5.1 Generalization Beyond Deductive Reasoning
While primarily evaluated on mathematical reasoning, DTC extends naturally to broader generative domains:
1. **Source Divergence in Summarization**: Tracking the Wasserstein distance between the source document manifold and the evolving summary trajectory enables real-time alerts when hallucinations depart from factual premises.
2. **Elimination of Tautological Prolixity**: Identifying stylistic loops in long-form generation ensures steady, forward-moving discursive progression.
3. **Real-Time Style and Constraint Enforcement**: Dynamically re-anchoring trajectories relative to canonical stylistic manifolds.

#### 5.2 Theoretical Boundaries: The Geometric Stabilizer
It is critical to demarcate the theoretical boundaries of DTC:
1. **No Fact Creation**: DTC cannot synthesize facts or parametric knowledge absent from the underlying model weights.
2. **No Capability Expansion**: DTC does not augment the intrinsic reasoning capacity beyond the theoretical upper bound established by the model's architecture and pretraining.
3. **The Stabilizer Role**: Rather than raising theoretical intelligence limits, DTC acts as a **geometric stabilizer**, preventing the model from collapsing into self-reinforcing local attractors and ensuring that its latent reasoning capabilities are fully realized.

#### 5.3 Mitigating False Positives via the Epistemic State-Transition Filter
A central dilemma in applying TDA to reasoning is the risk of false positives: penalizing legitimate mathematical backtracking, proof-by-contradiction verification, or sanity-checking calculations.

To resolve this, DTC incorporates an **Epistemic State-Transition Filter** based on a ternary state machine:
- `State 0 (Grounded Progression)`: Unconstrained generation.
- `State Both (Epistemic Watchlist)`: Transient $H_1$ loop detected; generation continues unhindered while trajectory dynamics are tracked over a 1–2 step lookahead window (20–40 tokens).
- `State 1 (Confirmed Circular Deadlock)`: If the trajectory fails to escape the localized attractor, the state transitions to confirmed deadlock, triggering stream termination.

If the loop was merely a healthy verification step, the reasoning path naturally unwinds into an open helical trajectory, transitioning back from `State Both` to `State 0`. This dynamic state filtering drastically suppresses false interruptions without incurring synchronization latency.

#### 5.4 Asymmetric Cost-Efficiency
DTC operates entirely outside the primary GPU execution path. Given the unit cost of LLM forward passes ($C_{\text{LLM}}$) versus CPU topological scans ($C_{\text{TDA}}$), the computational cost ratio satisfies $C_{\text{TDA}} \ll C_{\text{LLM}}$ (typically $< 10^{-3}$). By spending microsecond-scale CPU cycles to abort multi-thousand-token wasteful generation loops, DTC delivers an asymmetric return on investment for high-throughput inference clusters.

---

### 6. Conclusion

In this position paper, we formulated generative hallucinations in frontier reasoning models as topological cycle defects ($H_1$) within latent trajectory manifolds and introduced **Decoupled Topological Coprocessing (DTC)**. By orthogonally decoupling generation from topological audit, filtering candidate cycles through an Epistemic State-Transition Filter, and triggering contextual re-anchoring via native Prefix Caching, DTC functions as a non-intrusive geometric stabilizer for extended inference. Empirical evaluations demonstrate dramatic improvements in reasoning completion (+83.3%) and exact accuracy (+33.3%), massive reductions in wasted reasoning tokens (up to 91.8%), and negligible streaming throughput overhead (3.26%). As frontier models undertake increasingly autonomous, long-horizon deduction, decoupled geometric coprocessing represents a vital architectural step toward reliable, cost-efficient intelligence.

---

### References

[^1]: Manakul et al., "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models", *EMNLP*, 2023.  
[^2]: Kuleshova et al., "Hallucination Detection in LLMs with Topological Divergence on Attention Graphs", *arXiv:2504.10063*, ACL 2026.  
[^3]: "HalluZig: Detecting Hallucinations via Zigzag Topological Persistence in Reasoning Trajectories", *arXiv:2601.01552*, 2026.  
[^4]: "Topological Attribution Dynamics (TAD) for Segment-Level Hallucination Identification", *arXiv:2608.16775*, 2026.  
[^5]: Cuturi, M., "Sinkhorn Distances: Lightspeed Computation of Optimal Transport", *NeurIPS*, 2013.  
[^6]: "Batch Speculative Decoding Done Right", *arXiv:2510.22876*, 2025.  
[^7]: Center for AI Safety & Scale AI, "Humanity's Last Exam: A Multi-Modal Benchmark for Frontier Reasoning", 2025.  
[^8]: "A Survey of Topological Data Analysis Applications in Natural Language Processing", *OpenReview:1323b8e489*, 2024.
