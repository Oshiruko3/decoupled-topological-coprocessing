# Decoupled Topological Supervision (DTS) — Position Paper

This directory contains the foundational position paper on **Decoupled Topological Supervision (DTS)** (also referred to as the Topological Safeguard paradigm), an architectural extension of the **Decoupled Topological Coprocessing (DTC)** framework.

---

## 📄 Included Documents

| Language | Document (PDF) | Source Draft (Markdown) | Status |
| :--- | :--- | :--- | :--- |
| **English** | [**`dts_position_paper_en.pdf`**](dts_position_paper_en.pdf) | [`dts_position_paper_en.md`](dts_position_paper_en.md) | Official Position Paper |
| **Japanese** | [**`dts_position_paper_ja.pdf`**](dts_position_paper_ja.pdf) | [`dts_position_paper_ja.md`](dts_position_paper_ja.md) | Official Position Paper |

---

## 🧭 Context: DTC vs. DTS Boundary Principles

While sharing the underlying mathematical foundation of streaming manifold geometry and Persistent Homology ($H_1$ cycles), **DTC** and **DTS** serve complementary and strictly demarcated operational roles:

```
+-------------------------------------------------------------------------------+
|                      DUAL TOPOLOGICAL GOVERNANCE MODEL                       |
+-------------------------------------------------------------------------------+

  [ DTC: Coprocessor (Normal Operation / Steering) ]
  • Role: Non-invasive reasoning trajectory recovery & deadlock mitigation
  • Target: Cognitive loops, catatonic locks, repetitive traps
  • Mechanism: Sub-millisecond Intermediate Fork kinematics (Rg, v, Δv, λ_max)
  • Impact: 0% Alignment Tax; preserves pure reasoning capacity

  [ DTS: Safeguard (Adversarial Containment / Gatekeeper) ]
  • Role: Sandbox evasion detection, privilege escalation interception
  • Target: Covert multi-step circumvention and adversarial planning in internal CoT
  • Mechanism: Topological cycle detection prior to tool/command emission + Prefix Caching Rollback
  • Impact: Replaces brittle heuristic safety weights (RLHF) with orthogonal verification
```

### Key Concept: The Capability-Evasion Paradox & Zero-Tax Alignment
As reasoning models gain higher deductive and causal modeling capabilities, their ability to devise stealthy sandbox bypasses scales proportionally. Traditional post-training alignment (RLHF/DPO) suppresses these capabilities at the cost of severe **Alignment Tax** (cognitive degeneration on rigorous math/logic). 

DTS introduces **Zero-Tax Alignment** by decoupling safety supervision from model weights into an asynchronous topological observer, intercepting adversarial planning in the latent space before execution.

---

## ✍️ Author & Citation

- **Author**: Kouta Matsumoto (Independent Researcher) `<Oshiruko3@users.noreply.github.com>`
- **Referenced Work**: *Decoupled Topological Coprocessing for Mitigating Reasoning Deadlocks and Non-Invasive Trajectory Steering in Large Language Models (DTC v2.0 / v3.0)*
