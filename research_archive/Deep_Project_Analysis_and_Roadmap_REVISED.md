# Deep Project Analysis and Execution Roadmap (Revised)
## Streaming Continual Learning with Fisher-Information-Based Adaptive Memory

> **What changed from the original draft, and why:** the original roadmap was well-organized but had four problems: (1) it oversold several borrowed techniques as "industry standard" when they are actually recent, narrow, or unvalidated; (2) it stacked eight distinct research contributions from six different papers into one solo project, which is a scope-creep risk; (3) it created an unresolved conceptual conflict between Fisher-Information scoring (your headline novelty) and SFAO (which is explicitly pitched as an alternative to Fisher-based methods); (4) it underestimated the compute cost of running four separate gradient-heavy mechanisms together on a 4GB GPU / free-tier cloud instance. This revision keeps everything you wanted but restructures it into a **Core** you can finish and defend on its own, plus **Stretch** upgrades you only attempt once the Core works end-to-end.

---

## 1. Problem Foundation (unchanged, confirmed accurate)

**Catastrophic forgetting:** sequential training on Task B overwrites the weight configurations that encoded Task A, because both tasks share the same parameter space. Real-world systems (robotics, IoT, edge AI) receive unbounded, unchunked data streams and cannot retrain from scratch on each new batch — this is the practical motivation for continual learning.

**Base paper's mechanism (Zhou & Li, Scientific Reports 2026):**
1. **Neural ODEs** — hidden states evolve as continuous-time trajectories (`dh/dt = f(h,t,θ)`) instead of discrete layers, so new-task learning perturbs the trajectory smoothly rather than overwriting weights abruptly.
2. **Memory-Augmented Transformer** — an explicit 200-slot external memory bank, read via attention, decouples long-term knowledge storage from the model's own weights.
3. **Adaptive memory management** — an influence-function approximation scores which stored samples to keep.

*(Reminder from our earlier review: treat this base paper's specific numbers — 24% forgetting reduction, 10.3% accuracy gain — as a target to reproduce and verify yourself, not as an established ground truth. The paper has citation-integrity issues that warrant independent verification before you cite its deltas as your comparison baseline.)*

---

## 2. Your Three Original Contributions (the actual novelty — keep these central)

| # | Gap in base paper | Your contribution | Status |
|---|---|---|---|
| 1 | Assumes offline, pre-chunked tasks | **Streaming Continual Learning** — continuous ingestion, no task boundaries | Core |
| 2 | Influence-function memory scoring degrades at scale | **Fisher-Information-based memory importance** — parameter-sensitivity scoring instead of influence-function approximation | Core |
| 3 | Memory grows unbounded | **Adaptive fixed-size eviction** — evict lowest Fisher-score samples when buffer is full | Core |

Everything else in the implementation plan (DINOv2, Tri-Scale ODE, Titans/MIRAS gating, SFAO/SGP, Long-CL decay) exists to **fill in blanks the base paper left unspecified** — they are engineering choices, not your research novelty. Keep that distinction explicit in your own head and in any writeup, so the paper's contribution claims stay clean.

---

## 3. Honest Assessment of the Borrowed Techniques

Before adopting each one, here's what it actually is — verified, not assumed:

| Technique | What it really is | Fit for this project | Verdict |
|---|---|---|---|
| **DINOv2** (Meta, 2023) | Frozen self-supervised Vision Transformer, mature and widely used for feature extraction | Directly applicable — removing the backbone from the training loop is a legitimate way to save VRAM and sidestep an underspecified detail in the base paper | **Use in Core.** Low risk, high payoff. |
| **SFAO** (Selective Forgetting-Aware Optimization, arXiv 2603.26671, Algoverse AI Research) | A small, recent (not a major-lab) paper, validated only on MNIST-scale benchmarks, explicitly pitched as a *cheap alternative to Fisher-based gradient projection* | Conceptually in tension with your headline Fisher contribution if used for the same purpose; useful only if scoped to a clearly separate role | **Move to Stretch**, and if used, scope it strictly to gradient-update regulation — never let it substitute for your Fisher-based memory scoring, or your novelty claim gets muddled. |
| **Scaled Gradient Projection / SGP** (Saha & Roy, AAAI 2023) | Established, peer-reviewed, reasonably well-cited | Legitimate technique for plasticity/stability balancing | **Stretch**, safe to add once Core works. |
| **Titans / MIRAS** (Behrouz et al., Google Research, 2025) | Real, prominent, but built for **language sequence modeling at 2M+ token context**, not image-based continual learning | Adapting its "surprise metric" and MAG gating to your setting is genuine, non-trivial research work — not a drop-in module | **Stretch**, and frame it in your writeup as "inspired by Titans/MIRAS," not as installing an established component. |
| **Long-CL** (Huai et al., ECNU/Fudan, arXiv 2505.09952) | Real, built for **multi-modal/text long-term CL** (their own MMLongCL-Bench / TextLongCL-Bench), not vision classification | The "task-core memory" idea is a reasonable inspiration for your own decay formula, but don't expect a literal formula transplant | **Stretch**, reframe as "informed by Long-CL's memory-consolidation approach" rather than "using Long-CL's formulas." |

---

## 4. Hardware Strategy (kept, with one addition)

- **Local development (RTX 3050, 4GB VRAM):** write and debug all code with tiny dummy tensors (2–4 samples/batch) to validate shapes, gradient flow, and logic — never attempt real training locally.
- **Feature pre-caching:** run frozen DINOv2 once over each dataset, save embeddings to disk, and remove DINOv2 entirely from the training loop. This is the single biggest VRAM saver in the plan.
- **Cloud training (Colab/Kaggle, T4/P100, ~16GB):** this is where real training happens.
- **Addition — budget your gradient-heavy components.** Each of these adds a separate backward pass or gradient computation per step: the ODE adjoint solve, Fisher-Information estimation, and (if you add them in Stretch) SFAO's per-layer cosine gating and Titans' internal test-time gradient descent. On a 16GB free-tier GPU, running all of these simultaneously is the most likely point of failure. Get the Core (ODE + memory + Fisher scoring) training stably first; add Stretch components one at a time, profiling VRAM after each addition.

---

## 5. Step-by-Step Roadmap

### Phase 0 — Verification (new)
*Do this before writing any model code.*
- **Step 0a:** Implement the base paper's architecture (Neural ODE + Memory-Augmented Transformer + influence-function scoring) as faithfully as you can, and reproduce at least one baseline number on one dataset (e.g., Split CIFAR-100 accuracy). This gives you your own verified comparison point instead of relying solely on the base paper's reported deltas.
- **Step 0b:** Log this baseline clearly — you'll cite *your own* reproduction number alongside the base paper's claim in your final writeup.

### Phase 1 — Environment & Data Engineering
- **Step 1:** Python environment, `requirements.txt`, project structure (`src/models`, `src/training`, `src/data`).
- **Step 2:** `data.py` — download Split CIFAR-100, Permuted MNIST, CORe50.
- **Step 3:** DINOv2 feature-extraction pipeline in `data.py`; pre-compute and cache embeddings to disk.

### Phase 2 — Core Architectural Modules (The Brain)
- **Step 4:** `models/ode.py` — a **single-scale** Neural ODE function `f`, wrapped in `torchdiffeq`'s Dormand-Prince solver. (Tri-Scale is Stretch — see Phase 5.)
- **Step 5:** `models/memory.py` — the fixed-slot memory bank with content-addressable read/write (as specified in the base paper, Eq. 7–8).
- **Step 6:** `models/transformer.py` — the multi-head Transformer that fuses ODE output with memory-retrieved context (residual/gated fusion, per the base paper).
- **Step 7:** `models/hybrid.py` — stitch ODE + Memory + Transformer into one `nn.Module`.

### Phase 3 — Your Core Novel Contributions
- **Step 8:** `training/fisher.py` — compute Fisher Information scores per stored memory sample (this is your headline contribution — implement and document it carefully, it's what differentiates your work).
- **Step 9:** `training/eviction.py` — fixed-size buffer eviction using Fisher scores when memory is full.
- **Step 10:** `training/streaming.py` — the streaming buffer + classifier front-end that removes the predefined-task assumption; feed data continuously rather than in discrete task blocks.

### Phase 4 — Training Engine & Baseline Evaluation
- **Step 11:** `training/engine.py` — master loop: streaming ingestion → forward pass → memory read/write → Fisher scoring → eviction → backward pass → logging.
- **Step 12:** `main.py` and `train_colab.ipynb`.
- **Step 13:** Run on Colab/Kaggle across all three benchmarks; compute average accuracy, forgetting measure, BWT, FWT, memory utilization, streaming throughput. **This is your first complete, defensible result set — the Core is done here.**

### Phase 5 — Stretch Upgrades (attempt only after Phase 4 succeeds, one at a time)
- **Step 14 (optional):** Upgrade single-scale ODE to Tri-Scale (fine/mid/coarse hierarchical blocks). Re-run Phase 4 benchmarks to confirm it actually improves results before keeping it.
- **Step 15 (optional):** Add SGP for gradient-direction protection on top of your existing gradient step. Profile VRAM before/after.
- **Step 16 (optional):** Add a Titans/MIRAS-*inspired* surprise-gated memory write trigger, clearly documented as an adaptation, not a direct implementation of their paper.
- **Step 17 (optional):** Add SFAO only if scoped as a distinct gradient-regulation mechanism, kept separate from your Fisher-based memory scoring, with the distinction stated explicitly in your writeup.
- **Step 18 (optional):** Explore a Long-CL-*inspired* decay formula for older memories, framed as inspiration rather than transplant.

Each Stretch step should be added, benchmarked, and kept only if it measurably improves your metrics — otherwise it's added complexity without payoff, and you can note it as "attempted, did not improve results" in your final writeup, which is itself a legitimate finding.

---

## 6. Definition of Done for Each Phase

- **Phase 0 done when:** you have your own reproduced baseline number, not just the paper's claim.
- **Phase 1 done when:** all three datasets load and DINOv2 embeddings are cached to disk.
- **Phase 2 done when:** a dummy batch flows through ODE → Memory → Transformer without shape errors, on CPU or your local 4GB GPU.
- **Phase 3 done when:** Fisher scores are computed for a batch of stored memories and eviction correctly drops the lowest-scored ones under a simulated full-buffer condition.
- **Phase 4 done when:** you have a full metrics table (accuracy, forgetting, BWT, FWT, memory utilization, throughput) across all three benchmarks from a real cloud training run — this is your minimum publishable/defensible result.
- **Phase 5 items are each done when:** benchmarked against the Phase 4 numbers and kept only if they help.

---

*Review this before Phase 0. Let me know if you want to reorder any Stretch items into Core, or vice versa, before we start.*
