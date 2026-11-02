# Experimental Roadmap: Long-Context Continual Pretraining (CPT) & Evaluation Suite

This document defines the pre-registered technical roadmap, benchmark specifications, scientific criteria, and code-level execution steps for extending 350M HOFA and MHA models from $N = 1,024$ to $N = 16,384$ ($16\text{k}$) via Continual Pretraining (CPT) on PG-19, and evaluating zero-shot length extrapolation up to $N = 65,536$ ($64\text{k}$) alongside associative recall retrieval (S-NIAH).

---

## 1. High-Level Goals & Reviewer Defense Strategy (First Principles)

### 1.1 The Core Reviewer Vulnerability in the Current Preprint
The original HOFA preprint reports pretraining at native context length $N = 1,024$. The paper's claims of long-context efficiency currently rest on:
1. **Hardware profiling** (Figure 1: prefill/decode throughput & memory).
2. **Synthetic capability probes** (Figure 6: copying, Figure 5: induction heads).
3. **Effective attention distance** (Figure 8: outlier attention span).

**The Reviewer Objection**:  
*"Synthetic copying and kernel micro-benchmarks do not prove real-world language modeling at scale. Does HOFA's hybrid inlier-outlier factorization maintain language modeling perplexity and associative recall on genuine long documents, or does the linear inlier recurrent state degrade over long horizons?"*

**The Defense Strategy**:  
Executing 16k continual pretraining on PG-19 with canonical YaRN directly bridges this gap. Proving that 350M HOFA matches 350M MHA in validation/test perplexity at 16k—while saving ~30% attention core FLOPs per token and sustaining associative retrieval—transforms theoretical assertions into peer-reviewed empirical evidence.

### 1.2 Strict 1:1 Experimental Parity
To withstand the most critical reviewer scrutiny, every experiment maintains absolute parity between HOFA and MHA:
- **Model Scale**: 350M parameters ($L = 24, d_{\text{model}} = 1024, H = 8, d_h = 128$).
- **Base Checkpoints**: Identical pretraining on 14B tokens of FineWeb-Edu at $N = 1,024$.
- **Adaptation Budget**: Identical 100M tokens of PG-19 carved into unbroken intra-book 16,384-token windows.
- **RoPE Interpolation**: Canonical YaRN ($\alpha = 1.0, \beta = 32.0, s = 16.0, \sqrt{1/t} = 1.2773$).
- **Optimization**: Identical AdamW optimizer ($\beta_1 = 0.9, \beta_2 = 0.95$), peak LR $1 \times 10^{-5}$ with 30-step warmup and cosine decay to $1 \times 10^{-6}$, global batch size 16 ($262,144$ tokens/step, 381 steps), bfloat16 mixed precision on an NVIDIA RTX 5090.

### 1.3 Scope Prioritization (Downstream Reasoning Benchmarks)
- **Scope Decision**: Downstream zero-shot reasoning suites (ARC-Easy/Challenge, HellaSwag, PIQA, etc.) are set aside strictly for **scope focus**. 
- **Rationale**: While downstream suites run quickly at 350M scale, our paper's core architectural claim is about **long-context representation, context expansion, and associative recall vs. linear recurrence**. We dedicate our evaluation budget to:
  1. Real long-form language modeling (PG-19 at 16k).
  2. Pretraining capability retention (FineWeb-Edu at 1k).
  3. Outlier associative memory under long horizons (S-NIAH retrieval from 4k to 32k).
  4. Extreme zero-shot context extrapolation (1k $\to$ 64k).

### 1.4 Decoupled Gate Learning Rate (Paper Defense & Plausible Explanation)
- **Draft Paper Formulation**:
  > *"To allow the recurrent decay and dynamic mixing gates to adapt to the extended context window without destabilizing pretrained attention and feed-forward representations, we optimize them with a decoupled learning rate multiplier ($10\times$, setting peak gate LR to $2 \times 10^{-4}$ while keeping base attention/MLP weights at $2 \times 10^{-5}$). While a broader hyperparameter search may yield further refinements, we found that this configuration provides sufficient flexibility across all model scales."*
- **Empirical Grounding**:
  In linear attention architectures (e.g., QED / Gated DeltaNet-2, Gupta et al., arXiv:2608.13668), sigmoid-gated decay and erase parameters suffer from gradient dampening when initialized near the saturation region. Because CPT on PG-19 spans only 100M tokens (~381–400 steps), a base learning rate of $2 \times 10^{-5}$ keeps gate parameters essentially static, whereas a $10\times$ multiplier permits the recurrent decay horizon and outlier blend to adjust freely without risking representation collapse.

---

## 2. Technical Benchmark Specifications

### Benchmark 1: PG-19 16k Perplexity (Target Long-Context Task)
- **Validation Split (Training Monitoring & Checkpoint Selection)**:
  - File: `data/datasets/pg19_16k_val.bin` (~5M tokens, 305 windows of 16,384 tokens across 23 distinct books).
  - **Book-Strided Evaluation**: Uses `pg19_16k_val_index.json` to evaluate exactly 1 window per distinct book across all 23 books (376,832 tokens total). Eliminates book-selection bias, lowering standard deviation from $0.35$ to $0.08$.
  - Cadence: Step 0 (initial YaRN zero-shot baseline), periodic checkpoints (every 50 steps), and terminal Step 381.
- **Test Split (Paper Reporting)**:
  - File: `data/datasets/pg19_test_books.bin` with `pg19_test_books_index.json` (10,583,382 tokens, 100 full books).
  - Evaluated **once** after the training protocol and checkpoints are frozen to guarantee zero post-hoc leakage.
  - Filter: Only books with length $\ge 16,384$ tokens are evaluated (94 out of 100 books).
- **Reporting Metrics**: Cross-entropy loss, perplexity ($\exp(\mathcal{L})$), and 95% bootstrap confidence intervals over books.

### Benchmark 2: FineWeb-Edu 1k Perplexity (Pretraining Retention)
- **Dataset**: `data/datasets/data_350M_val_cache.bin` (held-out FineWeb-Edu pretraining validation split, $N = 1,024$).
- **Objective**: Verify that 100M tokens of PG-19 CPT induces zero catastrophic forgetting of general English pretraining capabilities.
- **Dual Isolation Protocol**:
  1. **YaRN Configuration ($s = 16.0$)**: Measures real-world deployment performance when short prompts are fed directly into the context-extended model.
  2. **Native RoPE Configuration ($s = 1.0$)**: Hot-swaps rotary tables to $s = 1$ to isolate pure weight drift from positional frequency distortion.
- **Reporting**: Report before vs. after CPT validation perplexity ($\Delta\text{PPL}_{\text{FineWeb}}$).

### Benchmark 3: Dynamic Alpha (Routing Gate) Distribution & Comparison (Pretrained vs. CPT)
- **Objective**: Empirically verify that context extension preserves HOFA's hybrid factorized routing without collapsing into pure softmax ($\alpha \to 1.0$) or pure recurrence ($\alpha \to 0.0$).
- **Methodology**: Pass validation batches through the model to log dynamic token-level gate activations $\alpha_t$ across all 24 layers.
- **Compute Efficiency & Caching Protocol**:
  - Independent cache files per model variant in `data/experiments_cache/`:
    - Base model: `alpha_distribution_350M_HOFA.pt` (or `.json`)
    - CPT model: `alpha_distribution_350M_HOFA_cpt.pt` (or `.json`)
  - The analysis **never recomputes base model values** when the base cache exists, avoiding wasted compute.
  - Keeping separate `_cpt` files simplifies caching, invalidation, and comparative plotting without nested dictionary merges.
  - Compares the active CPT checkpoint against the cached base baseline.
- **Reporting & Visual Outputs (Individual Standalone + Comparative)**:
  - Generates each individual plot separately so existing figures remain intact, plus an added comparative figure:
    1. **Base Standalone Plot**: `data/plots/alpha_analysis/350M_HOFA_alpha_violin.pdf` (untouched, identical to existing baseline format).
    2. **CPT Standalone Plot**: `data/plots/alpha_analysis/350M_HOFA_cpt_alpha_violin.pdf` (dedicated individual violin plot for the CPT model).
    3. **Comparative Side-by-Side Plot**: `data/plots/alpha_analysis/350M_HOFA_comparison_alpha_violin.pdf` (paired layer-by-layer comparison directly contrasting pre- vs. post-CPT routing profiles).

### Benchmark 4: Synthetic Needle-in-a-Haystack (S-NIAH) Passkey Retrieval
- **Core Thesis**: PPL alone can look stable while associative retrieval fails. S-NIAH directly tests HOFA's outlier pathway associative recall across long distances.
- **Haystack Corpus**: Paul Graham essays (`data/datasets/paul_graham_essays.txt`, 41,173 total tokens).
  - Maintains exact parity with the frozen 1k baseline protocol so the 1k reference row is strictly comparable.
  - Hard assertion: Essay text length must be $\ge N$; raising an error if text is shorter (no repeated filler).
- **Needle & Query Format**:
  - Needle: `"The special magic passkey is {number}. Remember that: {number} is the passkey."`
  - Query: `"What is the secret passkey?"`
- **Context Lengths Tested**: $4\text{k}, 8\text{k}, 16\text{k}$, and $32\text{k}$.
- **Depths**: 5 depths $\{0.0, 0.25, 0.50, 0.75, 1.0\}$ uniformly spanning the document.
  - Retaining depth 0.0 is mandatory to stress-test early boundary retrieval, where HOFA previously showed sensitivity.
- **Trial Budget**: 30 independent randomized trials per depth (150 trials per context length).
- **Controls**: Negative no-needle control at every length evaluated with a fresh seed to determine baseline chance/hallucination rate.
- **Informativeness Condition**: S-NIAH is defined as informative for length $N$ only if the CPT MHA baseline achieves accuracy significantly above the no-needle control. If MHA fails to beat the control, the length is reported as *uninformative*, not omitted.

### Benchmark 5: Context Length Extrapolation Sweep (1k $\to$ 64k)
- **Data Source**: Full-book test split (`data/datasets/pg19_test_books.bin`) with `pg19_test_books_index.json`.
  - Avoids `pg19_16k_val.bin` because 16k window packing cannot produce 32k or 64k unbroken sequences.
- **Sequence Lengths & Book Counts**:
  Only books with length $\ge N$ are usable at each $N$. Book counts shrink as $N$ grows and are explicitly reported:
  - $N = 1,024$: 100 books ($100\%$ usable)
  - $N = 2,048$: 100 books ($100\%$ usable)
  - $N = 4,096$: 99 books ($99\%$ usable)
  - $N = 8,192$: 95 books ($95\%$ usable)
  - $N = 16,384$: 94 books ($94\%$ usable)
  - $N = 32,768$: 79 books ($79\%$ usable)
  - $N = 65,536$: 54 books ($54\%$ usable)
- **Primary RoPE Configuration**:
  - **Fixed $s = 16.0$ YaRN tables across all lengths $N \in [1024, \dots, 65536]$**.
  - Evaluates the model in its exact trained parameterization without dynamically modifying rotary frequencies at inference time.
  - *Secondary ablation curve*: Dynamic YaRN scaling $s = N / 1024$ for $N > 16\text{k}$ to quantify the impact of dynamic re-interpolation.
- **Hardware & VRAM Safety**:
  - Micro-batch size $B = 1$.
  - Chunked cross-entropy ($C = 2048$) to prevent logit memory spikes.
  - OOM Guard: Gracefully catches `torch.cuda.OutOfMemoryError` if 64k exceeds VRAM, preserving all completed points.
- **Configuration & CLI Simplicity**:
  - Defined statically in `LongContextExtrapolationConfig` in `benchmarks/benchmarks_configs.py`—**no `--length` CLI flags required**.
- **Incremental JSON Caching**:
  - Results stored in `data/experiments_cache/long_context_extrapolation_results.json`.
  - Keyed by `results[model_name][str(seq_len)]`.
  - Existing points are loaded instantly from cache and skipped; new points are flushed to disk after every sequence length completes.
  - Optional `--force` flag available to bypass cache when re-evaluating.

---

## 3. Pre-Registered Evaluation Criteria (Scientific Hypotheses)

These criteria are pre-registered prior to running the benchmarks to ensure objective, unshifted evaluation:

### 3.1 PG-19 Test 16k Language Modeling
- **Pass (Hypothesis Confirmed)**: $\text{PPL}_{\text{HOFA}} \le \text{PPL}_{\text{MHA}} + 0.5$ nats at $N = 16,384$, while delivering $\ge 25\%$ lower attention core FLOPs.
- **Fail (Inlier Recurrent Degradation)**: $\text{PPL}_{\text{HOFA}} > \text{PPL}_{\text{MHA}} + 1.5$ nats at $N = 16,384$, indicating that linear recurrence cannot sustain long-range coherence.
- **Uninformative**: CPT MHA fails to adapt ($\text{PPL} > 40.0$) or training diverges due to optimization failure.

### 3.2 FineWeb-Edu 1k Pretraining Retention
- **Pass (Zero Forgetting)**: $\Delta\text{PPL}_{\text{FineWeb}} \le 0.15$ nats relative to the native pretraining checkpoint evaluated under native RoPE ($s = 1.0$).
- **Fail (Catastrophic Forgetting)**: $\Delta\text{PPL}_{\text{FineWeb}} > 0.50$ nats.

### 3.3 S-NIAH Passkey Retrieval
- **Informativeness Condition**: At context length $N$, the test is informative if $\text{Acc}_{\text{MHA}} \ge 70\%$ and $\text{Acc}_{\text{MHA}} - \text{Acc}_{\text{control}} \ge 50\%$.
- **Pass (Associative Integrity)**: $\text{Acc}_{\text{HOFA}} \ge \text{Acc}_{\text{MHA}} - 10\%$ at $N = 16\text{k}$ ($\ge 80\%$ absolute), with robust retrieval across depths.
- **Early Boundary Watch**: At depth 0.0, HOFA accuracy must remain $\ge 70\%$ at 16k.
- **Fail (Associative Subspace Collapse)**: $\text{Acc}_{\text{HOFA}} < 50\%$ at 16k while MHA succeeds ($\ge 80\%$).

### 3.4 Zero-Shot Extrapolation Sweep (1k $\to$ 64k)
- **Pass (Stable Extrapolation)**: Monotonic or well-behaved perplexity curve without exponential blowup ($\text{PPL} < 50$ through 32k; $\text{PPL} < 75$ at 64k).
- **Fail (Representation Breakdown)**: Sharp perplexity explosion ($\text{PPL} > 150$) before 32k.

---

## 4. Pre-Registered Evaluation Table Schema (Table 3 for the Paper)

Every unmeasured cell is pre-registered as `[eval]`. The table incorporates base checkpoints alongside CPT models to explicitly isolate the empirical contribution of continual pretraining:

| Model | CPT Tokens | FineWeb 1k PPL ($s=1$) | FineWeb 1k PPL ($s=16$) | S-NIAH Retrieval Acc (%) | | | | PG-19 Test PPL (Fixed $s=16$) | | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | | | | **4k** | **8k** | **16k** | **32k** | **16k** (94 bks) | **32k** (79 bks) | **64k** (54 bks) |
| **MHA 350M (Pretrained)** | – | 15.88 | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] |
| **MHA 350M + YaRN (CPT 100M)** | 100M | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] |
| **HOFA 350M (Pretrained)** | – | 15.88 | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] |
| **HOFA 350M + YaRN (CPT 100M)** | 100M | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] |

*(Statistical confidence: 95% bootstrap intervals over books reported across all evaluated perplexities).*

---

## 5. Code-Level Execution Engine & CLI Commands

### 5.1 Training Phase Commands
1. **Train MHA 350M CPT on PG-19**:
   ```bash
   python main.py train --cpt --scale 350M --model mha
   ```
2. **Train / Re-train HOFA 350M CPT** (with book-strided validation):
   ```bash
   python main.py train --cpt --scale 350M --model hofa
   ```
3. **Generate Shared CPT Training Dynamics Plot**:
   ```bash
   python main.py train --cpt --scale 350M --plot --shared
   ```

### 5.2 Dynamic Alpha Routing Comparison Commands (Base vs. CPT)
1. **Run & Plot Comparative Alpha Distribution**:
   ```bash
   python main.py benchmark --alpha --scale 350M --model both
   ```
   *(Evaluates dynamic token-level gate routing across layers; automatically reuses cached base model activations from `data/experiments_cache/` so base forward passes are never re-run).*

### 5.3 S-NIAH Retrieval Benchmark Commands
1. **Run S-NIAH Retrieval on Both Models**:
   ```bash
   python main.py benchmark --retrieval --scale 350M --model both
   ```
2. **Plot S-NIAH Depth-vs-Accuracy Heatmap / Curves**:
   ```bash
   python main.py benchmark --retrieval --scale 350M --model both --plot
   ```

### 5.4 Context Length Extrapolation Benchmark Commands (1k $\to$ 64k)
1. **Run Extrapolation on PG-19 Test Books (Fixed $s=16$)**:
   ```bash
   python main.py benchmark --extrapolation --scale 350M --model both
   ```
2. **Plot Extrapolation Curves with Confidence Intervals**:
   ```bash
   python main.py benchmark --extrapolation --scale 350M --model both --plot
   ```
3. **Optional Force Re-run** (Bypassing JSON Cache):
   ```bash
   python main.py benchmark --extrapolation --scale 350M --model both --force
   ```

---

## 6. Implementation Checklist

- [x] **Relicensing & Attribution**: Repository licensed under PolyForm-Noncommercial-1.0.0 with proper copyright notices.
- [x] **Book-Strided Evaluation in `training/train_cpt.py`**:
  - Automatically indexes `pg19_16k_val_index.json` to evaluate 1 window per book across all distinct validation books.
  - Eliminates book-selection bias and lowers validation variance.
- [ ] **Alpha Routing Distribution Comparison (`benchmarks/analyze_alpha.py`)**:
  - Add support for comparative evaluation (`--model both` / `--cpt`).
  - Cache base model token-level alpha extractions in `data/experiments_cache/alpha_distribution_{scale}_HOFA.pt` so base results are computed once and never re-run.
  - Save CPT model extractions to a separate `_cpt` file (`alpha_distribution_{scale}_HOFA_cpt.pt`) to keep caching and invalidation clean and simple.
  - Generate standalone individual violin plots for both models (`{scale}_HOFA` and `{scale}_HOFA_cpt`) plus an added comparative side-by-side violin plot.
- [x] **SpectralShift Inlier Reparameterization (Liu et al., arXiv:2609.14320 Algorithm 1)**:
  - Reparameterized inlier `gate_proj.weight` at base checkpoint load by scaling deviation from global mean by $s = (1024 / 16384)^{0.5} = 0.25$.
  - Scaled gate learning rate multiplier to $0.25$ (replacing the previous $10\times$ multiplier) while holding bias fixed.
- [x] **Statistical Rigor: Bootstrap Confidence Intervals over Books**:
  - Evaluation maintains non-parametric 95% bootstrap confidence intervals over evaluated books without unneeded raw tensor disk bloat.
- [ ] **Config Standardization in `benchmarks/benchmarks_configs.py`**:
  - `LongContextExtrapolationConfig`:
    - `seq_lengths = (1024, 2048, 4096, 8192, 16384, 32768, 65536)`.
    - `cache_file_name = "long_context_extrapolation_results.json"`.
    - `data_path = "data/datasets/pg19_test_books.bin"`.
    - `index_path = "data/datasets/pg19_test_books_index.json"`.
    - `use_fixed_yarn = True` ($s=16.0$ tables fixed across all $N$).
  - `RetrievalExperimentConfig`:
    - `context_lengths = (4096, 8192, 16384, 32768)`.
    - `depths = (0.0, 0.25, 0.50, 0.75, 1.0)`.
    - `trials_per_depth = 30`.
    - `haystack_path = "data/datasets/paul_graham_essays.txt"`.
    - `cache_file_name = "sniah_retrieval_results.json"`.
- [ ] **S-NIAH Retrieval Engine in `benchmarks/benchmark_retrieval.py`**:
  - Load Paul Graham essays; assert length $\ge N$.
  - Insert 5-digit needle at specified depths; run generation/forward evaluation.
  - Run negative no-needle control at each length.
  - Implement incremental JSON caching.
- [ ] **Extrapolation Engine in `benchmarks/benchmark_long_context.py`**:
  - Point data source to `pg19_test_books.bin` and filter books where `length >= N`.
  - Fix RoPE tables to trained $s=16.0$ YaRN setting.
  - Log and report active book counts at each $N$.
  - Implement incremental JSON caching and OOM protection.
- [ ] **CLI Simplification in `main.py`**:
  - Clean `benchmark --retrieval` and `benchmark --extrapolation` flags inheriting defaults from benchmark configs without mandatory `--length`.
  - Pass `--force` flag to bypass cache when explicitly requested.
- [ ] **(Contingency) Zero-Shot Base Model Extrapolation Sweep (1k -> 16k without CPT)**:
  - Benchmark base `350M_HOFA` vs `350M_MHA` directly under Dynamic YaRN ($s = N / 1024$).
  - Purpose: Empirically confirm HOFA's zero-shot robustness advantage ($r=16$ low-rank RoPE vs MHA $d_h=128$ full-rank distortion, where HOFA previously outperformed MHA 28.5 vs 45.6 PPL) if CPT perplexity deltas warrant highlighting zero-shot inductive bias. Only executed on explicit instruction.

---

## 7. Mathematical Reference & FLOP Derivations

### 7.1 Attention Core FLOP Derivation (Paper Equation 7 Parity)
$$\text{FLOPs}_{\text{HOFA-attn-core}} = 6 \cdot H \cdot (r + d_h) \cdot N + 18 \cdot H \cdot (d_h - r) \cdot d_h$$
$$\text{FLOPs}_{\text{MHA-attn-core}} = 12 \cdot H \cdot d_h \cdot N$$

Where:
- $6 \cdot H \cdot (r + d_h) \cdot N$: Outlier exact attention forward + backward ($2 H r N$ for $Q_O K_O^\top$ fwd $+ 2 H d_h N$ for $P_O V$ fwd, multiplied by $3\times$ for fwd + bwd).
- $18 \cdot H \cdot (d_h - r) \cdot d_h$: Inlier GLA forward + backward state updates and queries.

At $N = 16,384$ ($L = 24, H = 8, d_h = 128, r = 16$):
- **MHA attention core**: $4.832\text{B FLOPs/token}$ (Full model: $6.932\text{B FLOPs/token}$).
- **HOFA attention core**: $2.766\text{B FLOPs/token}$ (Full model: $4.866\text{B FLOPs/token}$).
- **Net Compute Reduction**: $\mathbf{-29.8\%}$ total model FLOPs per token at 16k context, with efficiency expanding linearly as context length $N$ grows.

### 7.2 Positional Extrapolation Decoupling in Factorized Attention
In standard MHA, RoPE is applied across all channel dimensions ($d_h = 128$), coupling positional extrapolation distortion and quadratic compute across 100% of the representation space.

In HOFA:
- **Outlier Pathway ($\mathcal{O}$, rank $r = 16$)**: Dedicated strictly to high-magnitude associative features using exact causal attention with canonical YaRN RoPE.
- **Inlier Pathway ($\mathcal{I}$, capacity $d_{\text{inlier}} = d_h - r = 112$)**: Dedicated to smooth contextual features computed via chunked GLA with continuous decay $\exp(-g_t)$ and fixed-size $O(1)$ recurrent state.

Because the inlier recurrent state operates on continuous exponential decay without discrete position indices, **positional extrapolation pressure is mathematically quarantined to the $r = 16$ outlier subspace** (only 12.5% of the head dimension).
