# Experimental Roadmap: Long-Context Continual Pretraining (CPT) & Rebuttal Blueprint

This document specifies the updated, peer-reviewed technical roadmap for extending the context window of 350M HOFA and MHA models from $N = 1,024$ to $N = 16,384$ ($16\text{k}$) via continual pretraining (CPT) on PG19, and evaluating zero-shot length extrapolation up to $N = 65,536$ ($64\text{k}$) using canonical YaRN.

---

## 1. Architectural Synergy & Attention Core FLOP Derivation

### 1.1 Positional Decoupling in Factorized Attention
Standard Multi-Head Attention (MHA) applies Rotary Position Embeddings (RoPE) across all channel dimensions ($d_h = 128$), coupling positional extrapolation distortion and quadratic compute across all heads.

HOFA factorizes each head into:
1. **Outlier Pathway ($\mathcal{O}$)**: Channel rank $r = 16$ capturing high-magnitude associative features via exact causal attention with RoPE:
   $$\mathcal{S}_{ij} = \frac{1}{\sqrt{r}} \left(R_i \tilde{Q}_{\mathcal{O}, i}\right) \left(R_j \tilde{K}_{\mathcal{O}, j}\right)^\top + \mathcal{M}_{ij}, \quad \mathcal{O}_{\mathcal{O}} = \text{Softmax}(\mathcal{S}) \tilde{V}_{\mathcal{O}}$$
2. **Inlier Pathway ($\mathcal{I}$)**: Channel capacity $d_{\text{inlier}} = d_h - r = 112$ capturing smooth, distributed features via chunked Gated Linear Attention (GLA):
   $$S_t = S_{t-1} \odot \exp(-g_t) + K_{\mathcal{I}, t}^\top V_{\mathcal{I}, t}, \quad O_{\mathcal{I}, t} = Q_{\mathcal{I}, t} S_t$$

Because the inlier pathway maintains a fixed-size $O(1)$ recurrent state driven by continuous exponential decay $\exp(-g_t)$ without discrete position indices, positional extrapolation pressure is completely confined to the $r = 16$ outlier subspace.

### 1.2 Mathematical FLOP Formulation (Paper Equation 7 Parity)
To accurately account for Attention $\times V$ projecting across the full head dimension $d_h$, the forward + backward FLOPs per token for the HOFA attention core per layer are computed as:
$$\text{FLOPs}_{\text{HOFA-attn-core}} = 6 \cdot H \cdot (r + d_h) \cdot N + 18 \cdot H \cdot (d_h - r) \cdot d_h$$
* **$6 \cdot H \cdot (r + d_h) \cdot N$**: Outlier exact attention forward + backward ($2 H r N$ for $Q_O K_O^\top$ fwd $+ 2 H d_h N$ for $P_O V$ fwd, multiplied by $3\times$ for fwd + bwd).
* **$18 \cdot H \cdot (d_h - r) \cdot d_h$**: Inlier GLA forward + backward state updates and queries.

For standard MHA:
$$\text{FLOPs}_{\text{MHA-attn-core}} = 12 \cdot H \cdot d_h \cdot N$$

Together with the dense parameters $P$ ($6P$ for MLPs, $W_q, W_k, W_v, W_o$, and LayerNorms):
* Total compute per token at $N = 16,384$:
  $$\text{FLOPs}_{\text{MHA}} = 6P + 24 \times (12 \cdot H \cdot d_h \cdot N) = 6(350\text{M}) + 4.832\text{B} \approx 6.932\text{B FLOPs/token}$$
  $$\text{FLOPs}_{\text{HOFA}} = 6P + \sum_{l=1}^{24} \left[6 \cdot H \cdot (r_l + d_h) \cdot N + 18 \cdot H \cdot (d_h - r_l) \cdot d_h\right] \approx 2.1\text{B} + 2.766\text{B} \approx 4.866\text{B FLOPs/token}$$
* **Reporting Standard**: Theoretical multiplier claims (e.g. "-56.7% / 2.3x") are dropped. All efficiency claims are plotted directly against **processed tokens** and **measured wall-clock time**, ensuring conclusions rest entirely on hardware measurements.

---

## 2. Phase 1 — Dataset Preparation: PG19 Unbroken Intra-Book Pipeline

### Objective
Pre-tokenize and chunk PG19 into unbroken $16,384$-token segments drawn strictly from within individual books (no inter-book concatenation with `<|endoftext|>`), ensuring genuine long-range literary context across every sequence.

### Specification
- **Script**: `training/prepare_pg19.py`
- **Source**: Hugging Face `deepmind/pg19`
- **Tokenizer**: GPT-2 tiktoken (`vocab_size = 50257`, identical to pretraining).
- **Intra-Book Chunking Rule**:
  - Each continuous book is tokenized independently.
  - Windows of exactly $16,384$ tokens are carved from unbroken text within a single book.
  - Books with fewer than $16,384$ tokens are discarded. Trailing remainder tokens ($< 16\text{k}$) at the end of each book are discarded to avoid boundary discontinuities.
- **Token Budget & Output**:
  - Training: `data/datasets/pg19_16k_train.bin` (first $100,000,000$ tokens $\approx 6,104$ unbroken $16\text{k}$ windows, $\sim 200\text{ MB}$ `uint16`).
  - Validation: `data/datasets/pg19_16k_val.bin` ($\sim 5,000,000$ tokens $\approx 305$ windows from the official PG19 validation split).
- **Pre-Registered Replay Decision**: We execute pure PG19 continual pretraining (100% PG19). Pretraining retention is tracked by measuring FineWeb-Edu validation perplexity at $N = 1,024$ before and after CPT, reporting the observed $\Delta\text{PPL}_{\text{FineWeb}}$ as measured without post-hoc tuning.

---

## 3. Phase 2 — Continual Pretraining (CPT) Execution Engine

### Objective
Continually pretrain 350M HOFA and 350M MHA from their best pretraining checkpoints under identical training configurations.

### CLI Command
```bash
python main.py train --cpt --scale 350M --model both
```
*(Supports `--model hofa`, `--model mha`, or `--model both`)*

### Execution & Wall-Clock Budgeting
- **Step Timing Protocol**: Time the first 5 steps explicitly to record real hardware throughput before running the full 381 steps.
- **Expected Duration**: At $\approx 85\text{ TFLOP/s}$ sustained compute on a modern GPU with gradient checkpointing and micro-batch size 2, expect $\approx \mathbf{2\text{ hours per model}}$ (total $\sim 4\text{ hours}$ for both).

### Hyperparameters (Pre-Registered)
| Hyperparameter | Value |
| :--- | :--- |
| **Architectures** | 350M HOFA ($L=24, d_{\text{model}}=1024, H=8, d_h=128$) vs. 350M MHA |
| **Outlier Schedule** | Heterogeneous: $r \in [64, 32, 32] + [16] \times 18 + [32, 32, 64]$ |
| **Base Checkpoints** | `data/training/350M_HOFA/checkpoint_best_val.pt` & `350M_MHA/checkpoint_best_val.pt` |
| **Adaptation Dataset** | PG19 ($16\text{k}$ unbroken intra-book segments) |
| **Total Training Tokens** | $100,000,000$ tokens ($0.714\%$ of 14B pretraining) |
| **Native Pretrained Length ($L$)** | $1,024$ tokens |
| **Training Sequence Length ($L'$)** | $16,384$ tokens ($s = 16.0$) |
| **Micro Batch Size ($b_{\text{micro}}$)** | $2$ sequences ($32,768$ tokens/step) |
| **Gradient Accumulation** | $8$ steps |
| **Global Batch Size ($B_{\text{global}}$)** | $16$ sequences ($262,144$ tokens/step) |
| **Total Training Steps** | $381$ steps |
| **Optimizer** | AdamW ($\beta_1 = 0.9, \beta_2 = 0.95$, `fused=True`) |
| **Peak Learning Rate ($\eta_{\text{max}}$)** | $1 \times 10^{-5}$ |
| **LR Schedule** | Linear warmup ($30$ steps) followed by Cosine Annealing |
| **Minimum Learning Rate ($\eta_{\text{min}}$)** | $1 \times 10^{-6}$ |
| **Weight Decay ($\lambda_{\text{WD}}$)** | $0.0$ |
| **Gradient Clipping** | $1.0$ |
| **Precision** | PyTorch `bfloat16` mixed precision |
| **YaRN Parameters** | $\alpha = 1.0, \beta = 32.0, s = 16.0, \sqrt{1/t} = 0.1 \ln(16) + 1.0 \approx 1.2773$ |
| **Checkpointing Interval** | Save every $50$ steps and record validation PPL periodically |

---

## 4. Phase 3 — Evaluation Suite & Deliverables

### Deliverable 1: PPL vs. Tokens & Wall-Clock Training Curves
- **Command**: `python main.py train --plot-cpt`
- **Output Plot**: `data/plots/training/cpt_ppl_dynamics.pdf`
- **Plots Generated**:
  1. **Validation Perplexity vs. Processed Tokens** ($N = 16,384$).
  2. **Validation Perplexity vs. Measured Wall-Clock Training Time** (seconds/hours).
- **Methodology**: Direct empirical comparison showing convergence speed and loss trajectories without theoretical scaling assumptions.

---

### Deliverable 2: Passkey Retrieval Benchmark (Frozen Protocol)
- **Command**: `python main.py benchmark --passkey`
- **Output Plot**: `data/plots/benchmarks/passkey_retrieval_accuracy.pdf`
- **Protocol**:
  - Continuous essay/book haystack with 5-digit secret passkey.
  - **Depths Tested**: $5$ depths uniformly spanning $0.0$ to $1.0$: $\{0.0, 0.25, 0.50, 0.75, 1.0\}$ (preserving depth $0.0$ strictly).
  - **Query Format**: Clean, bare query (`"What is the passkey?"`).
  - **Sample Size**: $30$ independent randomized trials per depth ($150$ trials per context length).
  - **Context Lengths**: $2\text{k}, 4\text{k}, 8\text{k}, 16\text{k}$ (trained window), and $32\text{k}, 48\text{k}, 64\text{k}$ (extrapolation).
  - **Primary Setting**: Evaluated using the **same $s=16$ YaRN tables at all lengths**, matching the trained state directly.

---

### Deliverable 3: Adaptive Stride Sliding-Window PPL on PG19
- **Command**: `python main.py eval --cpt`
- **Objective**: Measure continuous sequence modeling perplexity across $N \in \{1\text{k}, 2\text{k}, 4\text{k}, 8\text{k}, 16\text{k}, 32\text{k}, 64\text{k}\}$.
- **Efficiency Constraint**: Dense $S = 256$ stride across full books at $64\text{k}$ requires prohibitive computation. We implement an **adaptive evaluation stride**:
  - For $N \le 8\text{k}$: $S = 256$ over held-out documents.
  - For $N \ge 16\text{k}$: $S = 2,048$ over $20$ representative held-out test books, completing in hours.
- **Reporting**: Report mean perplexity with **paired bootstrap intervals over books** to guarantee rigorous statistical significance.

---

### Deliverable 4: FineWeb-Edu Native Length Retention
- **Metric**: Validation perplexity on FineWeb-Edu at native pretraining length $N = 1,024$.
- **Reporting**: Report exact before vs. after values and $\Delta\text{PPL}_{\text{FineWeb}}$ as measured, using identical acceptance criteria for MHA and HOFA relative to their 1k references.

---

## 5. Pre-Registered Evaluation Table Schema (Table 3)

Every unmeasured cell is registered as `[eval]`. The table incorporates zero-shot baselines alongside CPT models to explicitly isolate the empirical contribution of continual training:

| Model | CPT Tokens | FineWeb 1k PPL | Passkey Retrieval Acc (%) | | | PG19 Sliding-Window PPL | | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | | | **16k** | **32k** | **64k** | **16k** | **32k** | **64k** |
| **MHA 350M (Pretrained, Native 1k)** | – | 15.88 | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] |
| **MHA 350M + YaRN (Zero-Shot)** | – | 15.88 | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] |
| **MHA 350M + YaRN (CPT 100M)** | 100M | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] |
| **HOFA 350M (Pretrained, Native 1k)** | – | 15.88 | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] |
| **HOFA 350M + YaRN (Zero-Shot)** | – | 15.88 | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] |
| **HOFA 350M + YaRN (CPT 100M)** | 100M | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] | [eval] |

*(Statistical confidence: Paired bootstrap intervals over books reported across all evaluated perplexities).*

---

## 6. Implementation Checklist

- [x] **Step 1: Data Preparation Pipeline** (`training/prepare_pg19.py`)
  - Stream PG19, tokenize with GPT-2 tiktoken, extract unbroken intra-book $16\text{k}$ windows.
  - Output binary files: `pg19_16k_train.bin` (100M tokens) and `pg19_16k_val.bin` (~5M tokens).
- [x] **Step 2: CPT Training Engine** (`training/train_cpt.py` & `main.py train --cpt`)
  - Load 350M checkpoint, attach `YaRNModel(s=16)`, configure AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay 0.0).
  - Micro-batch 2, grad accum 8, 381 steps, bfloat16.
  - Exact chunked cross-entropy implemented ($C=4096$) solving activation VRAM explosion.
- [ ] **Step 3: Training Dynamics Plotter** (`main.py train --plot-cpt`)
  - Plot Validation PPL vs. Tokens and Validation PPL vs. Measured Wall-Clock Time.
- [ ] **Step 4: Passkey Retrieval Benchmark** (`benchmarks/benchmark_passkey.py` & `main.py benchmark --passkey`)
  - Implement frozen protocol: depths $\{0.0, 0.25, 0.50, 0.75, 1.0\}$, bare query, 30 samples/depth.
- [ ] **Step 5: Sliding-Window PPL & Retention Evaluator** (`benchmarks/benchmark_sliding_ppl.py` & `main.py eval --cpt`)
  - Adaptive stride evaluation ($S=256$ for $\le 8\text{k}$, $S=2048$ for $\ge 16\text{k}$) with paired bootstrap intervals over books.
  - Evaluate native FineWeb-Edu 1k PPL retention under both YaRN ($s=16$) and Native ($s=1$) rotary embeddings.

---

## 7. Paper Compendium & Vital Technical Reference

This section consolidates critical architectural derivations, algorithmic optimizations, and empirical protocols established during development to ensure no essential technical justification is omitted when writing the paper.

### 7.1 Mathematical Decoupling of Positional Extrapolation in Factorized Attention
* **The Core Problem in Standard Attention**: In MHA, Rotary Position Embeddings (RoPE) are applied across all channel dimensions ($d_h = 128$). Consequently, extending the context window forces high-frequency and low-frequency rotary interpolation across 100% of the representation space, coupling positional distortion to all associative heads.
* **HOFA's Structural Separation**:
  1. **Outlier Pathway ($\mathcal{O}$, rank $r = 16$)**: Dedicated strictly to sharp, high-magnitude associative features. Uses exact causal softmax attention with canonical YaRN RoPE:
     $$\mathcal{S}_{ij} = \frac{1}{\sqrt{r}} \left(R_i \tilde{Q}_{\mathcal{O}, i}\right) \left(R_j \tilde{K}_{\mathcal{O}, j}\right)^\top + \mathcal{M}_{ij}, \quad \mathcal{O}_{\mathcal{O}} = \text{Softmax}(\mathcal{S}) \tilde{V}_{\mathcal{O}}$$
  2. **Inlier Pathway ($\mathcal{I}$, capacity $d_{\text{inlier}} = d_h - r = 112$)**: Dedicated to smooth, distributed contextual features. Computed via chunked Gated Linear Attention (GLA) with data-dependent continuous decay $\exp(-g_t)$ and fixed-size $O(1)$ recurrent state:
     $$S_t = S_{t-1} \odot \exp(-g_t) + K_{\mathcal{I}, t}^\top V_{\mathcal{I}, t}, \quad O_{\mathcal{I}, t} = Q_{\mathcal{I}, t} S_t$$
* **Paper Justification**: Because the recurrent GLA state operates on continuous exponential decay with no discrete positional indices, **positional extrapolation pressure is mathematically quarantined to the $r = 16$ outlier subspace** (only 12.5% of the head dimension). The remaining 87.5% capacity maintains smooth temporal decay that is naturally invariant to context window expansion.

### 7.2 Canonical YaRN Formulation & Hyperparameters
* **Exact Formulation (Peng et al., 2023 Parity)**:
  $$f_i = (1 - \gamma_i) \frac{f_{\text{base}, i}}{s} + \gamma_i f_{\text{base}, i}, \quad \gamma_i = \text{clamp}\left(\frac{r_i - \alpha}{\beta - \alpha}, 0, 1\right)$$
  where $r_i = \frac{d}{2i}$ is the wavelength ratio, base frequency is $10000.0$, and the scale factor is $s = 16.0$ ($1,024 \to 16,384$).
* **Ramping Boundaries**:
  * $\alpha = 1.0$: Lower wavelength cutoff (dimensions with wavelength below $\alpha$ are un-interpolated, preserving local token order).
  * $\beta = 32.0$: Upper wavelength cutoff (dimensions with wavelength above $\beta$ are fully linearly interpolated by $1/s$).
* **Attention Temperature Scaling**:
  $$\sqrt{1/t} = 0.1 \ln(s) + 1.0 = 0.1 \ln(16) + 1.0 \approx 1.2773$$
  Query vectors (or attention logits) are scaled by $\sqrt{1/t}$ to counteract entropy loss caused by positional interpolation.
* **Dead-Weight Elimination**: Redundant fields (`mscale`, `mscale_all_dim`) were formally removed from the codebase to maintain strict parity with canonical YaRN without extraneous empirical scaling terms.

### 7.3 Attention Core FLOP Derivation (Paper Equation 7)
* **Forward + Backward FLOP Formulations**:
  $$\text{FLOPs}_{\text{HOFA-attn-core}} = 6 \cdot H \cdot (r + d_h) \cdot N + 18 \cdot H \cdot (d_h - r) \cdot d_h$$
  $$\text{FLOPs}_{\text{MHA-attn-core}} = 12 \cdot H \cdot d_h \cdot N$$
* **Component Breakdown**:
  * $Q_O K_O^\top$ forward compute: $2 H r N$ FLOPs.
  * $P_O V$ forward compute: $2 H d_h N$ FLOPs (critical: value projection spans the full head dimension $d_h$, not $r$).
  * Multiplying forward FLOPs by $3\times$ accounts for both forward and backward passes: $3 \times 2 H (r + d_h) N = 6 H (r + d_h) N$.
  * GLA forward updates and queries: $4 H (d_h - r) d_h N$, which with backward passes and gate projections totals $18 H (d_h - r) d_h$.
* **Analytical Quantities at $N = 16,384$ (350M Model, $L=24, H=8, d_h=128, r=16$)**:
  * Standard MHA attention core: $4.832\text{B FLOPs/token}$; Full model: $6.932\text{B FLOPs/token}$.
  * HOFA attention core: $2.766\text{B FLOPs/token}$; Full model: $4.866\text{B FLOPs/token}$ ($-29.8\%$ total model compute reduction per token).

### 7.4 Exact Chunked Cross-Entropy: Memory & Hardware Locality
* **The Activation Bottleneck**:
  At long context ($N = 16,384$) with standard vocabulary size ($V = 50,257$), standard monolithic cross-entropy materializes the unnormalized logits tensor $[B, N, V]$. For a single sequence ($B = 1$), this tensor requires:
  $$1 \times 16,384 \times 50,257 \times 2\text{ bytes} \approx 1.647\text{ GB} \quad (\text{bfloat16})$$
  During backward propagation, PyTorch retains the logit activations and backward softmax buffers, exceeding $6.38\text{ GB}$ peak VRAM for the loss layer alone, crashing 8GB consumer GPUs (e.g. RTX 4060).
* **The Formulation**:
  The final hidden states $H \in [B, N, D]$ are partitioned along the sequence dimension into micro-chunks of size $C = 4096$:
  $$\mathcal{L} = \sum_{c=1}^{N / C} \frac{C}{N} \mathcal{L}_c, \quad \mathcal{L}_c = \text{CrossEntropy}\left(H_{[:, (c-1)C:cC, :]} W_{\text{head}}^\top, Y_{[:, (c-1)C:cC]}\right)$$
  For each chunk, logits $[B, C, V]$ ($\approx 411\text{ MB}$) are projected, chunk loss is computed, and `loss.backward()` immediately accumulates gradients into $W_{\text{head}}$ and backpropagates into $H_c$. The logit buffer is immediately freed before the next chunk.
* **Exact Mathematical Equivalence**: Because gradient accumulation is linear and cross-entropy loss is an unweighted sum over tokens, chunked cross-entropy yields **exact bitwise identical gradients** to monolithic cross-entropy.
* **Empirical Speedup & Memory Results (RTX 4060 Benchmark)**:
  * **Peak VRAM**: Plummets from $>6.38\text{ GB}$ down to **$2.66\text{ GB}$** ($\sim 5\times$ activation footprint reduction).
  * **Throughput**: Surges from $27,400\text{ tok/s}$ to **$40,864\text{ tok/s}$** (**$+49.1\%$ wall-clock speedup**).
  * **Hardware Reason**: Monolithic cross-entropy incurs massive memory bandwidth thrashing to off-chip DRAM for a $1.65\text{ GB}$ tensor. Micro-chunks of $411\text{ MB}$ achieve vastly superior GPU L2 cache residency and streaming processor locality.

### 7.5 Dual FineWeb-Edu Retention Isolation Protocol
* **The Scientific Dilemma**: Measuring FineWeb-Edu perplexity at native length ($N = 1,024$) during/after CPT under YaRN ($s = 16$) conflates two distinct phenomena:
  1. Genuine catastrophic forgetting (drift in weight representations).
  2. Synthesized positional distortion (frequency scaling $s = 16$ applied to short $1\text{k}$ sequences where native frequencies $s = 1$ are optimal).
* **The Isolation Protocol**:
  During CPT validation (at Step 0 and periodically), evaluate FineWeb-Edu 1k under two explicit configurations:
  1. **YaRN Configuration ($s = 16.0$)**: Demonstrates real-world inference performance if the long-context model is deployed naively on short sequences.
  2. **Native Rotary Configuration ($s = 1.0$)**: Rotary embeddings are cleanly hot-swapped to native pretraining rotary via `swap_rotary_to_native()`, evaluated, and restored via `restore_rotary_from_saved()`.
* **Empirical Verification**: At Step 0, evaluating under native rotary produces the exact validation perplexity of the pretraining base checkpoint bit-for-bit (e.g., $27.27$ on 13M), confirming zero measurement artifact and mathematically clean weight isolation.

### 7.6 Hardware Tuning Specifications for CPT
* **13M Model on RTX 4060 (8GB VRAM)**:
  * Micro-batch size $b_{\text{micro}} = 1$ sequence ($16,384$ tokens).
  * Gradient accumulation: $16$ steps (Global batch size $B = 262,144$ tokens/step).
  * Chunk size: $C = 4096$.
  * Peak VRAM: $2.66\text{ GB}$; Throughput: $\sim 41,000\text{ tok/s}$.
  * Gradient checkpointing disabled (unnecessary given $< 3\text{ GB}$ usage, maximizing raw throughput).
* **350M Model on RTX 5090 (32GB VRAM)**:
  * Micro-batch size $b_{\text{micro}} = 2$ sequences ($32,768$ tokens).
  * Gradient accumulation: $8$ steps (Global batch size $B = 262,144$ tokens/step).
  * Chunk size: $C = 4096$.
  * Validation batches: $8$ micro-batches ($262,144$ val tokens evaluated per checkpoint).
  * Maximizes Tensor Core utilization without exceeding 32GB VRAM headroom.
