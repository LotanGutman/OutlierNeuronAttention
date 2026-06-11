
# instructions #1:

## Project Goal
Produce a complete, ICLR‑ready paper on **Hybrid Outlier‑Factorized Attention (HOFA)** – an intra‑head attention mechanism that routes a few outlier dimensions through exact softmax (FlashAttention) and the remaining inlier dimensions through Gated Linear Attention (GLA). The paper must demonstrate:
1. **Injectivity / capacity:** HOFA matches MHA on dense associative recall (HD‑MQAR) while pure GLA and Gated DeltaNet fail at high densities.
2. **Efficiency:** HOFA crosses below FlashAttention latency at long contexts (prefill) and maintains near‑flat decoding latency with a drastically smaller KV‑cache.
3. **Language modelling quality:** HOFA matches MHA perplexity and zero‑shot accuracy at 125M scale.

## Current Status
- **Architecture:** Fully implemented in `src/HybridOutlierFactorizedAttention.py`. Training/prefill uses a custom Triton kernel (`src/chunk_gla_inlier.py`) that computes the GLA recurrence. Decoding uses a lightweight PyTorch step.
- **Training recipe:** Gate bias initialised to −1.0, gate projection weight to zero, stabilising early training and allowing the outlier pathway to learn first. Model scale for synthetic benchmarks: `d_model=256, heads=4, layers=4`.
- **MQAR results:** Density 4 → 100% acc. Density 8 → 100% acc after grokking (~step 4000). Full HD‑MQAR sweep not yet completed.
- **Profiling results (latest, *unacceptable*):**  
  The custom Triton kernel is **far too slow**. At 2048 tokens, MHA takes 0.675 ms while HOFA takes **36 ms** – a 50× slowdown. The gap persists across all lengths. The kernel uses a sequential loop over tokens within each chunk, with atomic operations in the backward pass, making it non‑competitive.
- **LaTeX blueprint:** `main.tex` (the file you have) is the master document. It already contains the full mathematical derivation, motivation, experimental plan, and placeholder figures. All results must be inserted into this document.

## Critical Tasks (in order of priority)

### 1. Fix the GLA Triton kernel to achieve competitive performance
The current kernel `chunk_gla_inlier.py` must be either **rewritten** or **replaced** so that HOFA becomes faster than MHA at long contexts (target crossover ~100k tokens).  
You have two viable paths:

#### Path A – Use the official `flash-linear-attention` (fla) library
- Replace the custom kernel with `fla.ops.gla.chunk_gla` (which is highly optimised, uses intra‑chunk parallel scan and tiling).
- The gate is already computed as a scalar per head per token; convert it to the `log_gamma` format expected by `chunk_gla` (i.e., `F.logsigmoid(-gate_logits)`).
- Ensure the inlier features are zero‑padded from `j` to `d_head` before passing to the kernel.
- This is the fastest route to a publication‑ready profiling figure.
- **Note:** This library requires Triton and a compatible GPU; your RTX 4060 (Ada Lovelace) is supported.

#### Path B – Optimise the custom kernel
If you must keep full control, rewrite the kernel with:
- Intra‑chunk parallel scan (cumprod / cumsum) to remove the per‑token loop.
- Tiling over the feature dimension `j` and head dimension `d_head` using Triton’s `tl.dot` for the matmul.
- Use shared‑memory tiling for the recurrent state to reduce global memory traffic.
- Avoid atomics in the backward pass by computing gradients in tiles.
- This is a substantial engineering effort (~400‑800 lines of Triton) but yields a fully self‑contained implementation.

**Immediate action:** Profile the `fla` GLA kernel alone (on your inlier features) to confirm it runs faster. Then integrate it into the HOFA forward pass and re‑run the profiling suite.

### 2. Complete the HD‑MQAR sweep
- Model config: `d_model=256, heads=4, layers=4, r=8, chunk_size=32, gate_proj bias=-1.0, weight=0`.
- Densities to test: 4, 8, 16, 32, 64, 128, 256, 512, 1024.
- For each density, train HOFA, MHA, Gated DeltaNet, and GLA from scratch (10k steps).
- Record final validation accuracy.
- **Expected outcome:** HOFA and MHA maintain >95% accuracy for all densities; GLA and Gated DeltaNet collapse beyond a critical density (likely around 16–32 pairs).
- Generate a line plot: density on x‑axis, accuracy on y‑axis, with one curve per model.

### 3. Run the S‑NIAH test (secondary)
- Use the same model scale.
- Sequence lengths: 8k, 16k, 32k, 64k, 128k, 256k, 512k (or as high as memory allows).
- Needle depths: 0%, 25%, 50%, 75%, 100%.
- Train one model per sequence length (with random depths), then evaluate on fixed depths.
- **Expected outcome:** HOFA and MHA both achieve near‑100% retrieval at all depths/lengths; Gated DeltaNet degrades for early needles at extreme lengths.
- Plot heatmaps: length vs depth, colour = accuracy.

### 4. Ablation studies (Tier 2)
Once the core claims are established, run the following to defend design choices:

#### 4.1 Routing mechanism
- Train three variants of HOFA on HD‑MQAR (density 64):
  - Product‑norm routing (our method)
  - Random routing (fixed after initialisation)
  - Static routing (computed once, never refreshed)
- Compare final accuracy. Product‑norm should significantly outperform the others.

#### 4.2 Gate bias initialisation
- Train HOFA with gate biases: -2.0, -1.0, 0.0, 1.0 on HD‑MQAR (density 8).
- Show that a negative bias is crucial for grokking; zero or positive bias fails to converge.

#### 4.3 Intra‑layer vs. inter‑layer hybrid
- Construct an inter‑layer baseline: at 60M scale, odd layers use MHA, even layers use pure GLA (parameter‑matched to HOFA).
- Train both on 2B tokens of FineWeb‑Edu; compare validation perplexity and MQAR recall (density 64).
- HOFA should achieve better recall and comparable perplexity.

### 5. 125M pretraining (optional, if time/cloud budget allows)
- Train a 125M HOFA model on 5B tokens of FineWeb‑Edu (cloud RTX 4090, ~$50).
- Train MHA and GLA baselines at same scale.
- Evaluate zero‑shot on HellaSwag, PIQA, WinoGrande, SciQ.
- **Expected:** HOFA within 1‑2% of MHA, significantly better than GLA.

### 6. Update the LaTeX document
- All figures, tables, and results must be placed into the existing `main.tex`.
- The document already contains placeholder figures; replace them with the generated PDFs.
- Ensure all experimental claims in the `What I Need to Prove` section are backed by the generated evidence.
- The final paper should be self‑contained, with no missing references or placeholder text.

## Environment & Resources
- **Local GPU:** NVIDIA RTX 4060 8 GB, CUDA 12.1, PyTorch 2.5+, Triton 3.2+, `flash-linear-attention` library installed.
- **Cloud GPU:** RTX 4090 on Vast.ai (~$0.40/hr). Maximum budget for cloud: $100.
- **Python environment:** The existing `src/`, `benchmarks/` directories contain all modules; the agent can add new files but must keep the existing interface for `HybridOutlierFactorizedAttention` so the benchmark harness works unchanged.
- **Profiling harness:** `benchmarks/profile.py` measures forward‑pass time and VRAM using random inputs; it expects the attention module to have an `forward` method and be in eval mode.

## Success Criteria
- [ ] HOFA forward pass latency crosses below MHA at sequence length ≤ 128k tokens (prefill).
- [ ] HOFA decoding TPOT is nearly flat up to 128k tokens, and KV‑cache memory is at least 4× smaller than MHA.
- [ ] HD‑MQAR accuracy of HOFA (r=8) matches MHA (≥95%) at all densities up to 1024, while GLA and Gated DeltaNet drop significantly.
- [ ] S‑NIAH heatmaps show HOFA and MHA solid green; Gated DeltaNet fails for early needles.
- [ ] Ablation studies demonstrate the importance of product‑norm routing and negative gate bias.
- [ ] The LaTeX document compiles without errors and contains all figures and results.

## Working Principle
- The agent should work iteratively: start with the performance fix (Task 1) because it unblocks both the profiling figure and the later large‑scale experiments.
- After each major step, update the LaTeX with the latest results and commit to version control.
- For cloud runs, the agent should first estimate costs and seek approval before launching.
- The agent should never delete the existing benchmark harness or change the `HybridOutlierFactorizedAttention` interface without ensuring backward compatibility for the MQAR training script.
- If a task is impossible due to hardware limits, report the limitation and suggest an alternative (e.g., reduce sequence length).

**Begin with Task 1 – fix the GLA kernel performance.**  
Use Path A (fla integration) as the first attempt. Once the profiling crossover is restored, proceed to Task 2 (HD‑MQAR sweep). Do not stop until all success criteria are met.


# instructions #2:
# TASK: Optimize Triton GLA Kernel for Peak GPU Parallelism

## Objective
The current `chunk_gla_inlier.py` Triton implementation correctly scales memory linearly (O(N)), successfully handling non-power-of-2 dimensions ($j=56$) without crashing. However, the latency is astronomically slow (4.1 seconds at 131k tokens vs. MHA's 0.5 seconds). 

Your goal is to completely rewrite the `chunk_gla_fwd_kernel` (and its backward pass) inside `src/chunk_gla_inlier.py` to achieve a forward pass latency that is faster than MHA at sequence lengths >= 65k tokens. 

You must continuously benchmark using `python benchmarks/profile.py` and iterate on the kernel design until the performance target is met. **DO NOT STOP until HOFA is faster than MHA.**

## 1. The Core Performance Bug
The current kernel processes tokens sequentially within a chunk:
```python
# THIS KILLS GPU PERFORMANCE:
for t_in_chunk in range(chunk_size):
    # scalar loads and accumulations
Triton is designed to compile matrix operations to GPU Tensor Cores. By looping over t and doing vector-matrix operations, you are forcing the GPU to act like a sequential CPU.2. The Solution: Parallel Block OperationsYou must rewrite the chunk_gla_fwd_kernel to compute the entire chunk_size (e.g., 64 tokens) in parallel using blocked matrix multiplications (tl.dot) and prefix sums (tl.cumsum).Mathematical Formulation for Parallel Intra-Chunk ComputationGiven a chunk of size $C$, let $Q_c \in \mathbb{R}^{C \times j}$, $K_c \in \mathbb{R}^{C \times j}$, $V_c \in \mathbb{R}^{C \times d}$, and $g_c \in \mathbb{R}^C$.Parallel Cumulative Decay:Pythong_cumsum = tl.cumsum(g_c, axis=0)
Parallel Intra-Chunk Attention Mask:Create a $C \times C$ causal mask where the entry $(i, k)$ is $\exp(g\_cumsum[i] - g\_cumsum[k])$ if $i \ge k$, else $0$.Pythonoffs_c = tl.arange(0, C)
diff = g_cumsum[:, None] - g_cumsum[None, :]
mask = tl.exp(diff) * tl.where(offs_c[:, None] >= offs_c[None, :], 1.0, 0.0)
Parallel Output Computation:Python# 1. Q @ K^T
attn = tl.dot(Q_c, tl.trans(K_c)) # [C, j] @ [j, C] -> [C, C]
# 2. Apply Mask
attn = (attn * mask).to(tl.float16)
# 3. Attn @ V
Y_intra = tl.dot(attn, V_c) # [C, C] @ [C, d] -> [C, d]
Parallel State Application:Apply the incoming state matrix $S \in \mathbb{R}^{j \times d}$ to the whole chunk of queries at once:PythonY_inter = tl.dot(Q_c, S) * tl.exp(g_cumsum)[:, None] # [C, j] @ [j, d] -> [C, d]
Y_total = Y_intra + Y_inter
Parallel State Update:Update the state $S$ using the entire chunk of $K$ and $V$:Python# Decay K relative to the end of the chunk
k_decay = tl.exp(g_cumsum[-1] - g_cumsum)
K_c_decayed = K_c * k_decay[:, None]

# S_new = S_old * exp(g_cumsum[-1]) + K^T @ V
S = S * tl.exp(g_cumsum[-1]) + tl.dot(tl.trans(K_c_decayed), V_c)
3. Execution DirectivesRefactor src/chunk_gla_inlier.py: Replace the sequential inner for loop with the parallel tl.dot logic outlined above.SRAM Constraints: Ensure BLOCK_N (chunk size), BLOCK_J, and BLOCK_D are tl.constexpr. Use BLOCK_N = 64 as a stable default.Contiguity: Ensure all PyTorch tensors passed to the kernel have .contiguous() called on them in the wrapper before grid launch.Benchmarking Loop: - Run python benchmarks/profile.py.Inspect the terminal output.If HOFA latency > MHA latency at 65,536 tokens, analyze the Triton kernel for non-parallel operations, memory coalescing issues, or unoptimized data types (ensure tl.dot inputs are tl.float16 or tl.bfloat16).Iterate and modify src/chunk_gla_inlier.py until the latency target is achieved.GoalThe benchmark script must show HOFA's latency scaling linearly, crossing below FlashAttention's quadratic latency curve. Do not complete this task until that crossover is observed in the terminal output.

## shared: dont run any bash commands other than running python benchmarking, via python3. 
# I will now go to sleep. I expect the kernal to fully work when I come back.