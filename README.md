# Hybrid Outlier-Factorized Attention (HOFA)

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg?logo=pytorch)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg?logo=python)

HOFA is an attention mechanism that splits each head's feature dimension into two pathways: an **outlier / exact pathway** (first `r` dims — standard SDPA) and an **inlier / linear pathway** (remaining `j = d_head - r` dims — Gated Linear Attention). A learned token-level mixing gate blends the two outputs. RoPE is applied **only** to the exact dimensions.

This gives **O(N) memory scaling** with strictly better recall than pure linear attention, at a fraction of full softmax attention's cost.

## Quickstart (Experimental)

> [!WARNING]
> This codebase is actively under experimental development. Installation scripts and core architectures may change rapidly.

```bash
# Clone the repository
git clone https://github.com/LotanGutman/OutlierNeuronAttention.git
cd OutlierNeuronAttention/official_code

# Install dependencies (forces PyTorch 2.6.0 + native compilation)
bash install.sh
```

---

## How It Works

For each head, given $Q, K, V \in \mathbb{R}^{B \times H \times N \times d_{\text{head}}}$ with $r < d_{\text{head}}$:

| Pathway | Dimensions | Mechanism | RoPE |
|---|---|---|---|
| **Outlier** `Y_O` | Q[:r], K[:r], V | Softmax attention (FlashAttention / Triton) | ✅ |
| **Inlier** `Y_I` | Q[r:], K[r:], V | Gated Linear Attention (chunked linear recurrence) | ❌ |

```
gate_logits = W_gate([Q, K])     → γ = σ(gate_logits)   (GLA decay)
mix_logits  = W_mix([Q, K])      → mix_g = σ(mix_logits) (pathway blend)
Y_out = mix_g · Y_O + (1 - mix_g) · Y_I
```

Because `r` is small relative to `d_head`, the exact attention cost is **O(N · r)** not **O(N · d_head)**, and the GLA pathway is **O(N)** in memory.

---

## Training vs. Inference Architecture

The codebase separates Training and Inference implementations to optimize for their respective workloads (parallel prefix versus autoregressive decoding).

### `src/HybridOutlierFactorizedAttentionTrain.py` (Training)
- Built on `fla.ops.gla.chunk_gla` for the highly parallelized inlier GLA pathway (handling the complex backpropagation).
- Handles layout mismatches dynamically (`[B, H, N, K]` vs `[B, N, H, K]`) to guarantee mathematical correctness without degrading throughput.
- Uses `scaled_dot_product_attention` for the exact pathway, leveraging `FlashAttention-2` and `Memory-Efficient Attention (xFormers)` automatically depending on the hardware architecture and dimension.
- Employs **`@torch.compile(mode="max-autotune")`** to dynamically fuse RMSNorms, Linear projections, and tensor layouts directly into optimized Triton kernels.
- Gradient checkpointing via `torch.utils.checkpoint.checkpoint`.

### `src/HybridOutlierFactorizedAttention.py` (Inference / Decode)
- Uses a **Single Fused Triton Decode Kernel** (`fused_hofa_decode_kernel`) that merges: online-softmax exact attention, GLA recurrent state updates, inlier RMSNorm, LayerScale, and pathway blending.
- Implements **Ping-Pong Buffer Architecture** for the recurrent state. Pre-allocates two buffers alongside the KV cache and ping-pongs pointers every token generation step. This completely eliminates expensive per-token memory allocations (`.clone()`) while safely isolating the state to prevent Triton memory corruption during tuning.
- Employs **`@triton.autotune`** to automatically benchmark and select the theoretical optimal warp and block configurations (`BLOCK_SEQ`) based on the executing GPU topology.
- KV Cache is strictly pre-allocated to prevent $O(N^2)$ `torch.cat` memory fragmentation.

---

## Testing and Validation

The integration tests verify numerical equivalence between the PyTorch training model and the custom Triton inference kernels.

`python main.py infer --validate`

This script runs the PyTorch model (`HOFA_Train`) and the Triton kernel (`HOFA_Infer`) through identical autoregressive steps in `bfloat16`. 
It evaluates:
- **Cosine Similarity:** Consistently achieves $> 0.9999$.
- **Mean Absolute Error (MAE):** Consistently bounded $\approx 10^{-4}$.
- **Max Absolute Difference:** Isolated to the natural boundary of `bfloat16` accumulation float drift ($\approx 10^{-3}$).

---

## Repository Layout

```
src/
├── HybridOutlierFactorizedAttention.py        # Inference-optimized HOFA (custom Triton kernels)
├── HybridOutlierFactorizedAttentionTrain.py   # Training-optimized HOFA (fla library)
├── exact_attention.py                         # Triton exact-attention kernel (Split-K, autotuned)
├── chunk_gla_inlier.py                        # Triton chunked GLA forward kernel (inference)
├── hofa_decode_triton.py                      # Fused Triton decode kernel (Ping-Pong buffers + Autotuning)
├── config.py                                  # ModelConfig dataclass
├── inference.py                               # InferenceEngine (generation loop)
└── modules/
    ├── modules.py                             # RotaryEmbedding, rotate_half, apply_rotary_pos_emb
    ├── triton_utils.py                        # SRAM querying, block-size calculation
    ├── checkpointing.py                       # save_checkpoint / load_checkpoint
    └── benchmark_utils.py                     # GenericBenchmarkLM, StandardMHA, build_attention

training/
├── training_config.py                         # LanguageModelingExperimentConfig (model_name, dataset, hparams)
├── data_utils.py                              # FastTokenLoader (np.memmap + pinned memory for optimized data streaming)
├── train.py                                   # Language modeling training loop (FineWeb-Edu) with crash-safe checkpointing
├── download_fineweb.py                        # FineWeb-Edu download & tokenization with atomic state flushing
├── inference.py                               # CLI interactive generation for trained models
└── plot_training.py                           # Training metrics multi-model shared plotting

benchmarks/
├── validate_kernels.py                        # Integration testing for HOFA_Train vs HOFA_Infer (fused from debug_decode)
├── benchmarks_configs.py                      # Experiment config dataclasses
├── benchmark_zeroshot.py                      # Full zero-shot common-sense suite (HellaSwag, ARC, PIQA, WinoGrande, OBQA) with caching
├── benchmark_induction.py                     # Induction head sequence length scaling (MHA vs HOFA vs GLA vs Mamba)
├── benchmark_induction_degradation.py         # Extended context length degradation sweep for HOFA (r=10)
├── benchmark_copying.py                       # Selective copying benchmark (MHA vs HOFA vs GLA vs Mamba)
├── benchmark_K_eff.py                         # Effective attention-mass measurement on real LLMs
├── benchmark_distance.py                      # Effective Attention Distance layerwise scaling
├── profile_prefill.py                         # Prefill latency + FLOPs (MHA vs HOFA)
├── profile_decode.py                          # Decode throughput + KV-cache footprint
└── plotting/
    ├── plot_induction.py                      # Induction head convergence plots
    ├── plot_K_eff.py                          # Attention-mass heatmaps
    ├── plot_layerwise.py                      # Per-layer regime stacking plots
    └── plot_distance.py                       # Layerwise distance magnifying plots

main.py                                         # CLI entry point (download-data | train | infer | profile)
```

---

## Training Stability

The training loop includes crash-safe checkpointing. If an unexpected interrupt occurs (e.g., OOM or keyboard interrupt), the model intercepts the signal and saves its state to `checkpoint_emergency.pt` to avoid corrupting the primary checkpoint file.

---

## Key Results (so far)

### Induction Head & Retrieval Scaling
HOFA matches standard MHA accuracy when $r \ge \lceil\log_2(N)\rceil$ (where $N$ is sequence length). Key discoveries:
1. **Context Length Capacity Limit:** For $N=1024$, $r=10$ ($2^{10}=1024$) is the exact threshold required for 100% retrieval. As $N$ extends beyond 1024 (up to $N=4096$), $r=10$ exhibits a smooth, continuous Gaussian tail degradation—bypassing the catastrophic 0% collapse suffered by pure linear recurrence models (GLA/Mamba).
2. **Vocabulary Size Decoupling ($V$-Robustness):** Scaling vocabulary size $V$ up to 43,008 ($43\text{k}$) at $N=512$ retains $>97\%$ accuracy for $r=8$. Attention rank is strictly bounded by context window size $N$, not raw vocabulary size $V$.

### Prefill Scaling
HOFA's exact pathway is bounded at `r` dimensions, so FLOPs scale as **O(N · r)** rather than **O(N · d_head)**. This yields 2-3× speedup over standard MHA at 131K sequence length, with the gap widening at longer contexts.

### Effective Attention Distance
By analyzing 125M evaluation checkpoints, we trace the "effective distance" of attention lookups layer by layer. HOFA's exact outlier pathway ($r=16$) maintains an effective distance of $\sim 257$ tokens (matching full MHA), preserving global reach. In stark contrast, the GLA-effective inlier pathway operates at a localized $\sim 4$ tokens, cleanly absorbing local context. This validates HOFA's core architectural claim of a division of labor: linear recurrence for short-term memory, and exact attention for long-range retrieval.

### Memory Footprint
Strict **O(N)** memory during training — no materialization of full attention matrices. The exact pathway uses softmax merging (online softmax) and the GLA pathway maintains only a compact recurrent state $S \in \mathbb{R}^{j \times d_{\text{head}}}$.

---

## Getting Started

```bash
# 1. Download and cache the dataset
python main.py download-data

# 2. Validate custom Triton decoding against PyTorch JIT compiler
python main.py infer --validate

# 3. Train the model (defaults to 125M)
python main.py train

# 4. Profile the model's throughput and memory
python main.py profile --prefill
python main.py profile --decode

# 5. Plot training metrics
python main.py train --plot
python main.py train --plot --shared   # Generates comparative academic plots (Tokens & FLOPs) between HOFA and MHA

# 7. Synthetic Benchmarks
python main.py benchmark --induction              # Run Induction Head sequence length scaling
python main.py benchmark --induction --plot       # Generate unified trendline and feature norm disparity plots
python main.py benchmark --induction-degradation  # Run extended context length degradation sweep (N=1024..4096, r=10)
python main.py benchmark --copy                   # Run sequential copying benchmark
python main.py benchmark --keff                   # Run Probability Mass Decomposition (K_eff) scaling benchmark on HuggingFace LLMs
python main.py benchmark --keff --plot            # Generate the 2x4 Attention Mass Heatmaps

# 8. Interactive generation
python main.py infer
python main.py infer --train   # Run using the pure PyTorch training autoregressive loop instead of Triton
python main.py infer --debug   # Enable detailed, dynamic gating statistics per-prompt
```

The experiment config lives in `training/training_config.py`. It includes presets for scaling models:
- `make_30m_pure_gla()`, `make_30m_hofa()`
- `make_125M_hofa()`, `make_125M_mha()`
- `make_350M_hofa()`, `make_350M_mha()`

To scale to a larger model, simply change the active config in `main.py`. The pipeline is smart enough to share downloaded datasets (`cache.bin` and `val_cache.bin`) between MHA and HOFA variants of the same size to save disk space and preparation time. Checkpoints and plot directories automatically derive from the full `model_name`.

---

## Environment & Compilation Strategy

PyTorch 2.0+ `torch.compile` provides massive speedups but introduces significant graph breaks and startup overhead when applied aggressively to individual inner modules (like `@torch.compile(mode="max-autotune")` on an attention class). 

To achieve maximum throughput and avoid endless autotune loops, the HOFA training pipeline explicitly avoids compiling inner modules. Instead, the entire model is compiled dynamically in `train.py`:
```python
model = torch.compile(model, dynamic=True)
```
This enables a single unified execution graph and allows training to start instantly without hanging on kernel benchmarking.

**Note on Installation**: The official `install.sh` establishes a **PyTorch 2.6.0** baseline (with CUDA 12.4). We strictly force native compilation of all C++ extensions (`causal-conv1d` and `mamba-ssm`) against this specific PyTorch version using `--no-build-isolation` and `--no-deps`. This prevents `pip` from downloading conflicting versions of PyTorch/Triton during the build step, ensuring absolute harmony between Flash Linear Attention (FLA) and Mamba in the exact same environment.

### Troubleshooting: The Mamba Version Trap

Historically, running PyTorch 2.6.0 alongside both `fla` and `mamba-ssm` was impossible due to a fragile circular dependency trap in the Mamba C++ ecosystem:
1. `mamba-ssm 2.2.4` expects the `causal-conv1d` extension to have exactly 7 arguments.
2. `causal-conv1d 1.4.0` has 7 arguments, but it **fails to compile** natively against the new C++ headers in PyTorch 2.6.0.
3. Upgrading to `causal-conv1d 1.6.2` compiles perfectly on PyTorch 2.6.0, but changes its signature to 8 arguments, immediately crashing `mamba-ssm 2.2.4`.
4. Upgrading to `mamba-ssm 2.3.2+` (which supports 8 arguments) strictly requires `triton>=3.5.0`.
5. Forcing a Triton upgrade to 3.5.0 breaks PyTorch 2.6.0's native `torch.compile` (which demands `triton==3.2.0`), causing PIP to panic and overwrite your CUDA environment.

**The Solution (Native Compilation Bypass):**
We solved this by leveraging the newly released `mamba-ssm==2.2.5` (which bridges the 8-argument C++ ABI gap without enforcing a Triton 3.5.0 upgrade) and forcing it to compile directly against PyTorch 2.6.0's headers.

In `install.sh`, we explicitly do:
```bash
# 1. Lock PyTorch 2.6.0
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 2. Install FLA without letting it downgrade/upgrade torch
python -m pip install -U "flash-linear-attention[cuda]" --no-deps

# 3. Force native compilation of causal-conv1d and mamba-ssm 2.2.5 
export TORCH_CUDA_ARCH_LIST="native"
export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
pip install causal-conv1d --no-build-isolation --no-cache-dir
pip install "mamba-ssm==2.2.5" --no-build-isolation --no-cache-dir --no-deps
```
By passing `--no-build-isolation` and `--no-deps`, we strip away their isolated build environments and force them to link directly against our active PyTorch 2.6.0 tensor library. This yields a single, flawless environment where HOFA, GLA, DeltaNet, and Mamba can all be benchmarked simultaneously!

**Note on Checkpoint Loading (`ValueError: different number of parameter groups`)**: 
If you try to load an older Mamba checkpoint into the current codebase, it may crash on `optimizer.load_state_dict()` because the new HOFA codebase splits the optimizer into two parameter groups (for Selective Regularization). Since you only need the model weights to evaluate Mamba as a baseline, you can safely bypass this by temporarily commenting out `optimizer.load_state_dict(ckpt['optimizer_state_dict'])` in `src/modules/checkpointing.py`.

## Inference Optimizations

We have implemented several changes to prevent unnecessary PyTorch/Triton recompilation during interactive generation:

1. **Prefill Tuning:** In `exact_attention.py`, the sequence length (`N_CTX`) is excluded from the `@triton.autotune` key. This prevents Triton from running a benchmark sweep when the prompt length changes.
2. **Decode Tuning:** In `fused_hofa_decode_kernel`, the sequence length is similarly excluded from the tuning keys to avoid recompilation per generated token.
3. **Initialization Warmup:** `InferenceEngine.__init__` executes a silent 1-token warmup generation after checkpoint loading. This absorbs JIT compilation overhead prior to interactive usage.
4. **Separated Metrics:** The `infer` CLI separates Prefill Tokens-Per-Second from Decode Tokens-Per-Second for accurate profiling.
5. **Gate Tracing:** Passing the `--debug` flag activates a monkey patch (`patch_attention_for_debugging`) to trace Mix Gate and GLA Gate logits during inference, keeping the core model file clean.

---

## Dependencies

- PyTorch ≥ 2.4
- Triton (latest nightly for Hopper support)
- `fla` (flash-linear-attention) — for training GLA pathway
- `matplotlib`, `numpy` (for plotting and analysis)

---

## Kernel Performance Analysis

The `fused_hofa_decode` kernel is memory-bound rather than config-bound — `@triton.autotune` configs (BLOCK_SEQ, num_warps, num_stages) all produce similar memory traffic patterns, so tuning alone cannot unlock significant gains. The root cause is **low Streaming Multiprocessor (SM) utilization**:

- **Grid size** = `(Batch, Heads)` = `(1, 16)` = 16 thread blocks
- On an RTX 4060 (24 SMs): **only 67% occupancy** — 8 SMs sit idle
- On an H100 (132 SMs): **only 12% occupancy** — 116 SMs sit idle
- Each block loops sequentially over the entire KV cache, so the exact-attention phase is effectively single-SM per head

Additionally, the GLA recurrent state (`j × d_head` = 112 × 128 = 14,336 floats) is too large for registers, causing spill to shared memory and increasing pressure on the exact-attention loop.

---

## Future Optimizations

Four complementary approaches to close the utilization gap, ordered by expected impact:

### 1. Split-K Over Sequence Length

Split the K/V cache loop across multiple SMs per head, then reduce partial results with an atomic or log-sum-exp merge. The prefill kernel (`exact_attention.py`) already implements Split-K — the same pattern can be applied to the decode kernel.

| | Current | Split-K (e.g. K=4) |
|---|---|---|
| Thread blocks (16 heads) | 16 | 64 |
| RTX 4060 SM utilization | 67% | 100% (oversubscribed) |
| H100 SM utilization | 12% | 48% |

Each block processes `seq_len / K` tokens instead of the full sequence, linear speedup in the exact-attention phase. The reduction step is lightweight (one atomic per head per split).

### 2. Async Prefetch / Double-Buffering

The exact-attention loop loads K and V cache tiles from global memory, then computes attention scores. These two phases are serialized — the compute units idle during loads. Double-buffer the next tile's load while computing the current tile:

$$ \text{Load } K_0, V_0 \mid \text{Compute } A_0 + \text{Load } K_1, V_1 \mid \text{Compute } A_1 + \text{Load } K_2, V_2 \mid \dots $$

On Ampere+ (RTX 3060+), `tl.async_copy` with `tl.async_wait` enables pipelined global→shared memory transfers. Expected speedup: 20–40% on memory-bound long contexts since the loop is entirely memory-latency-bound.

### 3. Two-Tier Kernel Dispatch

Short and long contexts have different bottlenecks. Use a lightweight kernel for short contexts and the full fused kernel for long ones:

| Context | Kernel | Key difference |
|---|---|---|
| N < 4,096 | `_hofa_decode_short` | No RMSNorm fusion, no GLA tiling — minimal register pressure, maximum occupancy |
| N ≥ 4,096 | `_hofa_decode_long` | Current fused kernel with Split-K |

The short-context kernel avoids the loop overhead entirely (unrolls or uses a single block) and removes the RMSNorm/scale fusion to reduce register pressure. At N=512, the loop overhead is a significant fraction of total work.

### 4. GLA State Tiling

The GLA state is `j × d_head` = 112 × 128 = 14,336 floats (~57 KB in fp32). This doesn't fit in registers (256 KB per SM on Ampere, shared across warps). Tile the state update along the `j` dimension:

```
for j_tile in range(0, j, J_TILE):
    state_new[j_tile, :] = gamma * state[j_tile, :] + k_J[j_tile, None] * v_step[None, :]
```

Each tile fits in registers, reducing spills to shared memory and freeing bandwidth for the exact-attention loop. The GLA output computation (`q_J @ state`) tiles naturally along the same dimension. Expected improvement: 10–15% on top of Split-K by reducing register pressure in the fused kernel.

---

## Contributing

We welcome contributions! To ensure absolute stability of the Triton and PyTorch graphs, please adhere to the following when submitting a Pull Request:

1. **Validation Proof**: If your PR touches the PyTorch training architecture (`src/HybridOutlierFactorizedAttentionTrain.py`), the Triton inference kernels (`src/hofa_decode_triton.py`, `src/exact_attention.py`, `src/chunk_gla_inlier.py`), or any core module routing, you **must** run the mathematical validation suite. Include proof of its successful execution in your PR description.
   ```bash
   python main.py infer --validate
   ```
2. **Documentation Check**: If your change introduces a new CLI flag, alters tuning logic, or updates the environment baseline, you must update the relevant sections of this `README.md`.

---

## Citation

If you use HOFA in your research, please cite:

```bibtex
% (Currently empty - to be added upon publication)
```

---

## License

This project is licensed under the MIT License.
