# Hybrid Outlier-Factorized Attention (HOFA)

HOFA is an attention mechanism that splits each head's feature dimension into two pathways: an **outlier / exact pathway** (first `r` dims — standard SDPA) and an **inlier / linear pathway** (remaining `j = d_head - r` dims — Gated Linear Attention). A learned token-level mixing gate blends the two outputs. RoPE is applied **only** to the exact dimensions.

This gives **O(N) memory scaling** with strictly better recall than pure linear attention, at a fraction of full softmax attention's cost.

---

## How It Works

For each head, given Q, K, V ∈ ℝ^(B × H × N × d_head) with r < d_head:

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

The codebase maintains strict separation between Training and Inference to maximize hardware utilization for their respective paradigms (Parallel vs. Autoregressive).

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

Mathematical equivalence between the PyTorch Training model and the Triton Custom Inference model is guaranteed via a rigorous integration test script.

`python main.py infer --validate`

This suite forces the highly-fused Autotuned PyTorch model (`HOFA_Train`) and the custom Autotuned Triton Kernel (`HOFA_Infer`) to execute identical autoregressive steps in `bfloat16`. 
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
├── train.py                                   # Language modeling training loop (FineWeb-Edu) with crash-safe checkpointing
├── download_fineweb.py                        # FineWeb-Edu download & tokenization with atomic state flushing
├── inference.py                               # CLI interactive generation for trained models
└── plot_training.py                           # Training metrics multi-model shared plotting

benchmarks/
├── validate_kernels.py                        # Integration testing for HOFA_Train vs HOFA_Infer (fused from debug_decode)
├── benchmarks_configs.py                      # Experiment config dataclasses
├── benchmark_zeroshot.py                      # Full zero-shot common-sense suite (HellaSwag, ARC, PIQA, WinoGrande, OBQA) with caching
├── benchmark_induction.py                     # Induction head training (MHA vs HOFA vs GLA vs Mamba)
├── benchmark_K_eff.py                         # Effective attention-mass measurement on real LLMs
├── profile_prefill.py                         # Prefill latency + FLOPs (MHA vs HOFA)
├── profile_decode.py                          # Decode throughput + KV-cache footprint
└── plotting/
    ├── plot_induction.py                      # Induction head convergence plots
    ├── plot_K_eff.py                          # Attention-mass heatmaps
    └── plot_layerwise.py                      # Per-layer regime stacking plots

main.py                                         # CLI entry point (download-data | train | infer | profile)
```

---

## Training Stability

The training loop incorporates built-in data leakage prevention for sequence tokenization and an **Emergency Checkpointing System**. If a crash occurs (e.g. out of disk space or an unexpected interrupt), the model safely dumps its state to `checkpoint_emergency.pt` instead of corrupting the valid checkpoint file, preserving hours of expensive training progress.

---

## Key Results (so far)

### Induction Head Task
HOFA matches standard MHA accuracy when `r ≥ ceil(log₂(V))` (where V = vocab size). Below that bound the exact pathway cannot disambiguate the token space and accuracy collapses — exactly as predicted by theory.

| Model | Acc @ vocab=8K, seq=1024 |
|---|---|
| MHA | ~100% |
| HOFA r=16 | ~100% |
| HOFA r=14 | ~100% |
| HOFA r=12 | ~85% (below bound) |
| GLA (pure) | ~55% |

### Prefill Scaling
HOFA's exact pathway is bounded at `r` dimensions, so FLOPs scale as **O(N · r)** rather than **O(N · d_head)**. This yields 2-3× speedup over standard MHA at 131K sequence length, with the gap widening at longer contexts.

### Memory Footprint
Strict **O(N)** memory during training — no materialization of full attention matrices. The exact pathway uses softmax merging (online softmax) and the GLA pathway maintains only a compact recurrent state `S ∈ ℝ^(j × d_head)`.

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

# 6. Evaluate zero-shot reasoning
python main.py train --eval            # Runs the full suite (HellaSwag, ARC, PIQA, WinoGrande, OBQA) with caching
python main.py train --eval --simple   # Runs only HellaSwag

# 7. Interactive generation
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

**Note on Installation**: The official `install.sh` downloads pre-compiled wheels for `mamba-ssm` and `causal-conv1d` to perfectly match PyTorch 2.6.0 on Python 3.10-3.12 (do not use 3.13 yet). We explicitly disable source compilation to avoid CUDA toolchain mismatches. We explicitly pin `mamba-ssm==2.2.4` to avoid `quack-kernels` dependencies introduced in 2.3+ which cause FP8 initialization crashes on standard PyTorch 2.6.0, and to retain native support for Triton 3.2.0 without requiring hot-patches.

## Inference & Autotuning Optimizations

During interactive inference, changing sequence lengths historically triggered catastrophic PyTorch/Triton compilation loops. We have implemented several mitigations to guarantee instant, fluid generation:

1. **Prefill Autotuning Fix:** In `exact_attention.py`, the sequence length (`N_CTX`) was removed from the `@triton.autotune` key. This prevents Triton from triggering a 5-second GPU benchmark sweep every time you type a prompt of a different length.
2. **Decode Autotuning Fix:** In `fused_hofa_decode_kernel`, the sequence length is similarly excluded from the tuning keys to prevent recompilation on every single generated token.
3. **Silent Initialization Warmup:** To ensure the very first prompt is perfectly fluid, `InferenceEngine.__init__` executes a silent 1-token "Warmup" generation in the background immediately after checkpoint loading. This forces Triton to absorb all JIT compilation overhead before the user is ever presented with a prompt.
4. **Separated Performance Metrics:** The interactive `infer` CLI clearly separates the mathematical Prefill Tokens-Per-Second from the Decode Tokens-Per-Second, allowing precise performance profiling without startup bias.
5. **Dynamic Gate Tracing (Zero-Overhead):** Using the `--debug` flag activates a PyTorch monkey patch (`patch_attention_for_debugging`) inside `benchmark_utils.py` that dynamically intercepts Mix Gate and GLA Gate Logit activations during inference, without polluting or slowing down the highly-optimized core model file.

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

```
Load K₀,V₀ | Compute A₀ + Load K₁,V₁ | Compute A₁ + Load K₂,V₂ | ...
```

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
