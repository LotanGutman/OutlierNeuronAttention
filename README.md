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

## Training vs. Inference

The codebase has two separate HOFA implementations:

### `src/HybridOutlierFactorizedAttentionTrain.py` (Training)
- Uses `fla.ops.gla.chunk_gla` for the inlier GLA pathway (gradients handled by the `fla` library).
- Uses PyTorch `scaled_dot_product_attention` for the exact pathway.
- Gradient checkpointing via `torch.utils.checkpoint.checkpoint`.
- Weight initialization (GPT-2 style: std=0.02, residual branches scaled by 1/√(2·num_layers)).
- 30M parameter language model training pipeline (FineWeb-Edu).

### `src/HybridOutlierFactorizedAttention.py` (Inference / Decode)
- Custom Triton kernel `chunk_gla_inlier_fwd` for the GLA forward pass (no gradients needed).
- Custom Triton kernel `exact_attention_triton` for exact attention (with optional Split-K for long sequences, auto-tuned with early SRAM pruning).
- Fused decode kernel `fused_hofa_decode` that combines online-softmax exact attention, GLA state update, inlier RMSNorm + LayerScale, and pathway blending in a **single Triton kernel**.

---

## Repository Layout

```
src/
├── HybridOutlierFactorizedAttention.py        # Inference-optimized HOFA (custom Triton kernels)
├── HybridOutlierFactorizedAttentionTrain.py   # Training-optimized HOFA (fla library)
├── exact_attention.py                         # Triton exact-attention kernel (Split-K, autotuned)
├── chunk_gla_inlier.py                        # Triton chunked GLA forward kernel (inference)
├── hofa_decode_triton.py                      # Fused Triton decode kernel
├── config.py                                  # ModelConfig dataclass
├── inference.py                               # InferenceEngine (generation loop)
└── modules/
    ├── modules.py                             # RotaryEmbedding, rotate_half, apply_rotary_pos_emb
    ├── triton_utils.py                        # SRAM querying, block-size calculation
    ├── checkpointing.py                       # save_checkpoint / load_checkpoint
    └── benchmark_utils.py                     # GenericBenchmarkLM, StandardMHA, build_attention

training/
├── training_config.py                         # LanguageModelingExperimentConfig (model_name, dataset, hparams)
├── train.py                                   # 30M training loop (FineWeb-Edu)
├── download_fineweb.py                        # FineWeb-Edu download & tokenization
├── inference.py                               # CLI interactive generation for trained models
└── plot_training.py                           # Training metrics plotting (loss / LR curves)

benchmarks/
├── benchmarks_configs.py                      # Experiment config dataclasses
├── benchmark_induction.py                     # Induction head training (MHA vs HOFA vs GLA vs Mamba)
├── benchmark_K_eff.py                         # Effective attention-mass measurement on real LLMs
├── profile_prefill.py                         # Prefill latency + FLOPs (MHA vs HOFA)
├── profile_decode.py                          # Decode throughput + KV-cache footprint
└── plotting/
    ├── plot_induction.py                      # Induction head convergence plots
    ├── plot_K_eff.py                          # Attention-mass heatmaps
    └── plot_layerwise.py                      # Per-layer regime stacking plots

main.py                                         # CLI entry point (download-data | train | infer | plot)
```

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

# 2. Train the 30M model
python main.py train

# 3. Plot training metrics
python main.py plot

# 4. Interactive generation
python main.py infer
```

The experiment config lives in `training/training_config.py`. To scale to a larger model, change `model_name` (e.g. to `"100M"`) and update the architecture hyperparameters — the cache paths, checkpoint directories, and plot directories all derive from `model_name`.

---

## Dependencies

- PyTorch ≥ 2.4
- Triton (latest nightly for Hopper support)
- `fla` (flash-linear-attention) — for training GLA pathway
- `tiktoken`, `datasets`, `tqdm`
- `matplotlib`, `numpy` (for plotting and analysis)