# Implementation Details

This document provides a more technical overview for the codebase.

For the main repository overview, see [README.md](../README.md).

---

## Kernel Architecture

This codebase uses two execution pipelines optimized for different workloads:

### Training Pipeline (`src/HybridOutlierFactorizedAttentionTrain.py`)
- **Outlier Pathway ($r$ dimensions):** Evaluated using PyTorch `scaled_dot_product_attention` (SDPA), which automatically dispatches to **FlashAttention-2** with Rotary Position Embeddings (RoPE).
- **Inlier Pathway ($j = d_{\text{head}} - r$ dimensions):** Evaluated using parallel chunked Gated Linear Attention via `fla.ops.gla.chunk_gla`.
- **Dynamic Layout Alignment:** Handles tensor layout conversions (`[B, H, N, K]` vs `[B, N, H, K]`) cleanly to ensure zero memory overhead during backward passes.

### Inference & Decoding Pipeline (`src/HybridOutlierFactorizedAttention.py`)
- **Fused Triton Decode Kernel (`fused_hofa_decode_kernel`):** Single-pass fused decoding kernel written in Triton. In a single GPU launch, it performs:
  1. Online Softmax computation for the $r$ outlier dimensions against KV cache.
  2. Sequential state update for the GLA recurrent matrix $S_t = \alpha S_{t-1} + K^\top V$.
  3. Inlier RMSNorm and LayerScale normalization.
  4. Token-level mixing gate blending ($\alpha_h \odot Y_{\text{outlier}} + (1 - \alpha_h) \odot Y_{\text{inlier}}$).

---

## Training-Free Extension Diagnostics (YaRN)

Training-free context extension diagnostics use YaRN implemented in `src/yarn.py` with canonical LLaMA defaults ($\alpha=1.0, \beta=32.0$).