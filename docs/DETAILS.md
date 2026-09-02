# Implementation Details & Kernel Specifications

This document provides technical specifications for the PyTorch training modules, fused Triton decode kernels, and hardware utilization analysis of **Hybrid Outlier-Factorized Attention (HOFA)**.

For the main repository overview, see [README.md](../README.md).

---

## 1. Dual-Pathway Execution & Kernel Architecture

HOFA uses two execution pipelines optimized for different workloads:

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

## 2. Recurrent State Memory Management

During auto-regressive generation, HOFA maintains a recurrent state tensor $S_t \in \mathbb{R}^{B \times H \times d_{\text{inlier}} \times d_v}$ alongside the KV cache.

* **Ping-Pong Buffer Allocation:** To avoid memory allocations during generation steps, HOFA pre-allocates two ping-pong buffers for $S_t$ at generation start and swaps pointers on each step.
* **Autotuning:** Uses `@triton.autotune` to dynamically select optimal GPU block sizes (`BLOCK_SEQ`) based on active batch size and sequence length.

---
