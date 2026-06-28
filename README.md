# Hybrid Outlier-Factorized Attention (HOFA)

This repository contains the implementation of Hybrid Outlier-Factorized Attention (HOFA), featuring a custom, highly optimized Triton backward pass that completely eliminates intermediate HBM allocations for mathematical exactness and performance.

## Architecture: Fully Fused Triton Backward Pass

The backward pass replaces a naive PyTorch implementation with custom Triton kernels to prevent intermediate HBM allocations for $Y_O$ and $Y_I$, prevent `tl.atomic_add` collisions on $dV$ and $dK$, and avoid numerical instability in the bfloat16 GLA pathway.

### The 1-Kernel Monolithic Approach
The implementation utilizes a **Monolithic K-Parallel Backward Kernel**. 
To achieve sub-FlashAttention latency, we accumulate $dK$ and $dV$ natively in SRAM without cross-block `tl.atomic_add` collisions on massive sequence dimensions. This strictly requires a Key-Parallel grid.

- **On-the-Fly `dS` Accumulation:** Instead of materializing `dS` to HBM, the block maintains the recurrent gradient state `dS` directly in SRAM, accumulating it sequentially as it iterates backwards through future $Q$ blocks.
- **Outlier Pathway:** Computes exact SDPA gradients ($dQ_O, dK_O, dV_O$).
- **Inlier Pathway:** Computes GLA gradients ($dQ_I, dK_I, dV_I, d\log\gamma$).
- **Dynamic SRAM Profiling:** Instead of relying on hardcoded bounds, the kernel uses `triton.runtime.driver.active.utils.get_device_properties` to query actual GPU hardware limits and adaptively bounds chunk sizes to prevent SRAM overflow, making it fully portable across hardware profiles like RTX 4060 vs A100.
- **Numerical Safety:** Applies strict clamp `gamma_clamped = tl.maximum(gamma, 1e-6)` before computing $\log(\gamma)$ to prevent $-\inf$ NaNs in bfloat16. 

### Forward-Recompute Router 
The gradient of the dynamic routing gate ($d\_mix\_g = dY_{out} \cdot (Y_O - Y_I)$) requires the exact outputs of both pathways. However, saving $Y_O$ and $Y_I$ during the forward pass breaks the $O(N)$ memory scaling for massive sequences. 

The router kernel runs a Q-Parallel block-by-block recomputation of $Y_O$ and $Y_I$ exclusively in SRAM, computes the element-wise gradient $d\_mix\_g$, and immediately discards the intermediate states, achieving a strict $O(N)$ memory footprint.

## Verification
A high-precision FP32 testing suite (`test_hofa_bwd.py`) verifies the correct operation of the Triton backward kernels. It compares eager PyTorch accumulated gradients against the Triton operations. It validates:
- Standard block sizes (e.g. $N=128, d=64, r=16$)
- Stress tests activating dynamic block downscaling (e.g. $N=256, d=128, r=64$)
- Edge cases specifically guaranteeing `log_gamma` limits without producing NaNs when $g \approx 1.0$.

## Code Layout
- `src/HybridOutlierFactorizedAttention.py`: Main HOFA model class containing the PyTorch wrapper and custom autograd function binding.
- `src/hofa_bwd_kernels.py`: The high-performance Triton forward and backward kernels alongside helper functions.
- `test_hofa_bwd.py`: High-precision validation and verification test script.
