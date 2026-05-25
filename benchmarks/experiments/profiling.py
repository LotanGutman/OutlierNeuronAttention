"""
Profiling: Triton-Optimized Hybrid vs. Standard MHA (FlashAttention).
Measures forward-pass time (ms) and peak VRAM (GB) for seq lengths 512–8192.
"""
import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from src.triton_model import OutlierFactorizedLinearAttention as OptimizedOutlierFactorizedLinearAttention
from src.config import ModelConfig, TrainingConfig

def profile():
    train_cfg = TrainingConfig()
    device = torch.device(train_cfg.device)
    torch.manual_seed(train_cfg.seed)
    model_cfg = ModelConfig()
    d_head = model_cfg.d_head
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    
    # --- Build the hybrid (Triton) ---
    hybrid_attn = OptimizedOutlierFactorizedLinearAttention(model_cfg).to(device).eval()
    
    # --- We'll also need a standard MHA wrapper (use built-in FlashAttention) ---
    class StandardMHAWrapper:
        """Applies the same pre-scaling as the hybrid, then calls FlashAttention."""
        def __init__(self, attn_module):
            self.W_q = attn_module.W_q
            self.W_k = attn_module.W_k
            self.W_v = attn_module.W_v
            self.d_head = attn_module.d_head
        def __call__(self, x):
            B, N, D = x.shape
            scale_factor = self.d_head ** 0.25
            Q = (self.W_q(x) / scale_factor).view(B, N, -1, self.d_head).transpose(1, 2)
            K = (self.W_k(x) / scale_factor).view(B, N, -1, self.d_head).transpose(1, 2)
            V = self.W_v(x).view(B, N, -1, self.d_head).transpose(1, 2)
            # Use PyTorch's FlashAttention implementation
            Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=1.0)
            return Y.transpose(1, 2).reshape(B, N, D)
            
    mha = StandardMHAWrapper(hybrid_attn)   # uses the same weights
    
    # Storage
    times_mha = []
    times_hyb = []
    mems_mha = []
    mems_hyb = []
    valid_lens = []
    
    # Warmup
    dummy = torch.randn(1, 512, model_cfg.d_model, device=device)
    with torch.no_grad():
        _ = mha(dummy)
        _ = hybrid_attn(dummy)
        
    print(f"{'seq_len':<8} | {'MHA Time':<10} | {'MHA Mem':<10} | {'Hyb Time':<10} | {'Hyb Mem':<10}")
    print("-" * 60)

    for sl in seq_lengths:
        x = torch.randn(1, sl, model_cfg.d_model, device=device)
        
        # --- Standard MHA ---
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                # warmup
                for _ in range(5): _ = mha(x)
                torch.cuda.synchronize()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10): _ = mha(x)
                end.record()
                torch.cuda.synchronize()
                
                t_mha = start.elapsed_time(end) / 10.0
                m_mha = torch.cuda.max_memory_allocated() / (1024**3)
                
            times_mha.append(t_mha)
            mems_mha.append(m_mha)
            valid_lens.append(sl)
            
        except torch.cuda.OutOfMemoryError:
            print(f"MHA OOM at seq_len={sl}, stopping.")
            break
            
        # --- Triton Hybrid ---
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                # warmup
                for _ in range(5): _ = hybrid_attn(x)
                torch.cuda.synchronize()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10): _ = hybrid_attn(x)
                end.record()
                torch.cuda.synchronize()
                
                t_hyb = start.elapsed_time(end) / 10.0
                m_hyb = torch.cuda.max_memory_allocated() / (1024**3)
                
            times_hyb.append(t_hyb)
            mems_hyb.append(m_hyb)
            
        except torch.cuda.OutOfMemoryError:
            print(f"Hybrid OOM at seq_len={sl}, stopping.")
            break
            
        print(f"{sl:<8d} | {t_mha:8.3f} ms | {m_mha:.3f} GB | {t_hyb:8.3f} ms | {m_hyb:.3f} GB")

    # Plotting
    if not valid_lens:
        print("No valid data to plot.")
        return

    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(valid_lens[:len(times_mha)], times_mha, 'o-', color='tab:red', label='Standard MHA (FlashAttention)')
    ax1.plot(valid_lens[:len(times_hyb)], times_hyb, 's-', color='tab:blue', label='Hybrid (Triton)')
    ax1.set_xlabel('Sequence length')
    ax1.set_ylabel('Forward pass time (ms)')
    ax1.set_title('Latency Comparison')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2.plot(valid_lens[:len(mems_mha)], mems_mha, 'o-', color='tab:red', label='Standard MHA (FlashAttention)')
    ax2.plot(valid_lens[:len(mems_hyb)], mems_hyb, 's-', color='tab:blue', label='Hybrid (Triton)')
    ax2.set_xlabel('Sequence length')
    ax2.set_ylabel('Peak VRAM (GB)')
    ax2.set_title('Memory Comparison')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_path = 'benchmarks/experiments/plots/hybrid_vs_mha_profile.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    print(f"\nProfile plot saved to {plot_path}")

if __name__ == "__main__":
    profile()
