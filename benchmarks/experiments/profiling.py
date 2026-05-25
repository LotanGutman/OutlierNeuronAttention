"""
Profiling: Triton-Optimized Hybrid vs. Standard MHA (FlashAttention).
Measures forward-pass time (ms) and peak VRAM (GB) for seq lengths 512–524288.
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from src.triton_model import OutlierFactorizedLinearAttention as OptimizedOutlierFactorizedLinearAttention
from src.config import ModelConfig, TrainingConfig
from matplotlib.ticker import FuncFormatter, ScalarFormatter

def profile(warmup_steps=3, active_steps=10, train_cfg=TrainingConfig(), model_cfg=ModelConfig()):
    device = torch.device(train_cfg.device)
    torch.manual_seed(train_cfg.seed)
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
    
    hybrid_attn = OptimizedOutlierFactorizedLinearAttention(model_cfg).to(device).eval()
    
    # --- Isolate Attention from GEMMs ---
    # 1. Force the model to compute and cache the routing indices now
    hybrid_attn._maybe_update_indices()
    # 2. Prevent it from trying to update indices during the benchmark loop
    hybrid_attn._refresh_steps = 999999999
    # 3. Replace the linear layers with Identity to instantly skip the O(N * D^2) compute
    import torch.nn as nn
    hybrid_attn.W_q = nn.Identity()
    hybrid_attn.W_k = nn.Identity()
    hybrid_attn.W_v = nn.Identity()
    
    class StandardMHAWrapper:
        def __init__(self, attn_module):
            self.d_head = attn_module.d_head
        def __call__(self, x):
            B, N, D = x.shape
            scale_factor = self.d_head ** 0.25
            # x maps directly to Q, K, V since we killed the linear projections
            Q = (x / scale_factor).view(B, N, -1, self.d_head).transpose(1, 2)
            K = (x / scale_factor).view(B, N, -1, self.d_head).transpose(1, 2)
            V = x.view(B, N, -1, self.d_head).transpose(1, 2)
            Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=1.0)
            return Y.transpose(1, 2).reshape(B, N, D)
            
    mha = StandardMHAWrapper(hybrid_attn)
    
    times_mha, times_hyb = [], []
    mems_mha, mems_hyb = [], []
    valid_lens = []
    
    print(f"{'seq_len':<8} | {'MHA Time':<10} | {'MHA Mem':<10} | {'Hyb Time':<10} | {'Hyb Mem':<10}")
    print("-" * 60)

    for sl in seq_lengths:
        torch.cuda.empty_cache() # Clean slate only once per seq length
        # Scale down x to prevent exponential blowup since Q=K=V now
        x = torch.randn(1, sl, model_cfg.d_model, device=device) * 0.1
        
        # --- Standard MHA ---
        try:
            # Force compilation / initialization if any
            _ = mha(x)
            torch.cuda.synchronize()
            
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for _ in range(warmup_steps):
                    _ = mha(x)
                torch.cuda.synchronize()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                for _ in range(active_steps):
                    _ = mha(x)
                end.record()
                torch.cuda.synchronize()
                
                t_mha = start.elapsed_time(end) / active_steps
                m_mha = torch.cuda.max_memory_allocated() / (1024**3)
                
            times_mha.append(t_mha)
            mems_mha.append(m_mha)
            valid_lens.append(sl)
            
        except torch.cuda.OutOfMemoryError:
            print(f"MHA OOM at seq_len={sl}")
            break
            
        # --- Triton Hybrid ---
        try:
            # Force Triton JIT compile outside the timing and memory window
            _ = hybrid_attn(x)
            torch.cuda.synchronize()
            
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for _ in range(warmup_steps):
                    _ = hybrid_attn(x)
                torch.cuda.synchronize()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                for _ in range(active_steps):
                    _ = hybrid_attn(x)
                end.record()
                torch.cuda.synchronize()
                
                t_hyb = start.elapsed_time(end) / active_steps
                m_hyb = torch.cuda.max_memory_allocated() / (1024**3)
                
            times_hyb.append(t_hyb)
            mems_hyb.append(m_hyb)
            
        except torch.cuda.OutOfMemoryError:
            print(f"Hybrid OOM at seq_len={sl}")
            break
            
        print(f"{sl:<8d} | {t_mha:8.3f} ms | {m_mha:.3f} GB | {t_hyb:8.3f} ms | {m_hyb:.3f} GB")
        del x

    # --- Plotting ---
    if not valid_lens:
        return

    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

    def format_ticks_x(x, pos):
        return f'{int(x/1024)}k' if x >= 1024 else str(int(x))

    formatter_x = FuncFormatter(format_ticks_x)
    formatter_y = ScalarFormatter()
    formatter_y.set_scientific(False)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    vl1 = valid_lens[:len(times_mha)]
    vl2 = valid_lens[:len(times_hyb)]
    
    ax1.plot(vl1, times_mha, 'o-', color='tab:red', label='FlashAttention')
    ax1.plot(vl2, times_hyb, 's-', color='tab:blue', label='Ours')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('linear')
    ax1.xaxis.set_major_formatter(formatter_x)
    ax1.yaxis.set_major_formatter(formatter_y)
    ax1.set_xticks(valid_lens)
    ax1.set_xlabel('Sequence Length ($N$)')
    ax1.set_ylabel('Forward Pass Latency [ms]')
    ax1.set_title('Computational Scaling')
    ax1.legend()
    ax1.grid(True, which="both", linestyle=':', alpha=0.6)
    
    ax2.plot(valid_lens[:len(mems_mha)], mems_mha, 'o-', color='tab:red', label='FlashAttention')
    ax2.plot(valid_lens[:len(mems_hyb)], mems_hyb, 's-', color='tab:blue', label='Ours')
    ax2.set_xscale('linear')
    ax2.set_yscale('linear')
    ax2.xaxis.set_major_formatter(formatter_x)
    ax2.yaxis.set_major_formatter(formatter_y)
    ax2.set_xticks(valid_lens)
    ax2.set_xlabel('Sequence Length ($N$)')
    ax2.set_ylabel('Peak VRAM [GB]')
    ax2.set_title('Memory Scaling')
    ax2.legend()
    ax2.grid(True, which="both", linestyle=':', alpha=0.6)
    
    min_len = min(len(times_mha), len(times_hyb))
    speedups = [times_mha[i] / times_hyb[i] for i in range(min_len)]
    vl_speed = valid_lens[:min_len]

    ax3.plot(vl_speed, speedups, '^-', color='tab:green', label='Speedup (FlashAttention / Ours)')
    ax3.axhline(1.0, color='black', linestyle='--', linewidth=1)
    ax3.set_xscale('log', base=2)
    ax3.xaxis.set_major_formatter(formatter_x)
    ax3.set_xticks(vl_speed)
    ax3.set_xlabel('Sequence Length ($N$)')
    ax3.set_ylabel('Speedup')
    ax3.set_title('Relative Speedup')
    ax3.legend()
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_path = 'benchmarks/experiments/plots/profiling.pdf'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nProfile plot saved to {plot_path}")

if __name__ == "__main__":
    profile()