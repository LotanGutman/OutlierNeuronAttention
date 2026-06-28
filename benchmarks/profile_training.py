import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.HybridOutlierFactorizedAttention import HybridOutlierFactorizedAttention
from src.config import ModelConfig
from benchmarks.benchmarks_configs import CACHE_PATH

class StandardMHAWrapper(nn.Module):
    def __init__(self, d_head, num_heads):
        super().__init__()
        self.d_head = d_head
        self.num_heads = num_heads
        self.W_q = nn.Identity()
        self.W_k = nn.Identity()
        self.W_v = nn.Identity()
        self.out_proj = nn.Identity()
        
    def forward(self, x):
        B, N, D = x.shape
        scale_factor = self.d_head ** 0.25
        Q = (x / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        K = (x / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = x.view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=1.0)
        Y = Y.transpose(1, 2).reshape(B, N, D)
        return Y

def profile_fwd_bwd(model, x, warmup=3, active=5):
    for _ in range(warmup):
        if x.grad is not None:
            x.grad.zero_()
        out = model(x)
        loss = out.sum()
        loss.backward(retain_graph=True)
            
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(active):
        if x.grad is not None:
            x.grad.zero_()
        out = model(x)
        loss = out.sum()
        loss.backward(retain_graph=True)
            
    end_event.record()
    torch.cuda.synchronize()
    
    latency = start_event.elapsed_time(end_event) / active
    memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return latency, memory

def main():
    device = "cuda"
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    
    model_cfg = ModelConfig(d_model=256, num_heads=4, r=16, chunk_size=64)
    d_head = model_cfg.d_head
    num_heads = model_cfg.num_heads
    
    hofa = HybridOutlierFactorizedAttention(model_cfg).to(device).to(torch.bfloat16)
    hofa.W_q = hofa.W_k = hofa.W_v = hofa.out_proj = nn.Identity()
    hofa.train()
    
    mha = StandardMHAWrapper(d_head, num_heads).to(device).to(torch.bfloat16)
    mha.train()
    
    cache_path = os.path.join(CACHE_PATH, "profile_training_cache.pt")
    force_rerun = False
    
    if os.path.exists(cache_path) and not force_rerun:
        print(f"Loading cached results from {cache_path}")
        results = torch.load(cache_path)
        plot_training_results(results, cache_path)
    else:
        results = []
        
        print("=" * 80)
        print(f"Training Profiling (Fwd+Bwd) (d_model={model_cfg.d_model}, num_heads={num_heads}, r={model_cfg.r})")
        print("=" * 80)
        print(f"{'Seq Len':<10} | {'Model':<6} | {'Latency (ms)':<15} | {'Max VRAM (MB)':<15}")
        print("-" * 80)
        
        for seq_len in seq_lengths:
            x = torch.randn(1, seq_len, model_cfg.d_model, device=device, dtype=torch.bfloat16).requires_grad_(True)
            
            # Profile MHA
            try:
                lat_mha, mem_mha = profile_fwd_bwd(mha, x)
                print(f"{seq_len:<10} | {'MHA':<6} | {lat_mha:<15.2f} | {mem_mha:<15.2f}")
            except Exception as e:
                lat_mha, mem_mha = None, None
                print(f"{seq_len:<10} | {'MHA':<6} | {'FAILED/OOM':<15} | {'-':<15}")
                
            # Profile HOFA
            try:
                lat_hofa, mem_hofa = profile_fwd_bwd(hofa, x)
                print(f"{seq_len:<10} | {'HOFA':<6} | {lat_hofa:<15.2f} | {mem_hofa:<15.2f}")
            except Exception as e:
                lat_hofa, mem_hofa = None, None
                print(f"{seq_len:<10} | {'HOFA':<6} | {'FAILED/OOM':<15} | {'-':<15}")
                
            print("-" * 80)
            results.append((seq_len, lat_mha, mem_mha, lat_hofa, mem_hofa))
            
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(results, cache_path)
        print(f"Results saved to {cache_path}")
        plot_training_results(results, cache_path)


def plot_training_results(results, cache_path):
    import numpy as np
    import os
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    import matplotlib.pyplot as plt

    valid_lens = [r[0] for r in results if r[1] is not None and r[3] is not None]
    if not valid_lens:
        return
        
    times_mha = [r[1] for r in results if r[1] is not None and r[3] is not None]
    times_hyb = [r[3] for r in results if r[1] is not None and r[3] is not None]
    mems_mha = [r[2] for r in results if r[1] is not None and r[3] is not None]
    mems_hyb = [r[4] for r in results if r[1] is not None and r[3] is not None]

    plt.rcParams.update({
        "font.size": 12, 
        "font.family": "serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12
    })

    def format_ticks_x(x, pos):
        return f"{int(x/1024)}k" if x >= 1024 else str(int(x))

    formatter_x = FuncFormatter(format_ticks_x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    
    times_mha_s = [t / 1000 for t in times_mha]
    times_hyb_s = [t / 1000 for t in times_hyb]
    
    style_mha = {"marker": "o", "color": "#D55E00", "linewidth": 2.5, "markersize": 7}
    style_hyb = {"marker": "s", "color": "#0072B2", "linewidth": 2.5, "markersize": 7}
    
    # --- Plot 1: Latency ---
    ax1.plot(valid_lens, times_mha_s, label="MHA (FlashAttention)", **style_mha)
    ax1.plot(valid_lens, times_hyb_s, label="HOFA (Ours)", **style_hyb)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=10) 
    ax1.xaxis.set_major_formatter(formatter_x)
    ax1.set_xticks(valid_lens)
    ax1.set_xlabel("Sequence Length ($N$)")
    ax1.set_ylabel("Fwd+Bwd Latency [s]")
    ax1.set_title("Training Latency Scaling")
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)

    # Convert MB to GB
    mems_mha_gb = [m / 1024 for m in mems_mha]
    mems_hyb_gb = [m / 1024 for m in mems_hyb]

    # --- Plot 2: Memory ---
    ax2.plot(valid_lens, mems_mha_gb, label="MHA (FlashAttention)", **style_mha)
    ax2.plot(valid_lens, mems_hyb_gb, label="HOFA (Ours)", **style_hyb)
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=10) 
    ax2.xaxis.set_major_formatter(formatter_x)
    ax2.set_xticks(valid_lens)
    ax2.set_xlabel("Sequence Length ($N$)")
    ax2.set_ylabel("Peak VRAM [GB]")
    ax2.set_title("Training Memory Scaling")
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85) 
    
    plot_path = "data/plots/profiling/profile_training.pdf"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nProfile plot saved to {plot_path}")
    plt.close()



if __name__ == "__main__":
    main()
