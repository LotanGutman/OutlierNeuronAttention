import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from torch.utils.flop_counter import FlopCounterMode
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.HybridOutlierFactorizedAttention import HybridOutlierFactorizedAttention
from benchmarks.benchmarks_configs import CACHE_PATH, TrainingExperimentConfig

class StandardMHAWrapper(nn.Module):
    def __init__(self, d_head, num_heads):
        super().__init__()
        self.d_head = d_head
        self.num_heads = num_heads
        
    def forward(self, x):
        B, N, D = x.shape
        scale_factor = self.d_head ** 0.25
        Q = (x / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        K = (x / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = x.view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=1.0)
        Y = Y.transpose(1, 2).reshape(B, N, D)
        return Y

def count_flops_fwd_bwd(model, x):
    # Warmup and allocate gradients
    if x.grad is not None:
        x.grad.zero_()
    out = model(x)
    loss = out.sum()
    loss.backward(retain_graph=True)
    
    with FlopCounterMode(display=False) as flop_counter:
        if x.grad is not None:
            x.grad.zero_()
        out = model(x)
        loss = out.sum()
        loss.backward(retain_graph=True)
        
    return flop_counter.get_total_flops()

def run_profiling_experiment(config: TrainingExperimentConfig, force_rerun: bool = False):
    device = config.device
    model_cfg = config.model_config
    
    cache_path = os.path.join(CACHE_PATH, config.cache_file_name)
    if os.path.exists(cache_path) and not force_rerun:
        print(f"Loading cached results from {cache_path}")
        results = torch.load(cache_path)
    else:
        results = []
        d_head = model_cfg.d_head
        num_heads = model_cfg.num_heads
        
        hofa = HybridOutlierFactorizedAttention(model_cfg).to(device).to(torch.bfloat16)
        hofa.W_q = hofa.W_k = hofa.W_v = hofa.out_proj = nn.Identity()
        hofa.train()
        
        mha = StandardMHAWrapper(d_head, num_heads).to(device).to(torch.bfloat16)
        mha.train()
        
        print("=" * 80)
        print(f"Training Profiling (Fwd+Bwd FLOPS) (d_model={model_cfg.d_model}, num_heads={num_heads}, r={model_cfg.r})")
        print("=" * 80)
        print(f"{'Seq Len':<10} | {'Model':<6} | {'Total FLOPs':<20}")
        print("-" * 80)
        
        for seq_len in config.seq_lengths:
            x = torch.randn(1, seq_len, model_cfg.d_model, device=device, dtype=torch.bfloat16).requires_grad_(True)
            
            try:
                flops_mha = count_flops_fwd_bwd(mha, x)
                print(f"{seq_len:<10} | {'MHA':<6} | {flops_mha:<20}")
            except Exception as e:
                flops_mha = None
                print(f"{seq_len:<10} | {'MHA':<6} | {'FAILED/OOM':<20}")
                
            try:
                flops_hofa = count_flops_fwd_bwd(hofa, x)
                print(f"{seq_len:<10} | {'HOFA':<6} | {flops_hofa:<20}")
            except Exception as e:
                flops_hofa = None
                print(f"{seq_len:<10} | {'HOFA':<6} | {'FAILED/OOM':<20}")
                
            print("-" * 80)
            results.append((seq_len, flops_mha, flops_hofa))
            
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(results, cache_path)
        print(f"Results saved to {cache_path}")
        
    return results

def plot_training_results(results):
    valid_lens = [r[0] for r in results if r[1] is not None and r[2] is not None]
    if not valid_lens:
        return
        
    flops_mha = [r[1] for r in results if r[1] is not None and r[2] is not None]
    flops_hyb = [r[2] for r in results if r[1] is not None and r[2] is not None]

    # Convert to GFLOPs for better readability on log scale
    flops_mha_g = [f / 1e9 for f in flops_mha]
    flops_hyb_g = [f / 1e9 for f in flops_hyb]

    plt.rcParams.update({
        "font.size": 12, 
        "font.family": "serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12
    })

    def format_ticks_x(x, pos):
        return f"{int(x/1024)}k" if x >= 1024 else str(int(x))

    formatter_x = FuncFormatter(format_ticks_x)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    style_mha = {"marker": "o", "color": "#D55E00", "linewidth": 2.5, "markersize": 7}
    style_hyb = {"marker": "s", "color": "#0072B2", "linewidth": 2.5, "markersize": 7}
    
    ax.plot(valid_lens, flops_mha_g, label="MHA (FlashAttention)", **style_mha)
    ax.plot(valid_lens, flops_hyb_g, label="HOFA, r = 16 (Ours)", **style_hyb)
    
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10) 
    ax.xaxis.set_major_formatter(formatter_x)
    ax.set_xticks(valid_lens)
    
    ax.set_xlabel("Sequence Length ($N$)")
    ax.set_ylabel("Fwd+Bwd Compute [GFLOPs]")
    ax.set_title("Training FLOP Scaling")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)

    ax.legend(loc="upper left", frameon=False, fontsize=12)

    plt.tight_layout()
    
    plot_path = "data/plots/profiling/profile_training.pdf"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nProfile plot saved to {plot_path}")
    plt.close()

def main():
    config = TrainingExperimentConfig()
    results = run_profiling_experiment(config, force_rerun=False)
    plot_training_results(results)

if __name__ == "__main__":
    main()
