"""
Profiling: Triton-Optimized HOFA vs. Standard MHA (FlashAttention).
Measures forward-pass time (ms) and peak VRAM (GB) for seq lengths 512 to 524288.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
from src.HybridOutlierFactorizedAttention import HybridOutlierFactorizedAttention
from src.config import ModelConfig, TrainingConfig
from matplotlib.ticker import FuncFormatter, ScalarFormatter

from benchmarks.benchmarks_configs import CACHE_PATH, PrefillExperimentConfig

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

def run_profiling_experiment(config: PrefillExperimentConfig = PrefillExperimentConfig(), warmup_steps=3, active_steps=10, force_rerun=True, save_results=True):
    model_cfg = config.model_config
    train_cfg = config.train_cfg
    cache_path = os.path.join(CACHE_PATH, config.cache_file_name)
    device = torch.device(config.device)
    torch.manual_seed(train_cfg.seed)
    seq_lengths = config.seq_lengths
    
    model_cfg.refresh_steps = 999999999
    hybrid_attn = HybridOutlierFactorizedAttention(model_cfg).to(device).eval()
    
    # (Static routing doesn't need to update indices)
    
    # Replace the linear layers with Identity to instantly skip the O(N * D^2) compute
    hybrid_attn.W_q = nn.Identity()
    hybrid_attn.W_k = nn.Identity()
    hybrid_attn.W_v = nn.Identity()
            
    mha = StandardMHAWrapper(hybrid_attn)
    
    if os.path.exists(cache_path) and not force_rerun:
        print(f"Loading cached results from {cache_path}")
        data = torch.load(cache_path)
        times_mha, times_hyb, mems_mha, mems_hyb, valid_lens = data
    else:
        times_mha, times_hyb = [], []
        mems_mha, mems_hyb = [], []
        valid_lens = []
        
        print(f"{'seq_len':<8} | {'MHA Time':<10} | {'MHA Mem':<10} | {'Hyb Time':<10} | {'Hyb Mem':<10}")
        print("-" * 60)

        for sl in seq_lengths:
            torch.cuda.empty_cache() # Clean slate only once per seq length
            
            # Scale down x to prevent exponential blowup since Q=K=V now
            x = torch.randn(1, sl, model_cfg.d_model, device=device, dtype=torch.bfloat16) * 0.1
            
            # --- Standard MHA ---
            try:
                with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _ = mha(x)
                    torch.cuda.synchronize()
                    
                    torch.cuda.reset_peak_memory_stats()
                    
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
                with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _ = hybrid_attn(x)
                    torch.cuda.synchronize()
                    
                    torch.cuda.reset_peak_memory_stats()
                    
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

        if save_results:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save((times_mha, times_hyb, mems_mha, mems_hyb, valid_lens), cache_path)

    plot_profile_results(data=(times_mha, times_hyb, mems_mha, mems_hyb, valid_lens), cache_path=cache_path, save_plot=save_results)





def plot_profile_results(data=None, cache_path=None, save_plot=True):
    import numpy as np
    import os
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch

    if data is None:
        if not os.path.exists(cache_path):
            print(f"Cache file {cache_path} not found.")
            return
        data = torch.load(cache_path)

    times_mha, times_hyb, mems_mha, mems_hyb, valid_lens = data

    if not valid_lens:
        return

    plt.rcParams.update({
        'font.size': 12, 
        'font.family': 'serif',
        'axes.titlesize': 14,
        'axes.labelsize': 12
    })

    def format_ticks_x(x, pos):
        return f'{int(x/1024)}k' if x >= 1024 else str(int(x))

    formatter_x = FuncFormatter(format_ticks_x)
    formatter_y = ScalarFormatter()
    formatter_y.set_scientific(False)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
    
    vl1 = valid_lens[:len(times_mha)]
    vl2 = valid_lens[:len(times_hyb)]
    times_mha_s = [t / 1000 for t in times_mha]
    times_hyb_s = [t / 1000 for t in times_hyb]
    
    style_mha = {'marker': 'o', 'color': '#D55E00', 'linewidth': 2.5, 'markersize': 7}
    style_hyb = {'marker': 's', 'color': '#0072B2', 'linewidth': 2.5, 'markersize': 7}
    
    # --- Plot 1: Latency (Linear Y for the massive gap) ---
    ax1.plot(vl1, times_mha_s, label='MHA (FlashAttention)', **style_mha)
    ax1.plot(vl2, times_hyb_s, label='HOFA (Ours)', **style_hyb)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('linear') # Restored to linear for the big visual gap
    ax1.xaxis.set_major_formatter(formatter_x)
    ax1.yaxis.set_major_formatter(formatter_y)
    ax1.set_xticks(valid_lens)
    ax1.set_xlabel('Sequence Length ($N$)')
    ax1.set_ylabel('Forward Pass Latency [s]')
    ax1.set_title('Computational Scaling')
    ax1.grid(True, which="both", linestyle=':', alpha=0.6)

    # Inset zoom
    axins = ax1.inset_axes([0.18, 0.48, 0.42, 0.42]) 
    
    axins.patch.set_facecolor('white')
    axins.patch.set_alpha(0.95)
    for spine in axins.spines.values(): 
        spine.set_linewidth(1.5)
        
    axins.plot(vl1, times_mha_s, **style_mha)
    axins.plot(vl2, times_hyb_s, **style_hyb)
    axins.set_xscale('log', base=2)
    axins.set_yscale('log', base=10) # Log-log for the zoomed area
    
    common_len = min(len(times_mha_s), len(times_hyb_s))
    if common_len > 0:
        zoom_seq_len = min(262144, valid_lens[common_len - 1])
        zoom_idx = valid_lens.index(zoom_seq_len)
        axins.set_xlim(512 * 0.85, zoom_seq_len * 1.15)
        zoom_max_y = max(times_mha_s[zoom_idx], times_hyb_s[zoom_idx])
        axins.set_ylim(min(times_mha_s[0], times_hyb_s[0]) * 0.5, zoom_max_y * 1.5)
        
        axins.xaxis.set_major_formatter(formatter_x)
        mid_val = 1 << (int(np.log2(zoom_seq_len) + 9) // 2)
        axins.set_xticks([512, mid_val, zoom_seq_len])
        axins.tick_params(axis='both', which='major', labelsize=10)
        axins.grid(True, which="both", linestyle=':', alpha=0.4)
        
        y_start = min(times_mha_s[0], times_hyb_s[0])
        con1 = ConnectionPatch(xyA=(512, axins.get_ylim()[0]), xyB=(512, y_start), 
                               coordsA="data", coordsB="data", 
                               axesA=axins, axesB=ax1, color="gray", alpha=0.6, lw=1.5)
        ax1.add_artist(con1)
        
        y_end = max(times_mha_s[zoom_idx], times_hyb_s[zoom_idx])
        con2 = ConnectionPatch(xyA=(zoom_seq_len, axins.get_ylim()[0]), xyB=(zoom_seq_len, y_end), 
                               coordsA="data", coordsB="data", 
                               axesA=axins, axesB=ax1, color="gray", alpha=0.6, lw=1.5)
        ax1.add_artist(con2)

    # Add two-sided arrow for Computational Scaling improvement
    if len(vl1) > 0 and len(vl2) > 0 and vl1[-1] == vl2[-1]:
        max_len = vl1[-1]
        val_mha = times_mha_s[-1]
        val_hyb = times_hyb_s[-1]
        if val_hyb > 0:
            improvement = (val_mha - val_hyb) / val_hyb * 100.0
            ax1.annotate(
                '', xy=(max_len, val_hyb), xytext=(max_len, val_mha),
                arrowprops=dict(arrowstyle="<->", color='black', lw=1.5)
            )
            ax1.text(max_len * 1.15, (val_mha + val_hyb) / 2, f'+{improvement:.1f}%', 
                     color='black', va='center', ha='left', fontsize=12, fontweight='bold')
    
    # --- Plot 2: Memory (Log Y to show parallel scaling lines) ---
    ax2.plot(valid_lens[:len(mems_mha)], mems_mha, label='MHA (FlashAttention)', **style_mha)
    ax2.plot(valid_lens[:len(mems_hyb)], mems_hyb, label='HOFA (Ours)', **style_hyb)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=10) 
    ax2.xaxis.set_major_formatter(formatter_x)
    ax2.yaxis.set_major_formatter(formatter_y)
    ax2.set_xticks(valid_lens)
    ax2.set_xlabel('Sequence Length ($N$)')
    ax2.set_ylabel('Peak VRAM [GB]')
    ax2.set_title('Memory Scaling')
    ax2.grid(True, which="both", linestyle=':', alpha=0.6)
    
    min_len = min(len(times_mha), len(times_hyb))
    speedups = [times_mha[i] / times_hyb[i] for i in range(min_len)]
    vl_speed = valid_lens[:min_len]

    # --- Plot 3: Speedup ---
    ax3.plot(vl_speed, speedups, marker='^', color='#009E73', linewidth=2.5, markersize=7, label='Speedup')
    ax3.axhline(1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.8)
    
    # Calculate interpolated crossover point
    crossover_x = None
    for i in range(len(speedups) - 1):
        if speedups[i] < 1.0 and speedups[i+1] >= 1.0:
            x0, x1 = np.log2(vl_speed[i]), np.log2(vl_speed[i+1])
            y0, y1 = speedups[i], speedups[i+1]
            log_cross = x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0)
            crossover_x = 2 ** log_cross
            break

    if crossover_x is not None:
        ax3.annotate(
            f'Approx. crossover\n$\\sim${int(crossover_x/1000)}k',
            xy=(crossover_x, 1.0),
            xytext=(crossover_x * 1.5, 0.4), 
            arrowprops=dict(arrowstyle="->", color='black', lw=1.2),
            fontsize=11,
            ha='left'
        )
        
    ax3.set_ylim(0, np.ceil(max(speedups) * 5) / 5)
    ax3.set_xscale('log', base=2)
    ax3.xaxis.set_major_formatter(formatter_x)
    ax3.set_xticks(vl_speed)
    ax3.set_xlabel('Sequence Length ($N$)')
    ax3.set_ylabel(rf'Speedup ($\times$ over FlashAttention)')
    ax3.set_title('Hybrid advantage grows\nwith sequence length')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85) 
    if save_plot:
        plot_path = 'data/plots/profiling/profile_prefill.pdf'
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nProfile plot saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    from src.config import ModelConfig
    from benchmarks.benchmarks_configs import PrefillExperimentConfig
    
    model_cfg = ModelConfig(r=16)
    config = PrefillExperimentConfig(model_config=model_cfg)
    run_profiling_experiment(config=config, force_rerun=True)