import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from matplotlib.patches import ConnectionPatch
from benchmarks.benchmarks_configs import PrefillExperimentConfig, DecodeExperimentConfig

def plot_shared_profiling():
    prefill_config = PrefillExperimentConfig()
    decode_config = DecodeExperimentConfig()
    
    prefill_cache_path = os.path.join("data/experiments_cache", prefill_config.cache_file_name)
    decode_cache_path = os.path.join("data/experiments_cache", decode_config.cache_file_name)

    if not os.path.exists(prefill_cache_path) or not os.path.exists(decode_cache_path):
        print(f"Error: One or both cache files are missing.")
        print(f"Prefill cache: {prefill_cache_path} (Exists: {os.path.exists(prefill_cache_path)})")
        print(f"Decode cache: {decode_cache_path} (Exists: {os.path.exists(decode_cache_path)})")
        print("Please run `python main.py profile --prefill` and `python main.py profile --decode` first.")
        return

    # Load Prefill Data
    prefill_data = torch.load(prefill_cache_path)
    p_times_mha, p_times_hyb, p_flops_mha, p_flops_hyb, p_valid_lens = prefill_data
    
    # Load Decode Data
    decode_data = torch.load(decode_cache_path)
    d_valid_lens, d_times_mha, d_times_hyb, d_cache_mha_mb, d_cache_hyb_mb = decode_data
    
    # Filter Decode Data based on config (so it doesn't plot lengths beyond what's in the current config)
    filtered_idx = [i for i, l in enumerate(d_valid_lens) if l in decode_config.seq_lengths]
    d_valid_lens = [d_valid_lens[i] for i in filtered_idx]
    d_times_mha = [d_times_mha[i] for i in filtered_idx]
    d_times_hyb = [d_times_hyb[i] for i in filtered_idx]
    d_cache_mha_mb = [d_cache_mha_mb[i] for i in filtered_idx]
    d_cache_hyb_mb = [d_cache_hyb_mb[i] for i in filtered_idx]

    plt.rcParams.update({
        'font.size': 11, 
        'font.family': 'serif',
        'axes.titlesize': 13,
        'axes.labelsize': 11
    })

    def format_ticks_x(x, pos):
        return f'{int(x/1024)}k' if x >= 1024 else str(int(x))

    formatter_x = FuncFormatter(format_ticks_x)
    def format_ticks_y(y, pos):
        if y == 0:
            return '0'
        elif y >= 1000:
            val = y / 1000
            return f'{val:g}k'
        return f'{y:g}'

    formatter_y = FuncFormatter(format_ticks_y)

    fig, axs = plt.subplots(2, 3, figsize=(18, 11))
    
    style_mha = {'marker': 'o', 'color': '#d62728', 'linewidth': 2.0, 'markersize': 6}
    style_hyb = {'marker': 's', 'color': '#1f77b4', 'linewidth': 2.0, 'markersize': 6}
    
    # ==========================================
    # TOP ROW: DECODE (Throughput, KV Cache, Speedup)
    # ==========================================
    ax_dec_thru = axs[0, 0]
    ax_dec_mem = axs[0, 1]
    ax_dec_spd = axs[0, 2]

    valid_mha_lens = [l for i, l in enumerate(d_valid_lens) if not np.isnan(d_times_mha[i])]
    valid_mha_times = [t for t in d_times_mha if not np.isnan(t)]
    valid_hyb_lens = [l for i, l in enumerate(d_valid_lens) if not np.isnan(d_times_hyb[i])]
    valid_hyb_times = [t for t in d_times_hyb if not np.isnan(t)]
    
    ax_dec_thru.plot(valid_mha_lens, valid_mha_times, label='MHA (FlashAttention)', **style_mha)
    ax_dec_thru.plot(valid_hyb_lens, valid_hyb_times, label='HOFA, r = 16', **style_hyb)
    
    ax_dec_thru.set_xscale('log', base=2)
    ax_dec_thru.xaxis.set_major_formatter(formatter_x)
    ax_dec_thru.yaxis.set_major_formatter(formatter_y)
    tick_lens = [l for l in d_valid_lens if l != 196608]
    ax_dec_thru.set_xticks(tick_lens)
    ax_dec_thru.set_ylabel('Tokens / sec')
    ax_dec_thru.set_title('Decoding Throughput')
    ax_dec_thru.grid(True, linestyle=':', alpha=0.3, color='#e0e0e0')

    # Non-overlapping inset zoom for Decoding Throughput
    ax_dec_thru_ins = ax_dec_thru.inset_axes([0.12, 0.45, 0.38, 0.38])
    ax_dec_thru_ins.patch.set_facecolor('white')
    ax_dec_thru_ins.patch.set_alpha(0.95)
    for spine in ax_dec_thru_ins.spines.values():
        spine.set_linewidth(1.2)
        
    ax_dec_thru_ins.plot(valid_mha_lens, valid_mha_times, **style_mha)
    ax_dec_thru_ins.plot(valid_hyb_lens, valid_hyb_times, **style_hyb)
    ax_dec_thru_ins.set_xscale('log', base=2)
    ax_dec_thru_ins.set_yscale('log', base=10)
    
    zoom_start_len = 65536
    zoom_end_len = max(valid_mha_lens) if valid_mha_lens else 131072
    if zoom_start_len in d_valid_lens and zoom_end_len in d_valid_lens:
        ax_dec_thru_ins.set_xlim(zoom_start_len * 0.85, zoom_end_len * 1.15)
        idx_start = d_valid_lens.index(zoom_start_len)
        idx_end = d_valid_lens.index(zoom_end_len)
        min_y = min(valid_mha_times[idx_end], valid_hyb_times[idx_end]) if idx_end < len(valid_mha_times) and idx_end < len(valid_hyb_times) else 10
        max_y = max(valid_mha_times[idx_start], valid_hyb_times[idx_start]) if idx_start < len(valid_mha_times) and idx_start < len(valid_hyb_times) else 1000
        ax_dec_thru_ins.set_ylim(min_y * 0.5, max_y * 1.5)
        ax_dec_thru_ins.xaxis.set_major_formatter(formatter_x)
        ticks_ins = [zoom_start_len]
        mid_len = zoom_start_len * 4
        if mid_len < zoom_end_len:
            ticks_ins.append(mid_len)
        ticks_ins.append(zoom_end_len)
        ax_dec_thru_ins.set_xticks(ticks_ins)
        ax_dec_thru_ins.tick_params(axis='both', which='major', labelsize=10)
        ax_dec_thru_ins.grid(True, which="both", linestyle=':', alpha=0.4)
        
        if idx_end < len(valid_mha_times) and idx_end < len(valid_hyb_times):
            val_mha = valid_mha_times[idx_end]
            val_hyb = valid_hyb_times[idx_end]
            improvement = (val_hyb - val_mha) / val_mha * 100.0
            ax_dec_thru_ins.annotate(
                '', xy=(zoom_end_len, val_hyb), xytext=(zoom_end_len, val_mha),
                arrowprops=dict(arrowstyle="<->", color='black', lw=1.5)
            )
            ax_dec_thru_ins.text(zoom_end_len * 1.15, (val_mha * val_hyb) ** 0.5, f'+{improvement:.1f}%', 
                         color='black', va='center', ha='left', fontsize=12, fontweight='bold', clip_on=False)
                         
        y_start = ax_dec_thru_ins.get_ylim()[0]
        con1 = ConnectionPatch(xyA=(zoom_start_len, ax_dec_thru_ins.get_ylim()[0]), xyB=(zoom_start_len, y_start), 
                               coordsA="data", coordsB="data", 
                               axesA=ax_dec_thru_ins, axesB=ax_dec_thru, color="gray", alpha=0.6, lw=1.5)
        ax_dec_thru.add_artist(con1)
        
        y_end = ax_dec_thru_ins.get_ylim()[0]
        con2 = ConnectionPatch(xyA=(zoom_end_len, ax_dec_thru_ins.get_ylim()[0]), xyB=(zoom_end_len, y_end), 
                               coordsA="data", coordsB="data", 
                               axesA=ax_dec_thru_ins, axesB=ax_dec_thru, color="gray", alpha=0.6, lw=1.5)
        ax_dec_thru.add_artist(con2)

    # Decode KV Cache Size
    ax_dec_mem.plot(d_valid_lens, d_cache_mha_mb, **style_mha)
    ax_dec_mem.plot(d_valid_lens, d_cache_hyb_mb, **style_hyb)
    ax_dec_mem.set_xscale('log', base=2)
    ax_dec_mem.xaxis.set_major_formatter(formatter_x)
    ax_dec_mem.set_xticks(tick_lens)
    ax_dec_mem.set_ylabel('KV Cache Size (GB)')
    ax_dec_mem.set_title('KV Cache Footprint')
    ax_dec_mem.grid(True, linestyle=':', alpha=0.3, color='#e0e0e0')

    max_len = max(d_valid_lens)
    idx_max = d_valid_lens.index(max_len)
    if not np.isnan(d_cache_mha_mb[idx_max]) and not np.isnan(d_cache_hyb_mb[idx_max]) and d_cache_hyb_mb[idx_max] > 0:
        val_mha = d_cache_mha_mb[idx_max]
        val_hyb = d_cache_hyb_mb[idx_max]
        improvement = (val_mha - val_hyb) / val_mha * 100.0
        ax_dec_mem.annotate(
            '', xy=(max_len, val_hyb), xytext=(max_len, val_mha),
            arrowprops=dict(arrowstyle="<->", color='black', lw=1.5)
        )
        ax_dec_mem.text(max_len * 1.15, (val_mha + val_hyb) / 2, f'-{improvement:.1f}%', 
                 color='black', va='center', ha='left', fontsize=12, fontweight='bold', clip_on=False)

    # Decode Speedup
    d_speedups = [d_times_hyb[i] / d_times_mha[i] for i in range(len(d_valid_lens))]
    valid_speedup_lens = [l for i, l in enumerate(d_valid_lens) if not np.isnan(d_speedups[i])]
    valid_speedups = [s for s in d_speedups if not np.isnan(s)]
    
    ax_dec_spd.plot(valid_speedup_lens, valid_speedups, marker='^', color='#009E73', linewidth=2.5, markersize=7, label='Speedup')
    ax_dec_spd.axhline(1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.8)
    
    crossover_x = None
    for i in range(len(valid_speedups) - 1):
        if valid_speedups[i] < 1.0 and valid_speedups[i+1] >= 1.0:
            x0, x1 = np.log2(valid_speedup_lens[i]), np.log2(valid_speedup_lens[i+1])
            y0, y1 = valid_speedups[i], valid_speedups[i+1]
            log_cross = x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0)
            crossover_x = 2 ** log_cross
            break

    if crossover_x is not None:
        ax_dec_spd.annotate(
            f'Approx. crossover\n$\\sim${round(crossover_x/1000)}k',
            xy=(crossover_x, 1.0),
            xytext=(crossover_x * 2.0, 0.4), 
            arrowprops=dict(arrowstyle="->", color='black', lw=1.2),
            fontsize=11,
            ha='left'
        )

    ax_dec_spd.set_ylim(0, np.ceil(max(valid_speedups) * 5) / 5 if valid_speedups else 5)
    ax_dec_spd.set_xscale('log', base=2)
    ax_dec_spd.xaxis.set_major_formatter(formatter_x)
    tick_speed_lens = [l for l in valid_speedup_lens if l != 196608]
    ax_dec_spd.set_xticks(tick_speed_lens)
    ax_dec_spd.set_ylabel(rf'Speedup ($\times$ over MHA)')
    ax_dec_spd.set_title('Decoding Speedup over MHA')
    ax_dec_spd.grid(True, linestyle=':', alpha=0.3, color='#e0e0e0')

    # ==========================================
    # BOTTOM ROW: PREFILL (Latency, FLOPs, Speedup)
    # ==========================================
    ax_pre_lat = axs[1, 0]
    ax_pre_flop = axs[1, 1]
    ax_pre_spd = axs[1, 2]
    
    vl1 = p_valid_lens[:len(p_times_mha)]
    vl2 = p_valid_lens[:len(p_times_hyb)]
    p_times_mha_s = [t / 1000 for t in p_times_mha]
    p_times_hyb_s = [t / 1000 for t in p_times_hyb]

    # Prefill Latency
    ax_pre_lat.plot(vl1, p_times_mha_s, **style_mha)
    ax_pre_lat.plot(vl2, p_times_hyb_s, **style_hyb)
    ax_pre_lat.set_xscale('log', base=2)
    ax_pre_lat.set_yscale('linear')
    ax_pre_lat.xaxis.set_major_formatter(formatter_x)
    ax_pre_lat.yaxis.set_major_formatter(formatter_y)
    ax_pre_lat.set_xticks(p_valid_lens)
    ax_pre_lat.set_xlabel('Context Length ($N$)')
    ax_pre_lat.set_ylabel('Forward Pass Latency [s]')
    ax_pre_lat.set_title('Prefill Latency')
    ax_pre_lat.grid(True, which="both", linestyle=':', alpha=0.3, color='#e0e0e0')

    # Inset zoom
    ax_pre_lat_ins = ax_pre_lat.inset_axes([0.18, 0.48, 0.42, 0.42]) 
    ax_pre_lat_ins.patch.set_facecolor('white')
    ax_pre_lat_ins.patch.set_alpha(0.95)
    for spine in ax_pre_lat_ins.spines.values(): 
        spine.set_linewidth(1.5)
        
    ax_pre_lat_ins.plot(vl1, p_times_mha_s, **style_mha)
    ax_pre_lat_ins.plot(vl2, p_times_hyb_s, **style_hyb)
    ax_pre_lat_ins.set_xscale('log', base=2)
    ax_pre_lat_ins.set_yscale('log', base=10)
    
    common_len = min(len(p_times_mha_s), len(p_times_hyb_s))
    if common_len > 0:
        zoom_seq_len = min(65536, p_valid_lens[common_len - 1])
        zoom_idx = p_valid_lens.index(zoom_seq_len)
        ax_pre_lat_ins.set_xlim(512 * 0.85, zoom_seq_len * 1.15)
        zoom_max_y = max(p_times_mha_s[zoom_idx], p_times_hyb_s[zoom_idx])
        ax_pre_lat_ins.set_ylim(min(p_times_mha_s[0], p_times_hyb_s[0]) * 0.5, zoom_max_y * 1.5)
        
        ax_pre_lat_ins.xaxis.set_major_formatter(formatter_x)
        mid_val = 1 << (int(np.log2(zoom_seq_len) + 9) // 2)
        ax_pre_lat_ins.set_xticks([512, mid_val, zoom_seq_len])
        ax_pre_lat_ins.tick_params(axis='both', which='major', labelsize=10)
        ax_pre_lat_ins.grid(True, which="both", linestyle=':', alpha=0.4)
        
        y_start = min(p_times_mha_s[0], p_times_hyb_s[0])
        con1 = ConnectionPatch(xyA=(512, ax_pre_lat_ins.get_ylim()[0]), xyB=(512, y_start), 
                               coordsA="data", coordsB="data", 
                               axesA=ax_pre_lat_ins, axesB=ax_pre_lat, color="gray", alpha=0.6, lw=1.5)
        ax_pre_lat.add_artist(con1)
        
        y_end = max(p_times_mha_s[zoom_idx], p_times_hyb_s[zoom_idx])
        con2 = ConnectionPatch(xyA=(zoom_seq_len, ax_pre_lat_ins.get_ylim()[0]), xyB=(zoom_seq_len, y_end), 
                               coordsA="data", coordsB="data", 
                               axesA=ax_pre_lat_ins, axesB=ax_pre_lat, color="gray", alpha=0.6, lw=1.5)
        ax_pre_lat.add_artist(con2)

    if len(vl1) > 0 and len(vl2) > 0 and vl1[-1] == vl2[-1]:
        max_len = vl1[-1]
        val_mha = p_times_mha_s[-1]
        val_hyb = p_times_hyb_s[-1]
        if val_hyb > 0:
            improvement = (val_mha - val_hyb) / val_hyb * 100.0
            ax_pre_lat.annotate(
                '', xy=(max_len, val_hyb), xytext=(max_len, val_mha),
                arrowprops=dict(arrowstyle="<->", color='black', lw=1.5)
            )
            ax_pre_lat.text(max_len * 1.15, (val_mha + val_hyb) / 2, f'+{improvement:.1f}%', 
                     color='black', va='center', ha='left', fontsize=12, fontweight='bold')

    # Prefill FLOPs
    flops_mha_g = [f / 1e9 for f in p_flops_mha]
    flops_hyb_g = [f / 1e9 for f in p_flops_hyb]

    ax_pre_flop.plot(p_valid_lens[:len(flops_mha_g)], flops_mha_g, **style_mha)
    ax_pre_flop.plot(p_valid_lens[:len(flops_hyb_g)], flops_hyb_g, **style_hyb)
    ax_pre_flop.set_xscale('log', base=2)
    ax_pre_flop.set_yscale('log', base=10) 
    ax_pre_flop.xaxis.set_major_formatter(formatter_x)
    ax_pre_flop.set_xticks(p_valid_lens)
    ax_pre_flop.set_xlabel('Context Length ($N$)')
    ax_pre_flop.set_ylabel('Forward Compute [GFLOPs]')
    ax_pre_flop.set_title('Prefill Compute (FLOPs)')
    ax_pre_flop.grid(True, which="both", linestyle=':', alpha=0.6)
    
    # Prefill Speedup
    min_len = min(len(p_times_mha), len(p_times_hyb))
    p_speedups = [p_times_mha[i] / p_times_hyb[i] for i in range(min_len)]
    vl_speed = p_valid_lens[:min_len]

    ax_pre_spd.plot(vl_speed, p_speedups, marker='^', color='#009E73', linewidth=2.5, markersize=7, label='Speedup')
    ax_pre_spd.axhline(1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.8)
    
    crossover_x = None
    for i in range(len(p_speedups) - 1):
        if p_speedups[i] < 1.0 and p_speedups[i+1] >= 1.0:
            x0, x1 = np.log2(vl_speed[i]), np.log2(vl_speed[i+1])
            y0, y1 = p_speedups[i], p_speedups[i+1]
            log_cross = x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0)
            crossover_x = 2 ** log_cross
            break

    if crossover_x is not None:
        ax_pre_spd.annotate(
            f'Approx. crossover\n$\\sim${round(crossover_x/1000)}k',
            xy=(crossover_x, 1.0),
            xytext=(crossover_x * 1.5, 0.4), 
            arrowprops=dict(arrowstyle="->", color='black', lw=1.2),
            fontsize=11,
            ha='left'
        )
        
    ax_pre_spd.set_ylim(0, np.ceil(max(p_speedups) * 5) / 5 if p_speedups else 5)
    ax_pre_spd.set_xscale('log', base=2)
    ax_pre_spd.xaxis.set_major_formatter(formatter_x)
    ax_pre_spd.set_xticks(vl_speed)
    ax_pre_spd.set_xlabel('Context Length ($N$)')
    ax_pre_spd.set_ylabel(rf'Speedup ($\times$ over MHA)')
    ax_pre_spd.set_title('Prefill Speedup over MHA')
    ax_pre_spd.grid(True, linestyle=':', alpha=0.6)

    # Shared Legend
    handles, labels = ax_dec_thru.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.2, wspace=0.25)
    
    plot_path = 'data/plots/profiling/profile_shared.pdf'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nShared profile plot saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    plot_shared_profiling()
