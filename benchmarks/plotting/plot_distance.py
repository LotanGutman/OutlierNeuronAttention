import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import seaborn as sns
import numpy as np

def run_plot_distance():
    cache_path = "data/experiments_cache/effective_distance_results.pkl"
    if not os.path.exists(cache_path):
        print(f"Error: Could not find {cache_path}. Run benchmark first.")
        return
        
    with open(cache_path, "rb") as f:
        results = pickle.load(f)
        
    # Detect model scale from cache keys
    if "350M_MHA" in results and "350M_HOFA" in results:
        scale_name = "350M"
        mha_key, hofa_key = "350M_MHA", "350M_HOFA"
    elif "125M_MHA" in results and "125M_HOFA" in results:
        scale_name = "125M"
        mha_key, hofa_key = "125M_MHA", "125M_HOFA"
    else:
        # Fallback to any matching pair
        keys = list(results.keys())
        mha_keys = [k for k in keys if "MHA" in k]
        hofa_keys = [k for k in keys if "HOFA" in k]
        if not mha_keys or not hofa_keys:
            print(f"Error: Missing required MHA and HOFA models in cache ({keys}).")
            return
        mha_key, hofa_key = mha_keys[0], hofa_keys[0]
        scale_name = mha_key.split('_')[0]
        
    if results[hofa_key].get("inlier") is None:
        print(f"Error: Cache for {hofa_key} is missing inlier distance data. Please re-run the benchmark.")
        return

    mha_dist = results[mha_key]["outlier"].numpy() # (num_layers, num_heads)
    hofa_outlier_dist = results[hofa_key]["outlier"].numpy()
    hofa_inlier_dist = results[hofa_key]["inlier"].numpy()
    
    # Average across heads
    mha_avg = mha_dist.mean(axis=1)
    hofa_outlier_avg = hofa_outlier_dist.mean(axis=1)
    hofa_inlier_avg = hofa_inlier_dist.mean(axis=1)
    
    num_layers = len(mha_avg)
    layers = np.arange(1, num_layers + 1)
    
    # Font styling to match paper publication rcParams
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 13,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 11,
        'legend.fontsize': 11
    })
    
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    # Compute overall dataset means for legend
    mha_mean_val = mha_avg.mean()
    hofa_outlier_mean_val = hofa_outlier_avg.mean()
    hofa_inlier_mean_val = hofa_inlier_avg.mean()
    
    label_mha = f'MHA Baseline (Mean: {mha_mean_val:.1f} tok)'
    label_out = f'HOFA Outlier (Mean: {hofa_outlier_mean_val:.1f} tok)'
    label_in = f'HOFA GLA Inlier (Mean: {hofa_inlier_mean_val:.1f} tok)'
    
    ax.plot(layers, mha_avg, marker='o', linewidth=2.0, color='#d62728', label=label_mha)
    ax.plot(layers, hofa_outlier_avg, marker='s', linewidth=2.0, color='#1f77b4', label=label_out)
    ax.plot(layers, hofa_inlier_avg, marker='^', linewidth=2.0, linestyle='--', color='#2ca02c', label=label_in)
    
    # Add inset axes for zooming in on the GLA line (0-10 range)
    axins = ax.inset_axes([0.22, 0.16, 0.56, 0.38])
    axins.plot(layers, hofa_inlier_avg, marker='^', linewidth=2.0, linestyle='--', color='#2ca02c')
    
    axins.set_xlim(layers.min(), layers.max())
    axins.set_ylim(-1, max(10, hofa_inlier_avg.max() * 1.2))
    
    # Remove x-axis tick labels to make it clean
    axins.set_xticks(layers)
    axins.set_xticklabels([])
    axins.tick_params(axis='y', which='major', labelsize=10)
    axins.tick_params(axis='x', bottom=False)
    axins.grid(True, linestyle=':', alpha=0.3, color='#e0e0e0')
    
    # Add gray lines showing where the zoom is coming from
    con1 = ConnectionPatch(xyA=(layers.min(), -1), xyB=(layers.min(), hofa_inlier_avg[0]), 
                           coordsA="data", coordsB="data", 
                           axesA=axins, axesB=ax, color="gray", alpha=0.6, lw=1.5)
    ax.add_artist(con1)
    
    con2 = ConnectionPatch(xyA=(layers.max(), -1), xyB=(layers.max(), hofa_inlier_avg[-1]), 
                           coordsA="data", coordsB="data", 
                           axesA=axins, axesB=ax, color="gray", alpha=0.6, lw=1.5)
    ax.add_artist(con2)
    
    ax.set_xlabel('Layer Depth (Layer Index)', fontsize=13)
    ax.set_ylabel('Effective Attention Distance (Tokens)', fontsize=13)
    ax.set_title(f'Effective Attention Distance by Layer ({scale_name})', pad=10)
    
    ax.set_xticks(layers)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(frameon=True, framealpha=0.9, loc='center right', bbox_to_anchor=(0.98, 0.75))
    
    fig.tight_layout()
    
    out_dir = "data/plots"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scale_name}_effective_attention_distance.pdf")
    
    fig.savefig(out_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    run_plot_distance()
