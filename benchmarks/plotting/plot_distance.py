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
        
    if "125M_MHA" not in results or "125M_HOFA" not in results:
        print("Error: Missing required models in cache.")
        return
        
    mha_dist = results["125M_MHA"]["outlier"].numpy() # (num_layers, num_heads)
    hofa_outlier_dist = results["125M_HOFA"]["outlier"].numpy()
    hofa_inlier_dist = results["125M_HOFA"]["inlier"].numpy()
    
    # Average across heads
    mha_avg = mha_dist.mean(axis=1)
    hofa_outlier_avg = hofa_outlier_dist.mean(axis=1)
    hofa_inlier_avg = hofa_inlier_dist.mean(axis=1)
    
    num_layers = len(mha_avg)
    layers = np.arange(1, num_layers + 1)
    
    # Font styling to match paper
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'stix'
    })
    
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", rc={"font.family": "serif"})
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute overall dataset means for legend
    mha_mean_val = mha_avg.mean()
    hofa_outlier_mean_val = hofa_outlier_avg.mean()
    hofa_inlier_mean_val = hofa_inlier_avg.mean()
    
    label_mha = f'MHA Baseline (Mean: {mha_mean_val:.1f} tok)'
    label_out = f'HOFA Outlier, r=16 (Mean: {hofa_outlier_mean_val:.1f} tok)'
    label_in = f'HOFA GLA Inlier (Mean: {hofa_inlier_mean_val:.1f} tok)'
    
    ax.plot(layers, mha_avg, marker='o', linewidth=2.5, color='black', label=label_mha)
    ax.plot(layers, hofa_outlier_avg, marker='s', linewidth=2.5, color='#d62728', label=label_out)
    ax.plot(layers, hofa_inlier_avg, marker='^', linewidth=2.5, linestyle='--', color='#2ca02c', label=label_in)
    
    # Add inset axes for zooming in on the GLA line (0-10 range)
    # Centered horizontally, anchored low
    axins = ax.inset_axes([0.2, 0.15, 0.6, 0.4])
    axins.plot(layers, hofa_inlier_avg, marker='^', linewidth=2.5, linestyle='--', color='#2ca02c')
    
    axins.set_xlim(layers.min(), layers.max())
    axins.set_ylim(-1, 10)
    
    # Remove x-axis tick labels to make it more natural, but keep ticks for vertical grid
    axins.set_xticks(layers)
    axins.set_xticklabels([])
    axins.tick_params(axis='y', which='major', labelsize=10)
    axins.tick_params(axis='x', bottom=False) # Hide the actual tick marks at the bottom if desired
    axins.grid(True, linestyle=':', alpha=0.6)
    
    # Add gray lines showing where the zoom is coming from
    con1 = ConnectionPatch(xyA=(layers.min(), -1), xyB=(layers.min(), hofa_inlier_avg[0]), 
                           coordsA="data", coordsB="data", 
                           axesA=axins, axesB=ax, color="gray", alpha=0.6, lw=1.5)
    ax.add_artist(con1)
    
    con2 = ConnectionPatch(xyA=(layers.max(), -1), xyB=(layers.max(), hofa_inlier_avg[-1]), 
                           coordsA="data", coordsB="data", 
                           axesA=axins, axesB=ax, color="gray", alpha=0.6, lw=1.5)
    ax.add_artist(con2)
    
    # Check for NaN/Inf/Zero
    all_vals = np.concatenate([mha_avg, hofa_outlier_avg, hofa_inlier_avg])
    if not np.all(np.isfinite(all_vals)):
        print("WARNING: Non-finite values detected in distances!")
    if np.all(all_vals == 0):
        print("WARNING: All distances are zero!")
        
    ax.set_xlabel('Layer Depth', fontsize=12)
    ax.set_ylabel('Effective Attention Distance (Tokens)', fontsize=12)
    ax.set_title('Effective Attention Distance by Layer (125M)', fontsize=14)
    
    ax.set_xticks(layers)
    ax.legend(fontsize=11, loc='center right', bbox_to_anchor=(0.98, 0.75))
    
    # Tight layout will handle `fig` since we created a subplots figure
    fig.tight_layout()
    
    out_dir = "data/plots"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "effective_attention_distance.pdf")
    
    fig.savefig(out_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    run_plot_distance()
