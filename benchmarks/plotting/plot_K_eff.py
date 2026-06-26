import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

MODEL_METADATA = {
    "gpt2": 768,
    "EleutherAI/pythia-410m": 1024,
    "meta-llama/Llama-3.2-1B": 2048,
    "meta-llama/Llama-3.2-3B": 3072
}

def plot_heatmap(results, max_r=128, normalize_y=False):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=not normalize_y)
    sns.set_theme(style="whitegrid")
    
    models = list(results.keys())
    
    for idx, model_name in enumerate(models):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        
        layerwise_cumsum = results[model_name]
        d_model = MODEL_METADATA.get(model_name, 1024)
        
        current_max_r = d_model if normalize_y else max_r
        current_max_r = min(current_max_r, 1024) # Because sequence length was 1024
        
        num_layers = len(layerwise_cumsum)
        Z = np.zeros((current_max_r, num_layers))
        
        for layer_idx in range(num_layers):
            Z[:, layer_idx] = layerwise_cumsum[layer_idx][:current_max_r]
            
        X_edges = np.linspace(0, 100, num_layers + 1)
        Y_edges = np.arange(1, current_max_r + 2, dtype=np.float64)
        
        if normalize_y:
            Y_edges = Y_edges / d_model
            
        mesh = ax.pcolormesh(X_edges, Y_edges, Z, cmap='magma', vmin=0.0, vmax=1.0, shading='auto')
        
        ax.set_yscale('log')
        ax.set_xlim(0, 100)
        
        if not normalize_y:
            ticks = [1, 2, 4, 8, 16, 32, 64, 128]
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticks)
            ax.axhline(y=16, color='white', linestyle='--', alpha=0.8, linewidth=2)
            ax.set_ylim(1, max_r)
        else:
            ticks = [2**i for i in range(0, int(np.log2(d_model))+1)]
            ticks_scaled = [t / d_model for t in ticks if t <= current_max_r]
            ax.set_yticks(ticks_scaled)
            # Use concise labels: '1/d', '2/d', etc.
            filtered_labels = []
            for t in ticks:
                if t <= current_max_r:
                    if t in [1, 16, 128, 1024, 2048, 3072] or t == d_model or t == current_max_r:
                        filtered_labels.append(f"{t}/{d_model}")
                    else:
                        filtered_labels.append("")
            ax.set_yticklabels(filtered_labels)
            
            ax.axhline(y=16/d_model, color='white', linestyle='--', alpha=0.8, linewidth=2)
            ax.set_ylim(1/d_model, current_max_r / d_model)
            
        if row == 1:
            ax.set_xlabel("Model Depth (%)")
        short_name = model_name.split('/')[-1]
        ax.set_title(f"{short_name} ($d_{{model}}={d_model}$)")
        
        if col == 0 or normalize_y:
            if not normalize_y:
                ax.set_ylabel("Routing Dimension ($r$)")
            else:
                ax.set_ylabel("Routing Dimension Ratio ($r / d_{model}$)")

    fig.subplots_adjust(right=0.90, hspace=0.3, wspace=0.3)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label("Cumulative Attention Mass Fraction")

    plt.tight_layout(rect=[0, 0, 0.90, 1])
    
    suffix = "normalized" if normalize_y else "absolute"
    out_path = f"data/plots/K_eff/heatmap_cumsum_{suffix}.pdf"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    cache_path = "data/experiments_cache/layerwise_cumsum_results.pkl"
    with open(cache_path, "rb") as f:
        results = pickle.load(f)
        
    plot_heatmap(results, max_r=128, normalize_y=False)
    plot_heatmap(results, normalize_y=True)
