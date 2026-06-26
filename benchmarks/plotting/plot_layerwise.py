import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    cache_path = "data/experiments_cache/layerwise_keff_results.pkl"
    plot_dir = "data/plots/K_eff"
    os.makedirs(plot_dir, exist_ok=True)
    
    if not os.path.exists(cache_path):
        print(f"Cache {cache_path} not found. Please run benchmark_keff.py first.")
        return

    with open(cache_path, "rb") as f:
        results = pickle.load(f)
        
    models_to_test = [
        "gpt2",
        "EleutherAI/pythia-410m",
        "meta-llama/Llama-3.2-1B"
    ]
    
    model_meta = {
        "gpt2": {"d": 768},
        "EleutherAI/pythia-410m": {"d": 1024},
        "meta-llama/Llama-3.2-1B": {"d": 2048}
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4'] # Exact (Green), Intermediate (Orange), GLA (Blue)
    labels = ["Exact Routing (K_eff <= 16)", "Intermediate (16 < K_eff <= 100)", "GLA Background (K_eff > 100)"]
    
    for ax, m in zip(axes, models_to_test):
        layerwise_k = results[m]
        
        num_layers = len(layerwise_k)
        layers = np.arange(1, num_layers + 1)
        
        exact_pct = []
        inter_pct = []
        gla_pct = []
        
        for l_idx in range(num_layers):
            k_array = layerwise_k[l_idx]
            total = len(k_array)
            
            exact = np.sum(k_array <= 16) / total * 100
            gla = np.sum(k_array > 100) / total * 100
            inter = 100 - exact - gla
            
            exact_pct.append(exact)
            inter_pct.append(inter)
            gla_pct.append(gla)
            
        exact_pct = np.array(exact_pct)
        inter_pct = np.array(inter_pct)
        gla_pct = np.array(gla_pct)
        
        ax.stackplot(layers, exact_pct, inter_pct, gla_pct, labels=labels, colors=colors, alpha=0.8)
        
        ax.set_title(f"{m.split('/')[-1]} (d={model_meta[m]['d']})", fontsize=12)
        ax.set_xlabel("Layer Index", fontsize=11)
        ax.set_xlim(1, num_layers)
        ax.set_ylim(0, 100)
        
    axes[0].set_ylabel("Percentage of Attention Events (%)", fontsize=11)
    
    # Add a single legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=11)
    
    plt.suptitle("Layer-wise Evolution of Attention Regimes", fontsize=14, y=1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "layerwise_keff.pdf"), format="pdf", bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nLayerwise plot successfully generated in {plot_dir}/")

if __name__ == "__main__":
    main()
