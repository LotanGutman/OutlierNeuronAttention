import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from benchmarks.benchmarks_configs import KEffExperimentConfig, CACHE_PATH

MODELS_ORDER = [
    "gpt2",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen2.5-7B"
]


def load_cache():
    cfg = KEffExperimentConfig()
    cache_path = os.path.join(CACHE_PATH, cfg.cache_file_name)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"No cache found at {cache_path}")
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def run_plot_k_eff(cfg=None):
    plot_heatmap_cumsum_all()


def plot_heatmap_cumsum_all():
    out_dir = "data/plots/K_eff"
    os.makedirs(out_dir, exist_ok=True)
    
    results = load_cache()
    available = [m for m in MODELS_ORDER if m in results]
    print(f"Plotting Cumulative Channel Power Heatmaps with d_eff Overlay for {len(available)} models: {available}")
    
    sns.set_theme(style="white")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11
    })
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 10.5))
    mesh = None
    line_ref = None
    line_neff = None
    
    for idx, model_name in enumerate(MODELS_ORDER):
        row, col = divmod(idx, 4)
        ax = axes[row, col]
        short_name = model_name.split('/')[-1]
        
        if model_name not in results:
            ax.set_title(f"{short_name} (Pending)")
            ax.text(0.5, 0.5, "Pending Run", ha='center', va='center', transform=ax.transAxes, color='gray')
            ax.set_xlim(0, 100)
            ax.set_ylim(1, 64)
            ax.set_yscale('log')
            ticks = [1, 2, 4, 8, 16, 32, 64]
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t}/64" for t in ticks] if col == 0 else [])
            if row == 1:
                ax.set_xlabel("Model Depth (%)")
            if col == 0:
                ax.set_ylabel("Routing Dimension ($r / d_h$)")
            continue
            
        m_data = results[model_name]
        num_layers = m_data["num_layers"]
        d_h = m_data["head_dim"]
        C_matrix = m_data["C_matrix"] # [d_h, num_layers]
        Neff_means = m_data["N_eff_means"] # [num_layers]
        
        X_edges = np.linspace(0, 100, num_layers + 1)
        Y_edges = np.arange(1, d_h + 2)
        
        # 1. Base Heatmap: Cumulative channel power C(r)
        mesh = ax.pcolormesh(X_edges, Y_edges, C_matrix, cmap='magma', vmin=0.0, vmax=1.0, shading='auto')
        ax.set_yscale('log')
        ax.set_xlim(0, 100)
        ax.set_ylim(1, d_h)
        
        ticks = [1, 2, 4, 8, 16, 32, 64, 128]
        ticks_filtered = [t for t in ticks if t <= d_h]
        ax.set_yticks(ticks_filtered)
        if col == 0:
            ax.set_yticklabels([f"{t}/{d_h}" for t in ticks_filtered])
        else:
            # For non-first columns, show clean ticks without colliding with the neighbor plot
            ax.set_yticklabels([f"{t}/{d_h}" for t in ticks_filtered], fontsize=9)
            ax.tick_params(axis='y', pad=3)
        
        # 2. Reference line at r = 16 (Exact Softmax allocation)
        line_ref = ax.axhline(y=16, color='#222222', linestyle='--', alpha=0.9, linewidth=1.6, label=r'$r = 16$')
        
        # 3. OVERLAY: d_eff(l) entropic effective channels (standard filled squares 's' with black edge)
        depth_centers = (np.arange(num_layers) + 0.5) / num_layers * 100
        line_deff, = ax.plot(
            depth_centers,
            Neff_means,
            color='#00E5FF',
            marker='s',
            markersize=4.5,
            linewidth=2.0,
            markerfacecolor='#00E5FF',
            markeredgecolor='black',
            markeredgewidth=0.6,
            label=r'$d_{\mathrm{eff}}$'
        )
        
        if row == 1:
            ax.set_xlabel("Model Depth (%)")
        ax.set_title(f"{short_name}", pad=8)
        
        if col == 0:
            ax.set_ylabel("Routing Dimension ($r / d_h$)")
            
    # Shared top legend across all subplots matching profile_shared.pdf style
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.88, hspace=0.28, wspace=0.22)
    if line_ref and line_deff:
        fig.legend(
            handles=[line_ref, line_deff],
            labels=[r'$r = 16$', r'$d_{\mathrm{eff}}$'],
            loc='upper center',
            bbox_to_anchor=(0.47, 0.99),
            ncol=2,
            frameon=False,
            fontsize=13
        )
    
    if mesh is not None:
        cbar_ax = fig.add_axes([0.90, 0.12, 0.015, 0.74])
        cbar = fig.colorbar(mesh, cax=cbar_ax)
        cbar.set_label(r"Cumulative Channel Power $\mathcal{C}(r)$", fontsize=12, labelpad=12)
        
    out_pdf = os.path.join(out_dir, "heatmap_cumsum_normalized.pdf")
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Cumulative Channel Power Heatmaps with d_eff Overlay to {out_pdf}")


if __name__ == "__main__":
    plot_heatmap_cumsum_all()
