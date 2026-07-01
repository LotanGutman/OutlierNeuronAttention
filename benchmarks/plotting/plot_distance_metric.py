import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from benchmarks.benchmarks_configs import CACHE_PATH


def plot_distance_metric(model_name="30M", cache_file="distance_metric_results.pt"):
    cache_path = os.path.join(CACHE_PATH, cache_file)
    if not os.path.exists(cache_path):
        print(f"Cache {cache_path} not found. Run benchmarks/compute_distance_metric.py first.")
        return

    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    dist_means = data["means"]  # (num_layers, num_heads)
    dist_stds = data["stds"]
    n_layers, n_heads = dist_means.shape

    plt.rcParams.update({"font.size": 11, "font.family": "serif"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # -- heatmap --
    vmin, vmax = dist_means.min(), dist_means.max()
    hm = ax1.imshow(dist_means, aspect="auto", cmap="viridis",
                    vmin=vmin, vmax=vmax)
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Layer")
    ax1.set_title("Effective Attention Distance")
    ax1.set_xticks(range(n_heads))
    ax1.set_yticks(range(n_layers))
    ax1.set_xticklabels([str(h) for h in range(n_heads)])
    ax1.set_yticklabels([str(l) for l in range(n_layers)])

    for li in range(n_layers):
        for hi in range(n_heads):
            val = dist_means[li, hi]
            color = "white" if val > (vmin + vmax) / 2 else "black"
            ax1.text(hi, li, f"{val:.0f}", ha="center", va="center",
                     fontsize=8, color=color)

    cbar = fig.colorbar(hm, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Mean distance (tokens)")

    # -- line plot --
    layer_means = dist_means.mean(axis=1)
    layer_stds = dist_means.std(axis=1)
    xs = np.arange(n_layers)

    ax2.plot(xs, layer_means, "o-", color="#0072B2", linewidth=2, markersize=6)
    ax2.fill_between(xs, layer_means - layer_stds, layer_means + layer_stds,
                     alpha=0.2, color="#0072B2")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Effective Distance (tokens)")
    ax2.set_title("Layer-wise Distance")
    ax2.set_xticks(xs)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(f"{model_name} HOFA — Effective Attention Distance", fontsize=13, y=1.02)
    plt.tight_layout()

    out_dir = "data/plots/language_distance_metric"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_distance_metric()