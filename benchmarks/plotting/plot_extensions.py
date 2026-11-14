"""Combined PG-19 context extrapolation figure: base models (left) vs. extended pretraining / CPT (right)."""
import json
import os

import matplotlib.pyplot as plt

from benchmarks.benchmarks_configs import CACHE_PATH

STYLES = {  # key -> (label, color, line style)
    "mha": ("MHA", "#d62728", "s--"),
    "hofa": ("HOFA", "#2ca02c", "^-."),
    "alpha": ("HOFA (Learnable $\\beta$)", "#1f77b4", "o-"),
}
ORDER = ["mha", "hofa", "alpha"]


def _kind(name: str) -> str:
    n = name.lower()
    return "alpha" if "alpha" in n else ("mha" if "mha" in n else "hofa")


def _load(cache_name: str, scale: str) -> dict:
    path = os.path.join(CACHE_PATH, cache_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run the corresponding extrapolation benchmark first.")
    with open(path, "r") as f:
        raw = json.load(f)
    return {_kind(k): v for k, v in raw.items()
            if k.startswith(scale) and isinstance(v, dict) and not k.startswith("paired_")}


def plot_extensions(scale: str = "125M", out_dir: str = "data/plots/length_extrapolation", logy: bool = True):
    panels = [
        ("PG19 extrapolation (base models)", f"{scale}_base_extrapolation_64k.json"),
        ("PG19 extrapolation (extended pretraining)", f"{scale}_long_context_extrapolation_results.json"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.1))

    for ax, (title, cache) in zip(axs, panels):
        data = _load(cache, scale)
        for k in ORDER:
            if k not in data:
                continue
            lengths = sorted(int(n) for n in data[k] if n.isdigit())
            vals = [data[k][str(n)]["ppl_mean"] for n in lengths]
            label, color, fmt = STYLES[k]
            ax.plot(lengths, vals, fmt, color=color, linewidth=2.0, markersize=6, label=label, zorder=3)
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks([1024, 2048, 4096, 8192, 16384, 32768, 65536])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x / 1024)}k"))
        ax.minorticks_off() if not logy else None
        ax.set_xlabel("Context Length (Tokens)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.3, color="#e0e0e0")
    axs[0].set_ylabel("Perplexity on PG-19 Test")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=len(labels), frameon=False)
    fig.subplots_adjust(top=0.83, wspace=0.18)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{scale}_extrapolation_combined.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out}")
