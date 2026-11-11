import os
import json
from typing import Optional, List, Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from benchmarks.benchmarks_configs import TemperatureYaRNConfig, CACHE_PATH
from training.training_config import YaRNConfig


def plot_temperature_segments(config: TemperatureYaRNConfig, results: Dict) -> tuple[Optional[str], Optional[str]]:
    seq_len = config.seq_len
    str_len = str(seq_len)

    available_models = [m for m in results.keys() if str_len in results[m]]
    if not available_models:
        first_m = next(iter(results.keys()), None)
        if first_m and results[first_m]:
            str_len = next(iter(results[first_m].keys()))
            seq_len = int(str_len)
            available_models = [m for m in results.keys() if str_len in results[m]]

    if not available_models:
        print(f"[WARNING] No results found for seq_len={seq_len} in cache.")
        return None, None

    def model_sort_key(m):
        m_lower = m.lower()
        if "mha" in m_lower:
            return 0
        elif "alpha" in m_lower:
            return 2
        return 1
    available_models = sorted(available_models, key=model_sort_key)

    out_dir = "data/plots/length_extrapolation_temp_scaling"
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2.0,
    })

    cmap = plt.cm.viridis

    # 1. Relative perplexity delta from baseline (t=1.0)
    fig1, axes1 = plt.subplots(1, len(available_models), figsize=(6.2 * len(available_models), 4.8), sharey=True, squeeze=False)
    axes1 = axes1[0]

    legend_handles1 = []
    legend_labels1 = []

    for p_idx, (ax, m_name) in enumerate(zip(axes1, available_models)):
        m_data = results[m_name][str_len]
        scale_keys = [k for k in sorted(m_data.keys(), key=lambda x: float(x)) if float(k) >= 1.0]
        if "1.00" not in m_data or len(scale_keys) < 2:
            print(f"[WARNING] Baseline 1.00 not found for {m_name}, skipping relative segment plot.")
            continue

        base_chunks = m_data["1.00"]["chunk_results"]
        chunk_names = list(base_chunks.keys())

        step = max(1, len(chunk_names) // 8)
        selected_chunk_indices = list(range(0, len(chunk_names), step))
        if (len(chunk_names) - 1) not in selected_chunk_indices:
            selected_chunk_indices.append(len(chunk_names) - 1)

        x_scales = [float(k) for k in scale_keys]

        for color_i, c_idx in enumerate(selected_chunk_indices):
            c_lbl = chunk_names[c_idx]
            base_ppl = base_chunks[c_lbl]["ppl"]
            deltas = []
            for sk in scale_keys:
                cur_ppl = m_data[sk]["chunk_results"][c_lbl]["ppl"]
                delta_pct = ((cur_ppl - base_ppl) / base_ppl) * 100.0
                deltas.append(delta_pct)

            c = cmap(color_i / max(1, len(selected_chunk_indices) - 1))
            line, = ax.plot(x_scales, deltas, label=c_lbl, color=c, linewidth=2.0)
            if p_idx == 0:
                legend_handles1.append(line)
                legend_labels1.append(c_lbl)

        m_scale = next((s for s in ["70M", "125M", "350M"] if s in m_name), "")
        if "alpha" in m_name.lower():
            display_name = f"{m_scale} HOFA (Learnable $\\alpha$)" if m_scale else "HOFA (Learnable $\\alpha$)"
        elif "hofa" in m_name.lower():
            display_name = f"{m_scale} HOFA" if m_scale else "HOFA"
        elif "mha" in m_name.lower():
            display_name = f"{m_scale} MHA Baseline" if m_scale else "MHA Baseline"
        else:
            display_name = m_name

        ax.set_title(display_name, fontweight='bold', pad=8)
        ax.set_xlabel(r"$1/\sqrt{t}$")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    axes1[0].set_ylabel(r"Perplexity Change %")
    if legend_handles1:
        fig1.legend(
            handles=legend_handles1,
            labels=legend_labels1,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=min(8, len(legend_handles1)),
            frameon=False,
        )
    plt.tight_layout()

    base_prefix = getattr(config, 'plot_file_name', None) or (config.cache_file_name.replace('.json', '') if config and config.cache_file_name else "temperature_scaling")
    pdf_pct_path = os.path.join(out_dir, f"{base_prefix}_position_segment_temperature_scaling.pdf")
    png_pct_path = os.path.join(out_dir, f"{base_prefix}_position_segment_temperature_scaling.png")
    fig1.savefig(pdf_pct_path, dpi=300, bbox_inches="tight")
    fig1.savefig(png_pct_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # 2. Raw perplexity vs temperature scale
    fig2, axes2 = plt.subplots(1, len(available_models), figsize=(6.2 * len(available_models), 4.8), sharey=True, squeeze=False)
    axes2 = axes2[0]

    legend_handles2 = []
    legend_labels2 = []

    for p_idx, (ax, m_name) in enumerate(zip(axes2, available_models)):
        m_data = results[m_name][str_len]
        scale_keys = sorted(m_data.keys(), key=lambda x: float(x))
        if not scale_keys:
            continue

        base_chunks = m_data[scale_keys[0]]["chunk_results"]
        chunk_names = list(base_chunks.keys())

        step = max(1, len(chunk_names) // 8)
        selected_chunk_indices = list(range(0, len(chunk_names), step))
        if (len(chunk_names) - 1) not in selected_chunk_indices:
            selected_chunk_indices.append(len(chunk_names) - 1)

        x_scales = [float(k) for k in scale_keys]

        for color_i, c_idx in enumerate(selected_chunk_indices):
            c_lbl = chunk_names[c_idx]
            ppls = [m_data[sk]["chunk_results"][c_lbl]["ppl"] for sk in scale_keys]
            c = cmap(color_i / max(1, len(selected_chunk_indices) - 1))
            line, = ax.plot(x_scales, ppls, label=c_lbl, color=c, linewidth=2.0)
            if p_idx == 0:
                legend_handles2.append(line)
                legend_labels2.append(c_lbl)

        m_scale = next((s for s in ["70M", "125M", "350M"] if s in m_name), "")
        if "alpha" in m_name.lower():
            display_name = f"{m_scale} HOFA (Learnable $\\alpha$)" if m_scale else "HOFA (Learnable $\\alpha$)"
        elif "hofa" in m_name.lower():
            display_name = f"{m_scale} HOFA" if m_scale else "HOFA"
        elif "mha" in m_name.lower():
            display_name = f"{m_scale} MHA Baseline" if m_scale else "MHA Baseline"
        else:
            display_name = m_name

        ax.set_title(display_name, fontweight='bold', pad=8)
        ax.set_xlabel(r"$1/\sqrt{t}$")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    from matplotlib.ticker import ScalarFormatter
    for ax in axes2:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(25, 180)
        ax.set_yticks([30, 40, 50, 75, 100, 150])

    axes2[0].set_ylabel("Perplexity (Log Scale)")

    if legend_handles2:
        fig2.legend(
            handles=legend_handles2,
            labels=legend_labels2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=min(8, len(legend_handles2)),
            frameon=False,
        )
    plt.tight_layout()

    pdf_raw_path = os.path.join(out_dir, f"{base_prefix}_perplexity_temperature_scaling.pdf")
    png_raw_path = os.path.join(out_dir, f"{base_prefix}_perplexity_temperature_scaling.png")
    fig2.savefig(pdf_raw_path, dpi=300, bbox_inches="tight")
    fig2.savefig(png_raw_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    return pdf_pct_path, pdf_raw_path


def plot_temperature_yarn(config: Optional[TemperatureYaRNConfig] = None):
    if config is None:
        config = TemperatureYaRNConfig()

    cache_file = os.path.join(CACHE_PATH, config.cache_file_name)
    if not os.path.exists(cache_file):
        fallback = os.path.join(CACHE_PATH, "temperature_yarn_results.json")
        if os.path.exists(fallback):
            cache_file = fallback
        else:
            print(f"[ERROR] Cache file {cache_file} not found. Run benchmark first.")
            return

    with open(cache_file, "r") as f:
        results = json.load(f)

    plot_temperature_segments(config, results)
