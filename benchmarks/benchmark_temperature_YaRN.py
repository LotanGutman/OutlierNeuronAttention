"""Position-segment pre-softmax temperature scaling benchmark."""

import os
import sys
import gc
import json
import math
import time
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.yarn import YaRNModel, YaRNMode
from training.training_config import YaRNConfig
from benchmarks.benchmarks_configs import TemperatureYaRNConfig, CACHE_PATH
from benchmarks.benchmark_long_context import compute_chunked_loss


def set_model_attention_scale(model: YaRNModel, scale: float) -> int:
    """Sets attention_scale across all attention layers of the model."""
    count = 0
    layers = getattr(getattr(model, "model", model), "layers", [])
    for layer in layers:
        attn = getattr(layer, "attn", None)
        if attn is not None and hasattr(attn, "attention_scale"):
            attn.attention_scale = scale
            count += 1
    return count


def evaluate_segments_for_scale(
    model: YaRNModel,
    data_path: str,
    books: List[dict],
    seq_len: int,
    chunk_size: int,
    scale_factor: float,
    device: str = "cuda",
    max_books: Optional[int] = None,
) -> Dict:
    """Evaluates 1/sqrt(t) scale factor across non-overlapping token chunks."""
    bin_data = np.memmap(data_path, dtype=np.uint16, mode='r')
    target_books = books if max_books is None else books[:max_books]
    num_chunks = seq_len // chunk_size

    # Divide out the effective_mscale^2 already baked into cos_cached and sin_cached
    orig_len = getattr(model, "original_max_seq_len", 1024)
    s_rope = max(1.0, float(seq_len) / float(orig_len))
    baked_mscale = 1.0 + 0.1 * math.log(s_rope)
    actual_attn_scale = (scale_factor / baked_mscale) ** 2
    set_model_attention_scale(model, actual_attn_scale)

    chunk_losses_per_book: List[List[float]] = [[] for _ in range(num_chunks)]
    overall_book_losses: List[float] = []

    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for b_info in target_books:
            start = b_info["start"]
            tok_slice = bin_data[start : start + seq_len + 1].astype(np.int64)
            toks = torch.from_numpy(tok_slice)
            x = toks[:seq_len].unsqueeze(0).to(device)
            y = toks[1 : seq_len + 1].unsqueeze(0).to(device)

            logits, _ = model(x)
            
            # Compute chunk losses
            book_loss_sum = 0.0
            y_flat = y.view(1, seq_len)
            V = logits.shape[-1]

            for c_idx in range(num_chunks):
                c_start = c_idx * chunk_size
                c_end = c_start + chunk_size
                sub_logits = logits[:, c_start:c_end, :].reshape(-1, V)
                sub_y = y_flat[:, c_start:c_end].reshape(-1)
                c_loss = F.cross_entropy(sub_logits, sub_y).item()
                chunk_losses_per_book[c_idx].append(c_loss)
                book_loss_sum += c_loss

            overall_book_losses.append(book_loss_sum / num_chunks)
            del logits, x, y, toks

            if seq_len >= 8192:
                torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    gc.collect()

    mean_overall_loss = float(np.mean(overall_book_losses)) if overall_book_losses else float('nan')
    mean_chunk_losses = [float(np.mean(losses)) if losses else float('nan') for losses in chunk_losses_per_book]
    mean_chunk_ppls = [math.exp(l) if not math.isnan(l) else float('nan') for l in mean_chunk_losses]

    chunk_dict = {}
    for c_idx in range(num_chunks):
        c_start = c_idx * chunk_size
        c_end = c_start + chunk_size
        lbl = f"{c_start // 1024}k-{(c_end) // 1024}k" if c_end >= 1024 else f"{c_start}-{c_end}"
        chunk_dict[lbl] = {
            "loss": mean_chunk_losses[c_idx],
            "ppl": mean_chunk_ppls[c_idx]
        }

    return {
        "scale_factor": scale_factor,
        "overall_loss": mean_overall_loss,
        "overall_ppl": math.exp(mean_overall_loss) if not math.isnan(mean_overall_loss) else float('nan'),
        "num_books": len(overall_book_losses),
        "chunk_results": chunk_dict
    }


def run_temperature_yarn_experiment(config: Optional[TemperatureYaRNConfig] = None):
    if config is None:
        config = TemperatureYaRNConfig()

    device = config.device
    models: List[YaRNConfig] = [config.models] if isinstance(config.models, YaRNConfig) else list(config.models)
    seq_len = config.seq_len
    chunk_size = config.chunk_size
    scale_factors = list(config.scale_factors)

    print("=" * 80)
    print("Position-Segment Pre-Softmax Temperature Scaling Benchmark (YaRN Fig. 5)")
    print(f"Device: {device}")
    print(f"Models: {[m.model.model_name for m in models]}")
    print(f"Sequence Length: {seq_len:,} tokens ({seq_len // chunk_size} chunks of {chunk_size} tokens)")
    print(f"Pre-softmax scale grid (1/sqrt(t)): {scale_factors}")
    print(f"Data source: {config.data_path}")
    print("=" * 80 + "\n")

    if not os.path.exists(config.data_path) or not os.path.exists(config.index_path):
        raise FileNotFoundError(f"PG-19 test dataset not found: {config.data_path} or {config.index_path}")

    with open(config.index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    all_books = index_data.get("books", [])
    eligible_books = [b for b in all_books if b["length"] >= seq_len + 1]
    n_books = len(eligible_books)
    eval_books = eligible_books[:config.max_books] if config.max_books else eligible_books

    print(f"Found {n_books} eligible books (>= {seq_len + 1} tokens). Evaluating {len(eval_books)} books.\n")

    os.makedirs(CACHE_PATH, exist_ok=True)
    cache_file = os.path.join(CACHE_PATH, config.cache_file_name)
    results: Dict[str, Dict] = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    for m in models:
        m_name = m.model.model_name
        if m_name not in results:
            results[m_name] = {}
        if str(seq_len) not in results[m_name]:
            results[m_name][str(seq_len)] = {}

    for yarn_cfg in models:
        model_name = yarn_cfg.model.model_name
        plot_name = yarn_cfg.model.plot_name
        print(f"\nEvaluating {plot_name} ({model_name}) at {seq_len:,} tokens...")

        t0 = time.time()
        yarn_model = YaRNModel(yarn_cfg, device=device, dtype=torch.bfloat16)
        scale_val = max(16.0, float(seq_len) / float(yarn_model.original_max_seq_len))
        ymode = getattr(yarn_cfg, 'yarn_mode', None) or YaRNMode.default()
        n_patched = yarn_model.set_context_length(seq_len, scale=scale_val, yarn_mode=ymode)
        print(f"[{model_name}] Patched {n_patched} layers (RoPE scale s={scale_val:.1f}x, mode={ymode}, init in {time.time() - t0:.1f}s)")

        for sf in scale_factors:
            sf_key = f"{sf:.2f}"
            if not config.force_rerun and sf_key in results[model_name][str(seq_len)]:
                cached_ppl = results[model_name][str(seq_len)][sf_key].get("overall_ppl", float("nan"))
                print(f"[{model_name}] 1/sqrt(t) = {sf:.2f} | Cached PPL: {cached_ppl:.2f}")
                continue

            print(f"[{model_name}] Evaluating 1/sqrt(t) = {sf:.2f} across {len(eval_books)} books...", end="", flush=True)
            t_eval = time.time()
            eval_res = evaluate_segments_for_scale(
                model=yarn_model,
                data_path=config.data_path,
                books=eval_books,
                seq_len=seq_len,
                chunk_size=chunk_size,
                scale_factor=sf,
                device=device,
                max_books=len(eval_books),
            )
            results[model_name][str(seq_len)][sf_key] = eval_res
            with open(cache_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f" Done in {time.time() - t_eval:.1f}s (Overall PPL: {eval_res['overall_ppl']:.2f})")

        del yarn_model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nResults saved to {cache_file}")


def plot_temperature_yarn(config: Optional[TemperatureYaRNConfig] = None):
    """Generates two comparative figures: 1) Percentage change replicating YaRN Fig. 5, 2) Raw perplexities."""
    if config is None:
        config = TemperatureYaRNConfig()

    cache_file = os.path.join(CACHE_PATH, config.cache_file_name)
    if not os.path.exists(cache_file):
        print(f"Cache file {cache_file} not found. Run benchmark first.")
        return

    with open(cache_file, "r") as f:
        results = json.load(f)

    seq_len = config.seq_len
    str_len = str(seq_len)

    available_models = [m for m in results.keys() if str_len in results[m]]
    if not available_models:
        print(f"No results found for seq_len={seq_len} in cache.")
        return

    out_dir = "data/plots/length_extrapolation_temp_scaling"
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 14
    })

    # =========================================================================
    # PLOT 1: Percentage Change replicating YaRN Fig. 5 (starting from 1.0)
    # =========================================================================
    fig1, axes1 = plt.subplots(1, len(available_models), figsize=(7.5 * len(available_models), 6), sharey=True)
    if len(available_models) == 1:
        axes1 = [axes1]

    for ax, m_name in zip(axes1, available_models):
        m_data = results[m_name][str_len]
        scale_keys = [k for k in sorted(m_data.keys(), key=lambda x: float(x)) if float(k) >= 1.0]
        if "1.00" not in m_data or len(scale_keys) < 2:
            print(f"Warning: Baseline 1.00 not found for {m_name}, skipping relative plot.")
            continue

        base_chunks = m_data["1.00"]["chunk_results"]
        chunk_names = list(base_chunks.keys())
        
        # Subsample to 8-9 evenly spaced segments like YaRN Fig. 5
        step = max(1, len(chunk_names) // 8)
        selected_chunk_indices = list(range(0, len(chunk_names), step))
        if (len(chunk_names) - 1) not in selected_chunk_indices:
            selected_chunk_indices.append(len(chunk_names) - 1)

        colors = cm.plasma(np.linspace(0.1, 0.9, len(selected_chunk_indices)))
        x_scales = [float(k) for k in scale_keys]

        for color_i, c_idx in enumerate(selected_chunk_indices):
            c_lbl = chunk_names[c_idx]
            base_ppl = base_chunks[c_lbl]["ppl"]
            deltas = []
            for sk in scale_keys:
                cur_ppl = m_data[sk]["chunk_results"][c_lbl]["ppl"]
                delta_pct = ((cur_ppl - base_ppl) / base_ppl) * 100.0
                deltas.append(delta_pct)

            ax.plot(x_scales, deltas, marker='o', markersize=4, label=c_lbl, color=colors[color_i], linewidth=2.0)

        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        display_name = "350M MHA (16k CPT)" if "MHA" in m_name else "350M HOFA (16k CPT)"
        ax.set_title(display_name, fontweight='bold')
        ax.set_xlabel(r"Attention Pre-Softmax Scaling ($1/\sqrt{t}$)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
        ax.legend(title="Token Segments", loc="lower left", framealpha=0.9)

    axes1[0].set_ylabel(r"Perplexity Change % (from $1/\sqrt{t}=1$ to higher)")
    fig1.suptitle(f"Perplexity Change % of Token Segments at {seq_len // 1024}k Context\n(YaRN Fig. 5 Replication)", fontweight='bold')
    plt.tight_layout()

    pdf_pct_path = os.path.join(out_dir, "position_segment_temperature_scaling.pdf")
    fig1.savefig(pdf_pct_path, dpi=300)
    plt.close(fig1)
    print(f"Saved percentage change plot to: {pdf_pct_path}")

    # =========================================================================
    # PLOT 2: Raw Perplexity vs 1/sqrt(t) (Absolute Values)
    # =========================================================================
    fig2, axes2 = plt.subplots(1, len(available_models), figsize=(7.5 * len(available_models), 6))
    if len(available_models) == 1:
        axes2 = [axes2]

    for ax, m_name in zip(axes2, available_models):
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

        colors = cm.plasma(np.linspace(0.1, 0.9, len(selected_chunk_indices)))
        x_scales = [float(k) for k in scale_keys]

        for color_i, c_idx in enumerate(selected_chunk_indices):
            c_lbl = chunk_names[c_idx]
            ppls = [m_data[sk]["chunk_results"][c_lbl]["ppl"] for sk in scale_keys]
            ax.plot(x_scales, ppls, marker='o', markersize=4, label=c_lbl, color=colors[color_i], linewidth=1.6, alpha=0.85)

        # Also plot overall document PPL
        overall_ppls = [m_data[sk].get("overall_ppl", float("nan")) for sk in scale_keys]
        ax.plot(x_scales, overall_ppls, marker='s', markersize=5, label="Overall Document", color="black", linestyle="--", linewidth=2.2)

        display_name = "350M MHA (16k CPT)" if "MHA" in m_name else "350M HOFA (16k CPT)"
        ax.set_title(display_name, fontweight='bold')
        ax.set_xlabel(r"Attention Pre-Softmax Scaling ($1/\sqrt{t}$)")
        ax.set_ylabel("Perplexity (PPL)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(title="Segments", loc="upper right", framealpha=0.9)

    fig2.suptitle(f"Raw Perplexity of Token Segments at {seq_len // 1024}k Context", fontweight='bold')
    plt.tight_layout()

    pdf_raw_path = os.path.join(out_dir, "raw_perplexity_temperature_scaling.pdf")
    fig2.savefig(pdf_raw_path, dpi=300)
    plt.close(fig2)
    print(f"Saved raw perplexity plot to: {pdf_raw_path}")


if __name__ == "__main__":
    cfg = TemperatureYaRNConfig()
    if "--plot" in sys.argv:
        plot_temperature_yarn(cfg)
    else:
        run_temperature_yarn_experiment(cfg)
        plot_temperature_yarn(cfg)

