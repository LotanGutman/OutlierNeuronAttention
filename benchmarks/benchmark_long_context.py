"""
Context Length Extrapolation with Canonical YaRN
"""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import json
import math
import time
from typing import List, Dict, Optional, Union, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.yarn import YaRNModel
from training.training_config import YaRNConfig
from training.data_utils import FastTokenLoader
from benchmarks.benchmarks_configs import LongContextExtrapolationConfig, CACHE_PATH

def bootstrap_ci(losses: List[float], num_resamples: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Computes mean and 95% bootstrap confidence intervals over sequences."""
    if not losses:
        return float('nan'), float('nan'), float('nan')
    if len(losses) == 1:
        return losses[0], losses[0], losses[0]

    losses_arr = np.array(losses, dtype=np.float64)
    n = len(losses_arr)
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(num_resamples):
        sample = rng.choice(losses_arr, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.sort(boot_means)
    low_idx = int((alpha / 2.0) * num_resamples)
    high_idx = int((1.0 - alpha / 2.0) * num_resamples)
    return float(np.mean(losses_arr)), float(boot_means[low_idx]), float(boot_means[high_idx])


def compute_chunked_loss(logits: torch.Tensor, targets: torch.Tensor, chunk_size: int = 2048) -> float:
    """Computes mean cross-entropy loss in sequence slices to prevent PyTorch's
    internal float32 upcast from allocating a ton of VRAM at extended lengths."""
    B, N, V = logits.shape
    total_loss_sum = 0.0
    y_flat = targets.view(B, N)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        sub_logits = logits[:, start:end, :].reshape(-1, V)
        sub_targets = y_flat[:, start:end].reshape(-1)
        sub_loss = F.cross_entropy(sub_logits, sub_targets, reduction='sum')
        total_loss_sum += sub_loss.item()
    return total_loss_sum / (B * N)


def evaluate_long_context_metrics(
    model: YaRNModel,
    val_cache_path: str,
    seq_len: int,
    total_tokens_target: int = 500000,
    device: str = 'cuda'
) -> Dict:
    """
    Evaluates pooled validation perplexity with bootstrap CI.
    """
    batch_size = 1
    num_batches = max(1, total_tokens_target // seq_len)
    loader = FastTokenLoader(val_cache_path, batch_size, seq_len, start_idx=0)

    seq_losses: List[float] = []

    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(num_batches):
            x, y, _ = loader.get_batch()
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)
            seq_loss = compute_chunked_loss(logits, y, chunk_size=2048)

            if not (math.isnan(seq_loss) or math.isinf(seq_loss)):
                seq_losses.append(seq_loss)

            del logits, x, y
            if seq_len >= 8192:
                torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    gc.collect()

    mean_loss, low_loss, high_loss = bootstrap_ci(seq_losses)
    ppl_mean = math.exp(mean_loss) if not math.isnan(mean_loss) else float('nan')
    ppl_low = math.exp(low_loss) if not math.isnan(low_loss) else float('nan')
    ppl_high = math.exp(high_loss) if not math.isnan(high_loss) else float('nan')

    return {
        'seq_len': seq_len,
        'num_batches': len(seq_losses),
        'total_tokens': len(seq_losses) * seq_len,
        'loss_mean': mean_loss,
        'loss_ci_95': [low_loss, high_loss],
        'ppl_mean': ppl_mean,
        'ppl_ci_95': [ppl_low, ppl_high],
    }


def run_long_context_experiment(config: Optional[LongContextExtrapolationConfig] = None, save_plot: bool = True):
    if config is None:
        config = LongContextExtrapolationConfig()

    device = config.device
    models: List[YaRNConfig] = [config.models] if isinstance(config.models, YaRNConfig) else list(config.models)
    seq_lengths = list(config.seq_lengths)

    print("YaRN Context Length Extrapolation Benchmark (Training-Free Extension Diagnostics)")
    print(f"Device: {device}")
    print(f"Models: {[m.model.model_name for m in models]}")
    print(f"Target Sequence Lengths: {seq_lengths}\n")

    os.makedirs(CACHE_PATH, exist_ok=True)
    cache_file = os.path.join(CACHE_PATH, config.cache_file_name)
    results: Dict[str, Dict[str, Dict]] = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}
    for m in models:
        if m.model.model_name not in results:
            results[m.model.model_name] = {}

    for yarn_cfg in models:
        model_name = yarn_cfg.model.model_name
        plot_name = yarn_cfg.model.plot_name
        scale_tag = model_name.split("_")[0]
        val_cache_path = f"data/datasets/data_{scale_tag}_val_cache.bin"

        if not os.path.exists(val_cache_path):
            print(f"[WARNING] Validation cache not found: {val_cache_path}. Skipping {model_name}.")
            continue

        print(f"\n{'#' * 80}")
        print(f"Evaluating {plot_name} ({model_name})")
        print(f"{'#' * 80}")

        for N in seq_lengths:
            if str(N) in results.get(model_name, {}) and 'ppl_mean' in results[model_name][str(N)]:
                ppl = results[model_name][str(N)]['ppl_mean']
                ci = results[model_name][str(N)]['ppl_ci_95']
                print(f"[{model_name}] N={N:5d} | Cached result found (PPL: {ppl:6.2f} [{ci[0]:5.2f}, {ci[1]:5.2f}]). Skipping.")
                continue

            print(f"\n--- Initializing fresh {model_name} instance for N={N} (zero state leakage guarantee) ---")
            t0 = time.time()
            # Fresh model instantiation per length to guarantee zero state leakage
            yarn_model = YaRNModel(yarn_cfg, device=device, dtype=torch.bfloat16)
            n_patched = yarn_model.set_context_length(N)
            s = N / yarn_model.original_max_seq_len
            print(f"Fresh model ready in {time.time() - t0:.1f}s (s={s:.1f}x | {n_patched} layers patched).")

            try:
                metrics = evaluate_long_context_metrics(
                    yarn_model,
                    val_cache_path,
                    seq_len=N,
                    total_tokens_target=config.total_tokens_target,
                    device=device,
                )
                results[model_name][str(N)] = metrics

                ppl = metrics['ppl_mean']
                ppl_low, ppl_high = metrics['ppl_ci_95']
                n_seqs = metrics['num_batches']
                n_toks = metrics['total_tokens']

                print(f"[{model_name}] N={N:5d} | {n_seqs:3d} seqs ({n_toks:7d} tok) | PPL: {ppl:7.2f} (95% CI: [{ppl_low:6.2f}, {ppl_high:6.2f}])")

                with open(cache_file, "w") as f:
                    json.dump(results, f, indent=2)


            except torch.cuda.OutOfMemoryError:
                print(f"[{model_name}] N={N:5d} | CUDA OOM encountered.")
                torch.cuda.empty_cache()
                break
            finally:
                del yarn_model
                gc.collect()
                torch.cuda.empty_cache()

    os.makedirs(CACHE_PATH, exist_ok=True)
    cache_file = os.path.join(CACHE_PATH, config.cache_file_name)
    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Comprehensive results cached to {cache_file}\n")

    # Summary table
    headers = [f"{m.model.plot_name:<24}" for m in models]
    print(f"{'Context Length (N)':<20} | " + " | ".join(headers))
    for N in seq_lengths:
        row_vals = []
        for m in models:
            m_res = results[m.model.model_name].get(str(N), {})
            if m_res and 'ppl_mean' in m_res:
                ppl = m_res['ppl_mean']
                ci = m_res['ppl_ci_95']
                row_vals.append(f"{ppl:6.2f} [{ci[0]:5.2f}, {ci[1]:5.2f}]")
            else:
                row_vals.append("N/A")
        print(f"N = {N:<16d} | " + " | ".join(f"{v:<24}" for v in row_vals))

    if save_plot:
        plot_long_context(config, results=results)

    return results


def plot_long_context(config: Optional[LongContextExtrapolationConfig] = None, results: Optional[dict] = None):
    if config is None:
        config = LongContextExtrapolationConfig()

    models: List[YaRNConfig] = [config.models] if isinstance(config.models, YaRNConfig) else list(config.models)
    label_map = {m.model.model_name: m.model.plot_name for m in models}

    cache_file = os.path.join(CACHE_PATH, config.cache_file_name)
    if results is None:
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file {cache_file} not found.")
        with open(cache_file, "r") as f:
            results = json.load(f)

    plot_dir = "data/plots/length_extrapolation"
    os.makedirs(plot_dir, exist_ok=True)

    # 1. PPL Extrapolation Plot
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    markers = ["o-", "s--", "^-.", "d:"]

    target_lengths = set(config.seq_lengths) if config and getattr(config, 'seq_lengths', None) else None
    target_models = {m.model.model_name for m in models} if models else None

    all_vals = []
    plot_items = [(m_name, m_results) for m_name, m_results in results.items() if (target_models is None or m_name in target_models)]
    for idx, (m_name, m_results) in enumerate(plot_items):
        lengths = sorted([int(k) for k in m_results.keys() if k.isdigit() and (target_lengths is None or int(k) in target_lengths)])
        if not lengths:
            continue
        vals = [m_results[str(n)]['ppl_mean'] for n in lengths]
        lows = [m_results[str(n)]['ppl_ci_95'][0] for n in lengths]
        highs = [m_results[str(n)]['ppl_ci_95'][1] for n in lengths]
        all_vals.extend(vals)

        color = palette[idx % len(palette)]
        marker = markers[idx % len(markers)]
        label = label_map.get(m_name, m_name)

        ax.plot(lengths, vals, marker, color=color, linewidth=2.0, markersize=7, label=f"{label} (YaRN)")
        yerr = [
            [v - l for v, l in zip(vals, lows)],
            [h - v for v, h in zip(vals, highs)]
        ]
        ax.errorbar(lengths, vals, yerr=yerr, fmt='none', ecolor=color, elinewidth=1.2, capsize=3.5, alpha=0.7)

    ax.set_xscale("log", base=2)
    any_lengths = sorted(list({int(k) for m_name, sub in results.items() if (target_models is None or m_name in target_models) for k in sub.keys() if k.isdigit() and (target_lengths is None or int(k) in target_lengths)}))
    if any_lengths:
        ax.set_xticks(any_lengths)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    if all_vals and max(all_vals) > 150:
        ax.set_yscale("log")
        ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    ax.set_xlabel("Context Length N (Tokens)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Validation Perplexity (95% Bootstrap CI)", fontsize=11, fontweight="medium")
    ax.set_title("Training-Free Extension Diagnostics: Context Length Extrapolation", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=10.0, loc="upper left")
    fig.tight_layout()

    plot_pdf = os.path.join(plot_dir, "yarn_length_extrapolation.pdf")
    fig.savefig(plot_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] PPL extrapolation plot saved to {plot_pdf}")


if __name__ == "__main__":
    run_long_context_experiment()
