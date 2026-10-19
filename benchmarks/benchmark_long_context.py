"""
Context length extrapolation benchmark on full PG-19 test books with canonical YaRN.
"""

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
import matplotlib.ticker as ticker

from src.yarn import YaRNModel
from training.training_config import YaRNConfig
from benchmarks.benchmarks_configs import LongContextExtrapolationConfig, CACHE_PATH


def bootstrap_ci(losses: List[float], num_resamples: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Computes mean and 95% bootstrap confidence interval over books."""
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
    """Computes cross-entropy loss in sequence chunks to prevent activation VRAM explosion."""
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


def evaluate_books_extrapolation(
    model: YaRNModel,
    data_path: str,
    books: List[dict],
    seq_len: int,
    device: str = 'cuda',
    max_books: Optional[int] = None,
    eval_target_tokens: Optional[int] = None,
) -> Dict:
    """Evaluates non-overlapping windows of length seq_len spanning eval_target_tokens from each book."""
    bin_data = np.memmap(data_path, dtype=np.uint16, mode='r')
    target_books = books if max_books is None else books[:max_books]

    seq_losses: List[float] = []

    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for b_info in target_books:
            start = b_info["start"]
            target_span = eval_target_tokens if (eval_target_tokens is not None and eval_target_tokens >= seq_len) else seq_len
            n_chunks = max(1, target_span // seq_len)
            book_chunk_losses: List[float] = []

            for c_idx in range(n_chunks):
                c_start = start + c_idx * seq_len
                tok_slice = bin_data[c_start : c_start + seq_len + 1].astype(np.int64)
                toks = torch.from_numpy(tok_slice)
                x = toks[:seq_len].unsqueeze(0).to(device)
                y = toks[1 : seq_len + 1].unsqueeze(0).to(device)

                logits, _ = model(x)
                chunk_loss = compute_chunked_loss(logits, y, chunk_size=2048)

                if not (math.isnan(chunk_loss) or math.isinf(chunk_loss)):
                    book_chunk_losses.append(chunk_loss)

                del logits, x, y, toks

            if book_chunk_losses:
                seq_losses.append(sum(book_chunk_losses) / len(book_chunk_losses))

            if seq_len >= 8192:
                torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    gc.collect()

    mean_loss, low_loss, high_loss = bootstrap_ci(seq_losses)
    ppl_mean = math.exp(mean_loss) if not math.isnan(mean_loss) else float('nan')
    ppl_low = math.exp(low_loss) if not math.isnan(low_loss) else float('nan')
    ppl_high = math.exp(high_loss) if not math.isnan(high_loss) else float('nan')

    actual_span = (eval_target_tokens if (eval_target_tokens is not None and eval_target_tokens >= seq_len) else seq_len)
    return {
        'seq_len': seq_len,
        'num_books_evaluated': len(seq_losses),
        'total_tokens': len(seq_losses) * actual_span,
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

    print("=" * 80)
    print("YaRN Context Length Extrapolation Benchmark (PG-19 Test Split)")
    print(f"Device: {device}")
    print(f"Models: {[m.model.model_name for m in models]}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Data source: {config.data_path}")
    print(f"RoPE Setting: {'Fixed s=16.0 (Trained CPT Tables)' if config.use_fixed_yarn else 'Dynamic s=N/1024'}")
    print("=" * 80 + "\n")

    if not os.path.exists(config.data_path) or not os.path.exists(config.index_path):
        raise FileNotFoundError(f"PG-19 test dataset not found: {config.data_path} or {config.index_path}")

    with open(config.index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    all_books = index_data.get("books", [])
    total_books_in_split = len(all_books)

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

        print(f"\nEvaluating {plot_name} ({model_name})...")

        for N in seq_lengths:
            req_len = max(N, getattr(config, 'eval_target_tokens', N) or N)
            eligible_books = [b for b in all_books if b["length"] >= req_len + 1]
            n_books = len(eligible_books)

            if n_books == 0:
                print(f"[{model_name}] N={N:5d} | No eligible books with length >= {req_len+1}. Skipping.")
                continue

            if not config.force_rerun and str(N) in results.get(model_name, {}) and 'ppl_mean' in results[model_name][str(N)]:
                ppl = results[model_name][str(N)]['ppl_mean']
                ci = results[model_name][str(N)]['ppl_ci_95']
                n_eval = results[model_name][str(N)].get('num_books_evaluated', n_books)
                print(f"[{model_name}] N={N:5d} | Cached result: PPL={ppl:6.2f} [{ci[0]:5.2f}, {ci[1]:5.2f}] ({n_eval}/{total_books_in_split} books)")
                continue

            print(f"[{model_name}] N={N:5d} | Initializing model... ({n_books}/{total_books_in_split} books eligible)")
            t0 = time.time()
            yarn_model = YaRNModel(yarn_cfg, device=device, dtype=torch.bfloat16)

            # Primary setting: s=16 fixed for N <= 16k; dynamic scale s=N/1024 for N > 16k
            if config.use_fixed_yarn:
                scale_val = max(16.0, float(N) / float(yarn_model.original_max_seq_len))
            else:
                scale_val = None
            n_patched = yarn_model.set_context_length(N, scale=scale_val)
            s_eff = scale_val if scale_val is not None else (N / yarn_model.original_max_seq_len)
            print(f"[{model_name}] Patched {n_patched} layers (RoPE scale s={s_eff:.1f}x, init: {time.time() - t0:.1f}s)")

            try:
                metrics = evaluate_books_extrapolation(
                    yarn_model,
                    data_path=config.data_path,
                    books=eligible_books,
                    seq_len=N,
                    device=device,
                    eval_target_tokens=getattr(config, 'eval_target_tokens', None),
                )
                results[model_name][str(N)] = metrics

                ppl = metrics['ppl_mean']
                ppl_low, ppl_high = metrics['ppl_ci_95']
                n_bks = metrics['num_books_evaluated']
                n_toks = metrics['total_tokens']

                print(f"[{model_name}] N={N:5d} | {n_bks:3d} books ({n_toks:7d} tok) | PPL: {ppl:7.2f} (95% CI: [{ppl_low:6.2f}, {ppl_high:6.2f}])")

                with open(cache_file, "w") as f:
                    json.dump(results, f, indent=2)

            except torch.cuda.OutOfMemoryError:
                print(f"[{model_name}] N={N:5d} | CUDA OutOfMemory encountered. Gracefully stopping sweep for this model.")
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
    print(f"\nAll completed results cached to {cache_file}\n")

    headers = [f"{m.model.plot_name:<24}" for m in models]
    print(f"{'Context Length (N)':<20} | {'Books':<7} | " + " | ".join(headers))
    for N in seq_lengths:
        req_len = max(N, getattr(config, 'eval_target_tokens', N) or N)
        eligible_books = [b for b in all_books if b["length"] >= req_len + 1]
        b_count = f"{len(eligible_books)}/{total_books_in_split}"
        row_vals = []
        for m in models:
            m_res = results[m.model.model_name].get(str(N), {})
            if m_res and 'ppl_mean' in m_res:
                ppl = m_res['ppl_mean']
                ci = m_res['ppl_ci_95']
                row_vals.append(f"{ppl:6.2f} [{ci[0]:5.2f}, {ci[1]:5.2f}]")
            else:
                row_vals.append("N/A")
        print(f"N = {N:<16d} | {b_count:<7} | " + " | ".join(f"{v:<24}" for v in row_vals))

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

        ax.plot(lengths, vals, marker, color=color, linewidth=2.0, markersize=7, label=label)
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

    ax.set_xlabel("Context Length N (Tokens)", fontsize=11)
    ax.set_ylabel("Perplexity on PG-19 Test", fontsize=11)
    ax.set_title("Context Length Extrapolation (PG-19)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=10.0, loc="upper left")
    fig.tight_layout()

    plot_pdf = os.path.join(plot_dir, "yarn_length_extrapolation.pdf")
    fig.savefig(plot_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_pdf}")


if __name__ == "__main__":
    run_long_context_experiment()
