"""Normalized attention entropy benchmark."""

import os
import sys
import gc
import json
import math
import time
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.yarn import YaRNModel, YaRNMode
from training.training_config import YaRNConfig
from benchmarks.benchmarks_configs import EntropyYaRNConfig, CACHE_PATH
from src.modules.modules import apply_rotary_pos_emb


def compute_normalized_entropy_chunked(
    Q: torch.Tensor,
    K: torch.Tensor,
    scale: float,
    query_indices: List[int],
) -> Tuple[float, float]:
    """Computes Shannon entropy H and normalized entropy H / ln(i+1) for sample queries without OOM."""
    # Q, K: (B=1, H, N, d)
    B, num_heads, N, d = Q.shape
    total_norm_entropy = 0.0
    total_raw_entropy = 0.0
    valid_count = 0

    for q_idx in query_indices:
        if q_idx <= 1:
            continue
        max_j = q_idx + 1  # Causal masking: attend to tokens 0 .. q_idx
        q_vec = Q[:, :, q_idx:q_idx+1, :]  # (1, H, 1, d)
        k_vecs = K[:, :, :max_j, :]        # (1, H, max_j, d)

        # Dot product: (1, H, 1, max_j)
        scores = (q_vec @ k_vecs.transpose(-1, -2)) * scale
        
        # Numerically stable entropy:
        # A_ij = softmax(scores)
        # H_i = - sum A_ij * ln(A_ij)
        # Using log_softmax:
        log_probs = F.log_softmax(scores, dim=-1)  # (1, H, 1, max_j)
        probs = log_probs.exp()
        # Entropy per head: (1, H, 1)
        h_per_head = -(probs * log_probs).sum(dim=-1).squeeze(-1)  # (1, H)

        h_mean_query = h_per_head.mean().item()  # Average across heads
        max_entropy = math.log(max_j)
        norm_h = h_mean_query / max_entropy if max_entropy > 0 else 0.0

        total_norm_entropy += norm_h
        total_raw_entropy += h_mean_query
        valid_count += 1

    if valid_count == 0:
        return 0.0, 0.0
    return total_raw_entropy / valid_count, total_norm_entropy / valid_count


def extract_layer_q_k(layer: nn.Module, x: torch.Tensor, N: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Extracts position-rotated Q and K for attention entropy computation."""
    attn = layer.attn
    num_heads = attn.num_heads
    d_head = attn.d_head
    scale_factor = d_head ** 0.25

    # Autocast linear projections
    Q_full = (attn.W_q(x) / scale_factor).view(1, N, num_heads, d_head).transpose(1, 2)
    K_full = (attn.W_k(x) / scale_factor).view(1, N, num_heads, d_head).transpose(1, 2)

    is_pure_mha = (not hasattr(attn, 'r')) or (attn.r == d_head) or (getattr(attn, 'forced_mha_heads', 0) > 0)
    
    if is_pure_mha:
        # MHA: All d_head dimensions have RoPE
        if getattr(attn, 'rotary_emb', None) is not None:
            cos, sin = attn.rotary_emb(N)
            Q, K = apply_rotary_pos_emb(Q_full, K_full, cos, sin)
        else:
            Q, K = Q_full, K_full
        scale = getattr(attn, 'attention_scale', 1.0)
    else:
        # HOFA: Outlier branch (top r dimensions)
        r = attn.r
        Q_O = Q_full[..., :r]
        K_O = K_full[..., :r]
        if getattr(attn, 'rotary_emb', None) is not None:
            cos, sin = attn.rotary_emb(N)
            Q_O, K_O = apply_rotary_pos_emb(Q_O, K_O, cos, sin)
        Q, K = Q_O, K_O
        sm_scale = ((d_head / r) ** 0.5) * getattr(attn, 'attention_scale', 1.0)
        scale = sm_scale

    return Q, K, scale


def evaluate_entropy_for_length(
    yarn_model: YaRNModel,
    data_path: str,
    books: List[dict],
    seq_len: int,
    num_sample_queries: int = 64,
    device: str = "cuda",
    max_books: Optional[int] = None,
) -> Dict:
    """Evaluates normalized attention entropy for a given sequence length across layers."""
    bin_data = np.memmap(data_path, dtype=np.uint16, mode='r')
    target_books = books if max_books is None else books[:max_books]

    # Sample query indices log-spaced / evenly spaced along sequence length
    # e.g. from 64 to seq_len - 1
    query_indices = np.linspace(64, seq_len - 1, num_sample_queries, dtype=int).tolist()
    query_indices = sorted(list(set(query_indices)))

    model = yarn_model.model
    layers = getattr(model, "layers", [])
    num_layers = len(layers)

    book_layer_norm_entropies: List[List[float]] = [[] for _ in range(num_layers)]
    book_layer_raw_entropies: List[List[float]] = [[] for _ in range(num_layers)]

    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for b_info in target_books:
            start = b_info["start"]
            tok_slice = bin_data[start : start + seq_len].astype(np.int64)
            toks = torch.from_numpy(tok_slice)
            x = toks.unsqueeze(0).to(device)

            # Pass token embeddings
            emb_layer = getattr(model, 'token_emb', getattr(model, 'tok_embeddings', None))
            if emb_layer is None:
                raise AttributeError(f"Model {type(model)} has neither 'token_emb' nor 'tok_embeddings'")
            h = emb_layer(x)
            if hasattr(model, 'dropout'):
                h = model.dropout(h)

            for l_idx, layer in enumerate(layers):
                # Compute Q and K for this layer
                if hasattr(layer, 'ln_1'):
                    h_norm = layer.ln_1(h)
                elif hasattr(layer, 'input_layernorm'):
                    h_norm = layer.input_layernorm(h)
                elif hasattr(layer, 'norm1'):
                    h_norm = layer.norm1(h)
                else:
                    h_norm = h
                Q, K, scale = extract_layer_q_k(layer, h_norm, seq_len)
                
                raw_h, norm_h = compute_normalized_entropy_chunked(Q, K, scale, query_indices)
                book_layer_raw_entropies[l_idx].append(raw_h)
                book_layer_norm_entropies[l_idx].append(norm_h)

                # Advance representation through layer for accurate deeper activations
                h = layer(h)
                del Q, K

            del h, x, toks
            if seq_len >= 8192:
                torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    gc.collect()

    mean_layer_norm_entropies = [float(np.mean(vals)) if vals else 0.0 for vals in book_layer_norm_entropies]
    mean_layer_raw_entropies = [float(np.mean(vals)) if vals else 0.0 for vals in book_layer_raw_entropies]
    model_mean_norm_entropy = float(np.mean(mean_layer_norm_entropies))
    model_mean_raw_entropy = float(np.mean(mean_layer_raw_entropies))

    # Layer group breakdowns
    n_third = num_layers // 3
    early_norm_entropy = float(np.mean(mean_layer_norm_entropies[:n_third]))
    mid_norm_entropy = float(np.mean(mean_layer_norm_entropies[n_third:2*n_third]))
    late_norm_entropy = float(np.mean(mean_layer_norm_entropies[2*n_third:]))

    return {
        "seq_len": seq_len,
        "normalized_entropy_mean": model_mean_norm_entropy,
        "raw_entropy_mean": model_mean_raw_entropy,
        "early_norm_entropy": early_norm_entropy,
        "mid_norm_entropy": mid_norm_entropy,
        "late_norm_entropy": late_norm_entropy,
        "layer_norm_entropies": mean_layer_norm_entropies,
        "num_books": len(target_books),
        "num_queries": len(query_indices)
    }


def run_entropy_yarn_experiment(config: Optional[EntropyYaRNConfig] = None):
    if config is None:
        config = EntropyYaRNConfig()

    device = config.device
    models: List[YaRNConfig] = [config.models] if isinstance(config.models, YaRNConfig) else list(config.models)
    seq_lengths = list(config.seq_lengths)

    print("=" * 80)
    print("Normalized Attention Entropy Invariance Benchmark (arXiv:2610.10114)")
    print(f"Device: {device}")
    print(f"Models: {[m.model.model_name for m in models]}")
    print(f"Sequence lengths (1k -> 64k): {seq_lengths}")
    print(f"Sample queries per sequence: {config.num_sample_queries}")
    print(f"Data source: {config.data_path}")
    print("=" * 80 + "\n")

    if not os.path.exists(config.data_path) or not os.path.exists(config.index_path):
        raise FileNotFoundError(f"PG-19 test dataset not found: {config.data_path} or {config.index_path}")

    with open(config.index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    all_books = index_data.get("books", [])

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

    for yarn_cfg in models:
        model_name = yarn_cfg.model.model_name
        plot_name = yarn_cfg.model.plot_name
        print(f"\nEvaluating Entropy for {plot_name} ({model_name})...")

        for N in seq_lengths:
            eligible_books = [b for b in all_books if b["length"] >= N + 1]
            n_books = len(eligible_books)
            if n_books == 0:
                print(f"[{model_name}] N={N:5d} | No eligible books with length >= {N+1}. Skipping.")
                continue

            eval_books = eligible_books[:config.max_books] if config.max_books else eligible_books

            if not config.force_rerun and str(N) in results[model_name] and "normalized_entropy_mean" in results[model_name][str(N)]:
                cached_val = results[model_name][str(N)]["normalized_entropy_mean"]
                print(f"[{model_name}] N={N:5d} | Cached H/ln(N): {cached_val:.4f} ({len(eval_books)} books)")
                continue

            print(f"[{model_name}] N={N:5d} | Initializing model and patching RoPE...", end="", flush=True)
            t0 = time.time()
            yarn_model = YaRNModel(yarn_cfg, device=device, dtype=torch.bfloat16)
            scale_val = max(16.0, float(N) / float(yarn_model.original_max_seq_len))
            ymode = getattr(yarn_cfg, 'yarn_mode', None) or YaRNMode.default()
            yarn_model.set_context_length(N, scale=scale_val, yarn_mode=ymode)

            t_eval = time.time()
            res = evaluate_entropy_for_length(
                yarn_model=yarn_model,
                data_path=config.data_path,
                books=eval_books,
                seq_len=N,
                num_sample_queries=config.num_sample_queries,
                device=device,
                max_books=len(eval_books),
            )
            results[model_name][str(N)] = res

            with open(cache_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f" Done in {time.time() - t_eval:.1f}s | H/ln(N): {res['normalized_entropy_mean']:.4f}")

            del yarn_model
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\nResults saved to {cache_file}")


def plot_entropy_yarn(config: Optional[EntropyYaRNConfig] = None):
    """Plot normalized attention entropy vs sequence length."""
    if config is None:
        config = EntropyYaRNConfig()

    cache_file = os.path.join(CACHE_PATH, config.cache_file_name)
    if not os.path.exists(cache_file):
        print(f"Cache file {cache_file} not found. Run benchmark first.")
        return

    with open(cache_file, "r") as f:
        results = json.load(f)

    available_models = list(results.keys())
    if not available_models:
        print("No models found in cache.")
        return

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })

    fig, ax = plt.subplots(figsize=(8, 5.5))

    colors = {
        "hofa": "#1f77b4",  # Blue
        "mha": "#d62728",   # Red
    }
    markers = {
        "hofa": "o",
        "mha": "s",
    }

    for m_name in available_models:
        m_data = results[m_name]
        sorted_lens = sorted([int(k) for k in m_data.keys()])
        norm_entropies = [m_data[str(k)]["normalized_entropy_mean"] for k in sorted_lens]

        is_mha = "MHA" in m_name
        label = "350M MHA (16k CPT)" if is_mha else "350M HOFA (16k CPT)"
        color = colors["mha"] if is_mha else colors["hofa"]
        marker = markers["mha"] if is_mha else markers["hofa"]
        linestyle = "--" if is_mha else "-"

        ax.plot(
            sorted_lens,
            norm_entropies,
            marker=marker,
            markersize=7,
            linewidth=2.2,
            label=label,
            color=color,
            linestyle=linestyle
        )

    ax.set_xscale('log', base=2)
    ax.set_xticks(config.seq_lengths)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xlabel("Context Length $N$ (Tokens)")
    ax.set_ylabel(r"Normalized Attention Entropy $\bar{H} = H / \ln(N)$")
    ax.set_title("Attention Entropy Invariance vs. Context Length\n(arXiv:2610.10114 Replication)", fontweight='bold')
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", framealpha=0.9)
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    out_dir = "data/plots/length_extrapolation_entropy_scaling"
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "attention_entropy_invariance.pdf")
    plt.savefig(pdf_path, dpi=300)
    plt.close()

    print(f"Saved attention entropy plot to: {pdf_path}")


if __name__ == "__main__":
    cfg = EntropyYaRNConfig()
    if "--plot" in sys.argv:
        plot_entropy_yarn(cfg)
    else:
        run_entropy_yarn_experiment(cfg)
        plot_entropy_yarn(cfg)

