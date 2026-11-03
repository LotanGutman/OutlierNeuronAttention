"""Normalized attention entropy and DuoAttention head classification benchmark."""

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
) -> Tuple[float, float, torch.Tensor]:
    """Computes Shannon entropy H, normalized entropy H / ln(i+1), and head retrieval score."""
    # Q, K: (B=1, H, N, d)
    B, num_heads, N, d = Q.shape
    total_norm_entropy = 0.0
    total_raw_entropy = 0.0
    valid_count = 0

    head_retrieval_sum = torch.zeros(num_heads, dtype=torch.float64, device=Q.device)
    retrieval_query_count = 0

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
        log_probs = F.log_softmax(scores, dim=-1)  # (1, H, 1, max_j)
        probs = log_probs.exp()
        
        # Entropy per head: (1, H)
        h_per_head = -(probs * log_probs).sum(dim=-1).squeeze(-1)

        h_mean_query = h_per_head.mean().item()  # Average across heads
        max_entropy = math.log(max_j)
        norm_h = h_mean_query / max_entropy if max_entropy > 0 else 0.0

        total_norm_entropy += norm_h
        total_raw_entropy += h_mean_query
        valid_count += 1

        # Retrieval score: mass on distant keys [4, q_idx - 127) excluding sinks [0, 4) and 128 local tokens
        if q_idx > 131:
            distant_mass = probs[0, :, 0, 4 : (q_idx - 127)].sum(dim=-1).to(torch.float64)  # (H,)
            head_retrieval_sum += distant_mass
            retrieval_query_count += 1

    mean_raw = total_raw_entropy / valid_count if valid_count > 0 else 0.0
    mean_norm = total_norm_entropy / valid_count if valid_count > 0 else 0.0
    if retrieval_query_count > 0:
        head_ret_scores = (head_retrieval_sum / retrieval_query_count).cpu()
    else:
        head_ret_scores = torch.zeros(num_heads, dtype=torch.float64)

    return mean_raw, mean_norm, head_ret_scores


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
    """Evaluates normalized attention entropy and DuoAttention head retrieval scores across layers."""
    bin_data = np.memmap(data_path, dtype=np.uint16, mode='r')
    target_books = books if max_books is None else books[:max_books]

    # Sample query indices evenly spaced along sequence length
    query_indices = np.linspace(64, seq_len - 1, num_sample_queries, dtype=int).tolist()
    query_indices = sorted(list(set(query_indices)))

    model = yarn_model.model
    layers = getattr(model, "layers", [])
    num_layers = len(layers)

    book_layer_norm_entropies: List[List[float]] = [[] for _ in range(num_layers)]
    book_layer_raw_entropies: List[List[float]] = [[] for _ in range(num_layers)]
    book_layer_head_rets: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]

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
                
                raw_h, norm_h, head_ret = compute_normalized_entropy_chunked(Q, K, scale, query_indices)
                book_layer_raw_entropies[l_idx].append(raw_h)
                book_layer_norm_entropies[l_idx].append(norm_h)
                book_layer_head_rets[l_idx].append(head_ret)

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

    # DuoAttention Head Classification across all layers (num_layers * num_heads)
    all_head_ret_scores: List[float] = []
    for l_idx in range(num_layers):
        if book_layer_head_rets[l_idx]:
            layer_mean_rets = torch.stack(book_layer_head_rets[l_idx], dim=0).mean(dim=0).tolist()
            all_head_ret_scores.extend(layer_mean_rets)

    duo_threshold = 1e-5
    streaming_head_count = sum(1 for s in all_head_ret_scores if s < duo_threshold)
    retrieval_head_count = sum(1 for s in all_head_ret_scores if s >= duo_threshold)
    total_heads = len(all_head_ret_scores)
    streaming_head_ratio = streaming_head_count / total_heads if total_heads > 0 else 0.0
    retrieval_head_ratio = retrieval_head_count / total_heads if total_heads > 0 else 0.0

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
        "head_retrieval_scores": all_head_ret_scores,
        "streaming_head_count": streaming_head_count,
        "retrieval_head_count": retrieval_head_count,
        "streaming_head_ratio": streaming_head_ratio,
        "retrieval_head_ratio": retrieval_head_ratio,
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
    print("Normalized Attention Entropy & Head Retrieval Benchmark")
    print(f"Device: {device}")
    print(f"Models: {[m.model.model_name for m in models]}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Sample queries per sequence: {config.num_sample_queries}")
    print(f"Max books per length: {config.max_books}")
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

            if (not config.force_rerun 
                and str(N) in results[model_name] 
                and "normalized_entropy_mean" in results[model_name][str(N)]
                and "head_retrieval_scores" in results[model_name][str(N)]):
                cached_val = results[model_name][str(N)]["normalized_entropy_mean"]
                cached_stream = results[model_name][str(N)].get("streaming_head_ratio", 0.0) * 100.0
                print(f"[{model_name}] N={N:5d} | Cached H/ln(N): {cached_val:.4f} | Streaming Heads: {cached_stream:.1f}% ({len(eval_books)} books)")
                continue

            print(f"[{model_name}] N={N:5d} | Initializing model and patching RoPE...", end="", flush=True)
            yarn_model = YaRNModel(yarn_cfg, device=device, dtype=torch.bfloat16)
            scale_val = max(1.0, float(N) / float(yarn_model.original_max_seq_len))
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

            stream_pct = res['streaming_head_ratio'] * 100.0
            print(f" Done in {time.time() - t_eval:.1f}s | H/ln(N): {res['normalized_entropy_mean']:.4f} | Streaming Heads: {stream_pct:.1f}% ({res['streaming_head_count']}/{len(res['head_retrieval_scores'])})")

            del yarn_model
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\nResults saved to {cache_file}")


def plot_entropy_yarn(config: Optional[EntropyYaRNConfig] = None):
    """Plot normalized attention entropy vs context length and DuoAttention head distribution."""
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

    out_dir = "data/plots/length_extrapolation_entropy_scaling"
    os.makedirs(out_dir, exist_ok=True)

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

    colors = {
        "hofa": "#1f77b4",  # Blue
        "mha": "#d62728",   # Red
    }
    markers = {
        "hofa": "o",
        "mha": "s",
    }

    # =========================================================================
    # PLOT 1: Attention Entropy Invariance vs Context Length (arXiv:2610.10114)
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(8, 5.5))

    all_plotted_lens = set()
    for m_name in available_models:
        m_data = results[m_name]
        sorted_lens = sorted([int(k) for k in m_data.keys() if "normalized_entropy_mean" in m_data[k]])
        if not sorted_lens:
            continue
        all_plotted_lens.update(sorted_lens)
        norm_entropies = [m_data[str(k)]["normalized_entropy_mean"] for k in sorted_lens]

        is_mha = "MHA" in m_name
        label = "350M MHA (16k CPT)" if is_mha else "350M HOFA (16k CPT)"
        color = colors["mha"] if is_mha else colors["hofa"]
        marker = markers["mha"] if is_mha else markers["hofa"]
        linestyle = "--" if is_mha else "-"

        ax1.plot(
            sorted_lens,
            norm_entropies,
            marker=marker,
            markersize=7,
            linewidth=2.2,
            label=label,
            color=color,
            linestyle=linestyle
        )

    if all_plotted_lens:
        ax1.set_xscale('log', base=2)
        tick_lens = sorted(list(all_plotted_lens))
        ax1.set_xticks(tick_lens)
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        ax1.set_xlabel("Context Length $N$ (Tokens)")
        ax1.set_ylabel(r"Normalized Attention Entropy $\bar{H} = H / \ln(N)$")
        ax1.set_title("Attention Entropy Invariance vs. Context Length", fontweight='bold')
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="best", framealpha=0.9)
        ax1.set_ylim([0.0, 1.05])
        plt.tight_layout()

        pdf_entropy_path = os.path.join(out_dir, "attention_entropy_invariance.pdf")
        fig1.savefig(pdf_entropy_path, dpi=300)
        plt.close(fig1)
        print(f"Saved attention entropy invariance plot to: {pdf_entropy_path}")

    # =========================================================================
    # PLOT 2: Head Classification & Retrieval Distribution
    # =========================================================================
    target_N_candidates = []
    for m_name in available_models:
        for k in results[m_name].keys():
            if "head_retrieval_scores" in results[m_name][k]:
                target_N_candidates.append(int(k))

    if not target_N_candidates:
        print("No head retrieval score data found in cache. Skipping DuoAttention plot.")
        return

    eval_N = max(target_N_candidates)
    str_eval_N = str(eval_N)

    fig2, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Stacked Bar Chart of Head Breakdown
    model_labels = []
    streaming_pcts = []
    retrieval_pcts = []
    streaming_raw = []
    retrieval_raw = []

    for m_name in available_models:
        if str_eval_N not in results[m_name] or "head_retrieval_scores" not in results[m_name][str_eval_N]:
            continue
        entry = results[m_name][str_eval_N]
        is_mha = "MHA" in m_name
        label = "350M MHA" if is_mha else "350M HOFA"
        model_labels.append(label)
        s_count = entry.get("streaming_head_count", 0)
        r_count = entry.get("retrieval_head_count", 0)
        tot = max(1, s_count + r_count)
        s_ratio = s_count / tot * 100.0
        r_ratio = r_count / tot * 100.0
        streaming_pcts.append(s_ratio)
        retrieval_pcts.append(r_ratio)
        streaming_raw.append(s_count)
        retrieval_raw.append(r_count)

    bar_width = 0.45
    x_indices = np.arange(len(model_labels))

    col_stream = "#2ca25f"
    col_ret = "#2b5c8f"

    p1 = ax_bar.bar(x_indices, streaming_pcts, bar_width, label=r"Streaming Heads ($\text{RetScore} < 10^{-5}$)", color=col_stream, edgecolor="black", linewidth=0.8)
    p2 = ax_bar.bar(x_indices, retrieval_pcts, bar_width, bottom=streaming_pcts, label=r"Retrieval Heads ($\text{RetScore} \geq 10^{-5}$)", color=col_ret, edgecolor="black", linewidth=0.8)

    for i in range(len(model_labels)):
        if streaming_pcts[i] > 8.0:
            ax_bar.text(
                x_indices[i],
                streaming_pcts[i] / 2.0,
                f"Streaming\n{streaming_raw[i]} heads\n({streaming_pcts[i]:.1f}%)",
                ha="center", va="center", color="white", fontweight="bold", fontsize=10
            )
        if retrieval_pcts[i] > 8.0:
            ax_bar.text(
                x_indices[i],
                streaming_pcts[i] + (retrieval_pcts[i] / 2.0),
                f"Retrieval\n{retrieval_raw[i]} heads\n({retrieval_pcts[i]:.1f}%)",
                ha="center", va="center", color="white", fontweight="bold", fontsize=10
            )

    ax_bar.set_xticks(x_indices)
    ax_bar.set_xticklabels(model_labels, fontweight="bold", fontsize=11)
    ax_bar.set_ylabel("Attention Heads Share (%)")
    ax_bar.set_ylim([0, 105])
    ax_bar.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax_bar.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_bar.set_title(f"(a) DuoAttention Head Ratio at $N={eval_N // 1024}$k Context", fontweight="bold")
    ax_bar.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Panel B: Sorted Head Retrieval Scores Curve
    for m_name in available_models:
        if str_eval_N not in results[m_name] or "head_retrieval_scores" not in results[m_name][str_eval_N]:
            continue
        entry = results[m_name][str_eval_N]
        raw_scores = sorted(entry["head_retrieval_scores"])
        is_mha = "MHA" in m_name
        label = "350M MHA" if is_mha else "350M HOFA"
        color = colors["mha"] if is_mha else colors["hofa"]
        marker = markers["mha"] if is_mha else markers["hofa"]
        linestyle = "--" if is_mha else "-"
        
        ax_curve.plot(
            range(len(raw_scores)),
            raw_scores,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker=marker,
            markersize=3.5,
            markevery=max(1, len(raw_scores) // 20)
        )

    ax_curve.axhline(1e-5, color="darkred", linestyle=":", linewidth=2.0, label=r"DuoAttention Threshold ($10^{-5}$)")
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("Sorted Attention Head Index (0 to 383)")
    ax_curve.set_ylabel("Attention Retrieval Score")
    ax_curve.set_title(f"(b) Sorted Head Retrieval Scores ($N={eval_N // 1024}$k)", fontweight="bold")
    ax_curve.grid(True, linestyle=":", alpha=0.5, which="both")
    ax_curve.legend(loc="upper left", framealpha=0.9, fontsize=9)

    fig2.suptitle("Attention Head Retrieval Distribution (DuoAttention)", fontweight="bold")
    plt.tight_layout()

    pdf_head_path = os.path.join(out_dir, "duo_attention_head_distribution.pdf")
    fig2.savefig(pdf_head_path, dpi=300)
    plt.close(fig2)
    print(f"Saved DuoAttention head distribution plot to: {pdf_head_path}")


if __name__ == "__main__":
    cfg = EntropyYaRNConfig()
    if "--all-lengths" in sys.argv:
        cfg.seq_lengths = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
    if "--force" in sys.argv:
        cfg.force_rerun = True

    if "--plot" in sys.argv:
        plot_entropy_yarn(cfg)
    else:
        run_entropy_yarn_experiment(cfg)
        plot_entropy_yarn(cfg)
