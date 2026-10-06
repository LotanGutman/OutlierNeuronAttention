import os
import gc
import json
import math
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.config import ModelConfig
from src.HybridOutlierFactorizedAttentionTrain import SubwordLM
from training.training_config import (
    make_70M_HOFA, make_70M_MHA,
    make_125M_HOFA, make_125M_MHA,
    make_350M_HOFA, make_350M_MHA
)
from training.data_utils import FastTokenLoader
from benchmarks.benchmarks_configs import LongContextExtrapolationConfig, CACHE_PATH

def apply_position_interpolation(model, seq_len: int, train_seq_len: int = 1024):
    """
    Applies linear Position Interpolation (PI) to all RoPE embeddings in the model.
    m' = m * (1024 / N) => scale = N / 1024.
    Leaves base frequency theta_base = 10000.0 untouched.
    """
    scale = max(1.0, float(seq_len) / float(train_seq_len))
    patched_count = 0
    for layer in model.layers:
        attn = layer.attn
        if hasattr(attn, 'rotary_emb') and attn.rotary_emb is not None:
            patched_count += 1
            rot = attn.rotary_emb
            dim = rot.dim
            device = rot.inv_freq.device
            base = 10000.0
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            rot.inv_freq = inv_freq
            
            cache_len = max(rot.max_seq_len_cached, seq_len)
            t = torch.arange(cache_len, dtype=torch.float32, device=device) / scale
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            rot.register_buffer("cos_cached", emb.cos(), persistent=False)
            rot.register_buffer("sin_cached", emb.sin(), persistent=False)
            rot.max_seq_len_cached = cache_len

    assert patched_count == len(model.layers), (
        f"Mismatch in patched layers: expected {len(model.layers)}, but patched {patched_count}."
    )
    return patched_count

def evaluate_perplexity(model, val_cache_path, seq_len, total_tokens_target=500000, device='cuda'):
    """
    Evaluates cross-entropy loss and perplexity on sequential validation chunks.
    """
    batch_size = 1
    num_batches = max(1, total_tokens_target // seq_len)
    
    loader = FastTokenLoader(val_cache_path, batch_size, seq_len, start_idx=0)
    
    total_loss = 0.0
    total_tokens = 0
    
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(num_batches):
            x, y, _ = loader.get_batch()
            x = x.to(device)
            y = y.to(device)
            
            logits, loss = model(x, targets=y)
            if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item() * seq_len
                total_tokens += seq_len
            del logits, loss, x, y
            
    torch.cuda.empty_cache()
    if total_tokens == 0:
        return float('nan'), 0, 0
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss), num_batches, total_tokens

def _get_configs(scale: str):
    if scale == "70M":
        return make_70M_HOFA(), make_70M_MHA()
    elif scale == "125M":
        return make_125M_HOFA(), make_125M_MHA()
    elif scale == "350M":
        return make_350M_HOFA(), make_350M_MHA()
    else:
        raise ValueError(f"Unsupported scale: {scale}")

def run_long_context_experiment(config: LongContextExtrapolationConfig = None, save_plot: bool = True):
    if config is None:
        config = LongContextExtrapolationConfig()
        
    device = config.device
    scale = config.scale
    val_cache_path = f"data/datasets/data_{scale}_val_cache.bin"
    
    if not os.path.exists(val_cache_path):
        raise FileNotFoundError(f"Validation cache not found: {val_cache_path}")
        
    seq_lengths = list(config.seq_lengths)
    
    print("=" * 75)
    print(f"{scale} Position Interpolation Extrapolation Benchmark (Protocol A)")
    print("=" * 75)
    print(f"Device: {device}")
    print(f"Validation Cache: {val_cache_path}")
    print(f"Target Sequence Lengths: {seq_lengths}\n")
    
    results = {
        'HOFA': {},
        'MHA': {}
    }
    
    hofa_exp_cfg, mha_exp_cfg = _get_configs(scale)
    
    # 1. Evaluate HOFA with Position Interpolation
    hofa_ckpt = f"data/training/{hofa_exp_cfg.model_name}/checkpoint_best_val.pt"
    print(f"--- Loading {scale} HOFA from {hofa_ckpt} ---")
    t0 = time.time()
    model_hofa = SubwordLM(50257, hofa_exp_cfg.model_config).to(device=device, dtype=torch.bfloat16)
    
    ckpt = torch.load(hofa_ckpt, map_location='cpu', weights_only=False)
    model_hofa.load_state_dict(ckpt['model_state_dict'])
    del ckpt
    gc.collect()
    print(f"{scale} HOFA loaded in {time.time() - t0:.1f}s.")
    
    for N in seq_lengths:
        try:
            n_patched = apply_position_interpolation(model_hofa, seq_len=N, train_seq_len=1024)
            ppl, n_seqs, n_toks = evaluate_perplexity(model_hofa, val_cache_path, seq_len=N, 
                                                      total_tokens_target=config.total_tokens_target, device=device)
            results['HOFA'][N] = ppl
            print(f"[HOFA] N={N:5d} (s={N/1024:.1f}x | {n_patched}/{len(model_hofa.layers)} layers patched) | {n_seqs:3d} docs ({n_toks:7d} tok) | PPL: {ppl:7.2f}")
        except torch.cuda.OutOfMemoryError:
            print(f"[HOFA] N={N:5d} | OOM during evaluation.")
            torch.cuda.empty_cache()
            break
            
    del model_hofa
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Evaluate MHA Baseline with Position Interpolation
    mha_ckpt = f"data/training/{mha_exp_cfg.model_name}/checkpoint_best_val.pt"
    print(f"\n--- Loading {scale} MHA from {mha_ckpt} ---")
    t0 = time.time()
    model_mha = SubwordLM(50257, mha_exp_cfg.model_config).to(device=device, dtype=torch.bfloat16)
    
    ckpt = torch.load(mha_ckpt, map_location='cpu', weights_only=False)
    model_mha.load_state_dict(ckpt['model_state_dict'])
    del ckpt
    gc.collect()
    print(f"{scale} MHA loaded in {time.time() - t0:.1f}s.")
    
    for N in seq_lengths:
        try:
            n_patched = apply_position_interpolation(model_mha, seq_len=N, train_seq_len=1024)
            ppl, n_seqs, n_toks = evaluate_perplexity(model_mha, val_cache_path, seq_len=N, 
                                                      total_tokens_target=config.total_tokens_target, device=device)
            results['MHA'][N] = ppl
            print(f"[MHA ] N={N:5d} (s={N/1024:.1f}x | {n_patched}/{len(model_mha.layers)} layers patched) | {n_seqs:3d} docs ({n_toks:7d} tok) | PPL: {ppl:7.2f}")
        except torch.cuda.OutOfMemoryError:
            print(f"[MHA ] N={N:5d} | OOM during evaluation.")
            torch.cuda.empty_cache()
            break
            
    del model_mha
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save cache
    os.makedirs(CACHE_PATH, exist_ok=True)
    cache_file = os.path.join(CACHE_PATH, f"{scale}_{config.cache_file_name}")
    with open(cache_file, 'w') as f:
        json_results = {k: {str(n): v for n, v in sub.items()} for k, sub in results.items()}
        json.dump(json_results, f, indent=2)
    print(f"\n[INFO] Results cached to {cache_file}")
    
    # Print Summary Table
    print("\n" + "=" * 65)
    print(f"{'Context Length (N)':<20} | {f'{scale} HOFA (PI)':<20} | {f'{scale} MHA (PI)':<20}")
    print("-" * 65)
    for N in seq_lengths:
        hofa_p = f"{results['HOFA'].get(N, float('nan')):.2f}" if N in results['HOFA'] else "N/A"
        mha_p = f"{results['MHA'].get(N, float('nan')):.2f}" if N in results['MHA'] else "N/A"
        print(f"N = {N:<16d} | {hofa_p:<20} | {mha_p:<20}")
    print("=" * 65 + "\n")
    
    if save_plot:
        plot_long_context_extrapolation(config, results=results)
        
    return results

def plot_long_context_extrapolation(config: LongContextExtrapolationConfig = None, results: dict = None):
    if config is None:
        config = LongContextExtrapolationConfig()
        
    scale = config.scale
    cache_file = os.path.join(CACHE_PATH, f"{scale}_{config.cache_file_name}")
    
    if results is None:
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file {cache_file} not found. Run benchmark first without --plot.")
        with open(cache_file, 'r') as f:
            raw_json = json.load(f)
            results = {k: {int(n): v for n, v in sub.items()} for k, sub in raw_json.items()}
            
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    
    c_hofa = "#1f77b4"
    c_mha = "#d62728"
    
    eval_lengths = sorted([int(N) for N in results['HOFA'].keys() if int(N) in results['MHA']])
    
    hofa_vals = [results['HOFA'][N] for N in eval_lengths]
    mha_vals = [results['MHA'][N] for N in eval_lengths]
    
    ax.plot(eval_lengths, hofa_vals, 'o-', color=c_hofa, linewidth=2.2, markersize=7, 
            label=f'HOFA {scale} (Position Interpolation)')
    ax.plot(eval_lengths, mha_vals, 's--', color=c_mha, linewidth=2.0, markersize=7, 
            label=f'MHA {scale} (Position Interpolation)')

    ax.set_xscale('log', base=2)
    ax.set_xticks(eval_lengths)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    
    # If values are bounded (< 100), use linear scale; else log scale
    max_val = max(max(hofa_vals), max(mha_vals))
    if max_val > 150:
        ax.set_yscale('log')
        ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    
    ax.set_xlabel('Context Length N', fontsize=12, fontweight='medium')
    ax.set_ylabel('Validation Perplexity', fontsize=12, fontweight='medium')
    ax.set_title(f'{scale} Context Length Extrapolation: Validation Perplexity', fontsize=13, fontweight='bold', pad=12)
    
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=10.5, loc='upper left')
    
    for N in [4096, 8192]:
        if N in results['HOFA']:
            val = results['HOFA'][N]
            ax.annotate(f"{val:.2f}", 
                        xy=(N, val), 
                        xytext=(0, -16), 
                        textcoords='offset points', 
                        ha='center', 
                        fontsize=9, 
                        fontweight='bold', 
                        color=c_hofa)
        if N in results['MHA']:
            val = results['MHA'][N]
            ax.annotate(f"{val:.2f}", 
                        xy=(N, val), 
                        xytext=(0, 10), 
                        textcoords='offset points', 
                        ha='center', 
                        fontsize=9, 
                        fontweight='bold', 
                        color=c_mha)
            
    fig.tight_layout()
    
    os.makedirs("data/plots", exist_ok=True)
    plot_pdf = f"data/plots/{scale}_length_extrapolation.pdf"
    
    fig.savefig(plot_pdf, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[INFO] Plot successfully saved to {plot_pdf}")

if __name__ == "__main__":
    run_long_context_experiment()
