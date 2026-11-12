"""Context length extrapolation benchmark on PG-19 test sequences with canonical YaRN."""

import os
import sys
import gc
import json
import math
import time
from typing import List, Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    from src.yarn import YaRNModel, YaRNMode
except ImportError:
    YaRNModel, YaRNMode = None, None
from training.training_config import YaRNConfig
from benchmarks.benchmarks_configs import LongContextExtrapolationConfig, CACHE_PATH

def format_tokens(n: int) -> str:
    """Formats token count to human-readable string (e.g. 1024 -> '1k', 65536 -> '64k', 131072 -> '128k')."""
    if n >= 1024 * 1024 and n % (1024 * 1024) == 0:
        return f"{n // (1024 * 1024)}M"
    elif n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    return str(n)


def generate_position_buckets(max_seq_len: int, min_bucket_len: int = 1024) -> Dict[str, Tuple[int, int]]:
    """
    Generates dyadic position buckets dynamically up to max_seq_len.
    For 32768: 0k-1k, 1k-2k, 2k-4k, 4k-8k, 8k-16k, 16k-32k
    For 65536: adds 32k-64k
    For 131072: adds 64k-128k
    """
    buckets = {}
    if max_seq_len < min_bucket_len:
        return {f"0-{max_seq_len}": (0, max_seq_len)}

    buckets[f"0k-{format_tokens(min_bucket_len)}"] = (0, min_bucket_len)
    curr = min_bucket_len
    while curr < max_seq_len:
        nxt = min(curr * 2, max_seq_len)
        buckets[f"{format_tokens(curr)}-{format_tokens(nxt)}"] = (curr, nxt)
        curr = nxt
    return buckets


# Default 32k dyadic position buckets for backwards compatibility
POSITION_BUCKETS_32K = generate_position_buckets(32768)


def bootstrap_ci(losses: List[float], num_resamples: int = 2000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    """Computes mean and 95% bootstrap confidence interval over books."""
    if not losses:
        return float('nan'), float('nan'), float('nan')
    if len(losses) == 1:
        return float(losses[0]), float(losses[0]), float(losses[0])

    losses_arr = np.asarray(losses, dtype=np.float64)
    n = len(losses_arr)
    rng = np.random.RandomState(seed)
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


def compute_bucket_losses(logits: torch.Tensor, targets: torch.Tensor, buckets: dict, chunk_size: int = 2048) -> dict:
    """Computes position-bucket NLLs from already-evaluated sequence logits."""
    B, N, V = logits.shape
    out = {}
    y_flat = targets.view(B, N)
    for b_name, (b_start, b_end) in buckets.items():
        if b_end <= N:
            s_loss = 0.0
            for s in range(b_start, b_end, chunk_size):
                e = min(s + chunk_size, b_end)
                sub_l = F.cross_entropy(logits[:, s:e, :].reshape(-1, V), y_flat[:, s:e].reshape(-1), reduction="sum")
                s_loss += sub_l.item()
            out[b_name] = s_loss / (b_end - b_start)
    return out


def evaluate_books_extrapolation(
    model: YaRNModel,
    data_path: str,
    books: List[dict],
    seq_len: int,
    device: str = 'cuda',
    max_books: Optional[int] = None,
    eval_target_tokens: Optional[int] = None,
    cached_book_losses: Optional[List[float]] = None,
    cached_book_indices: Optional[List[int]] = None,
    cached_bucket_losses: Optional[Dict[str, List[float]]] = None,
    compute_buckets: bool = True,
    cache_save_fn: Optional[callable] = None,
) -> Dict:
    """
    Evaluates non-overlapping windows of length seq_len spanning eval_target_tokens from each book.
    Supports incremental resumption from cached books and extracts 32k position buckets.
    """
    bin_data = np.memmap(data_path, dtype=np.uint16, mode='r')
    span = eval_target_tokens if (eval_target_tokens and eval_target_tokens >= seq_len) else seq_len
    n_chunks = max(1, span // seq_len)

    seq_losses: List[float] = []
    book_indices: List[int] = []
    per_book_ppls: List[float] = []
    active_buckets = generate_position_buckets(seq_len) if (compute_buckets and seq_len >= 2048) else {}
    bucket_losses_dict: Dict[str, List[float]] = {k: [] for k in active_buckets}

    # Seed with existing cached books if available
    evaluated_book_ids = set()
    if cached_book_losses and cached_book_indices and len(cached_book_losses) == len(cached_book_indices):
        for b_idx, b_loss in zip(cached_book_indices, cached_book_losses):
            seq_losses.append(b_loss)
            book_indices.append(b_idx)
            per_book_ppls.append(math.exp(b_loss))
            evaluated_book_ids.add(b_idx)
        if cached_bucket_losses:
            for bk, blist in cached_bucket_losses.items():
                if bk in bucket_losses_dict:
                    bucket_losses_dict[bk] = list(blist)

    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for b_info in books:
            b_idx = b_info.get("book_idx", -1)
            if b_idx in evaluated_book_ids:
                continue

            start = b_info["start"]
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

                if active_buckets and c_idx == 0:
                    b_buckets = compute_bucket_losses(logits, y, active_buckets, chunk_size=2048)
                    for bk, bv in b_buckets.items():
                        bucket_losses_dict[bk].append(bv)

                del logits, x, y, toks

            if book_chunk_losses:
                book_mean_loss = float(sum(book_chunk_losses) / len(book_chunk_losses))
                seq_losses.append(book_mean_loss)
                book_indices.append(b_idx)
                per_book_ppls.append(math.exp(book_mean_loss))
                evaluated_book_ids.add(b_idx)

            if seq_len >= 8192:
                torch.cuda.empty_cache()

            if cache_save_fn is not None and len(seq_losses) % 5 == 0:
                mean_l, low_l, high_l = bootstrap_ci(seq_losses)
                interim_metrics = {
                    'seq_len': seq_len,
                    'num_books_evaluated': len(seq_losses),
                    'total_tokens': len(seq_losses) * span,
                    'loss_mean': mean_l,
                    'loss_ci_95': [low_l, high_l],
                    'ppl_mean': math.exp(mean_l) if not math.isnan(mean_l) else float('nan'),
                    'ppl_ci_95': [math.exp(low_l), math.exp(high_l)],
                    'per_book_losses': seq_losses,
                    'book_indices': book_indices,
                    'per_book_ppls': per_book_ppls,
                }
                cache_save_fn(interim_metrics)

    torch.cuda.empty_cache()
    gc.collect()

    mean_loss, low_loss, high_loss = bootstrap_ci(seq_losses)
    ppl_mean = math.exp(mean_loss) if not math.isnan(mean_loss) else float('nan')
    ppl_low = math.exp(low_loss) if not math.isnan(low_loss) else float('nan')
    ppl_high = math.exp(high_loss) if not math.isnan(high_loss) else float('nan')

    out_dict = {
        'seq_len': seq_len,
        'num_books_evaluated': len(seq_losses),
        'total_tokens': len(seq_losses) * span,
        'loss_mean': mean_loss,
        'loss_ci_95': [low_loss, high_loss],
        'ppl_mean': ppl_mean,
        'ppl_ci_95': [ppl_low, ppl_high],
        'per_book_losses': seq_losses,
        'book_indices': book_indices,
        'per_book_ppls': per_book_ppls,
    }

    if active_buckets and bucket_losses_dict:
        bucket_summary = {}
        for bk, blist in bucket_losses_dict.items():
            bm, bl, bh = bootstrap_ci(blist)
            bucket_summary[bk] = {
                'loss_mean': bm,
                'loss_ci_95': [bl, bh],
                'ppl_mean': math.exp(bm) if not math.isnan(bm) else float('nan'),
                'ppl_ci_95': [math.exp(bl), math.exp(bh)] if not math.isnan(bl) else [float('nan'), float('nan')],
                'per_book_losses': blist,
            }
        out_dict['position_buckets'] = bucket_summary
        if seq_len == 32768:
            out_dict['position_buckets_32k'] = bucket_summary

    return out_dict


def compute_paired_book_statistics(results: dict, scale_prefix: str = "70M", hofa_name: Optional[str] = None, mha_name: Optional[str] = None) -> dict:
    """Computes paired per-book loss differences between HOFA and MHA (or HOFA_geom vs HOFA)."""
    h_key = hofa_name or f"{scale_prefix}_HOFA"
    m_key = mha_name or f"{scale_prefix}_MHA"

    if h_key not in results or m_key not in results:
        return {}

    paired_summary = {"lengths": {}, "position_buckets": {}, "position_buckets_32k": {}}
    h_model_data = results[h_key]
    m_model_data = results[m_key]

    common_lengths = sorted([int(k) for k in h_model_data if k.isdigit() and k in m_model_data])
    for N in common_lengths:
        sN = str(N)
        h_entry = h_model_data[sN]
        m_entry = m_model_data[sN]

        h_idx_map = {idx: loss for idx, loss in zip(h_entry.get("book_indices", []), h_entry.get("per_book_losses", []))}
        m_idx_map = {idx: loss for idx, loss in zip(m_entry.get("book_indices", []), m_entry.get("per_book_losses", []))}

        shared_b_ids = sorted(list(set(h_idx_map.keys()) & set(m_idx_map.keys())))
        if not shared_b_ids:
            continue

        h_losses = np.array([h_idx_map[i] for i in shared_b_ids])
        m_losses = np.array([m_idx_map[i] for i in shared_b_ids])
        diff_losses = h_losses - m_losses

        dl_mean, dl_low, dl_high = bootstrap_ci(diff_losses.tolist())
        hofa_wins = int(np.sum(diff_losses < 0))
        total_shared = len(shared_b_ids)

        h_ppl = h_entry.get("ppl_mean", math.exp(np.mean(h_losses)))
        m_ppl = m_entry.get("ppl_mean", math.exp(np.mean(m_losses)))
        delta_ppl = h_ppl - m_ppl
        rel_ppl_reduction = ((m_ppl - h_ppl) / m_ppl) * 100.0 if m_ppl > 0 else 0.0

        paired_summary["lengths"][sN] = {
            "seq_len": N,
            "num_shared_books": total_shared,
            "hofa_ppl": h_ppl,
            "mha_ppl": m_ppl,
            "delta_ppl": delta_ppl,
            "rel_ppl_reduction_pct": rel_ppl_reduction,
            "paired_loss_diff_mean": dl_mean,
            "paired_loss_diff_ci_95": [dl_low, dl_high],
            "per_book_loss_diffs": diff_losses.tolist(),
            "shared_book_indices": shared_b_ids,
            "hofa_books_won": hofa_wins,
            "total_books": total_shared,
        }

    # Dynamic position buckets for the largest common sequence length that evaluated buckets
    bucket_candidates = []
    for N in common_lengths:
        sN = str(N)
        hb = h_model_data[sN].get("position_buckets") or h_model_data[sN].get("position_buckets_32k")
        mb = m_model_data[sN].get("position_buckets") or m_model_data[sN].get("position_buckets_32k")
        if hb and mb:
            bucket_candidates.append(N)

    if bucket_candidates:
        target_N = max(bucket_candidates)
        s_target = str(target_N)
        hb = h_model_data[s_target].get("position_buckets") or h_model_data[s_target].get("position_buckets_32k", {})
        mb = m_model_data[s_target].get("position_buckets") or m_model_data[s_target].get("position_buckets_32k", {})
        for bk in hb:
            if bk in mb:
                h_b_losses = np.array(hb[bk].get("per_book_losses", []))
                m_b_losses = np.array(mb[bk].get("per_book_losses", []))
                n_b = min(len(h_b_losses), len(m_b_losses))
                if n_b > 0:
                    diff_b = h_b_losses[:n_b] - m_b_losses[:n_b]
                    bm, bl, bh = bootstrap_ci(diff_b.tolist())
                    h_ppl_b = hb[bk].get("ppl_mean", math.exp(np.mean(h_b_losses[:n_b])))
                    m_ppl_b = mb[bk].get("ppl_mean", math.exp(np.mean(m_b_losses[:n_b])))
                    bucket_stat = {
                        "hofa_ppl": h_ppl_b,
                        "mha_ppl": m_ppl_b,
                        "delta_ppl": h_ppl_b - m_ppl_b,
                        "paired_loss_diff_mean": bm,
                        "paired_loss_diff_ci_95": [bl, bh],
                        "hofa_books_won": int(np.sum(diff_b < 0)),
                        "total_books": n_b,
                    }
                    paired_summary["position_buckets"][bk] = bucket_stat
                    if target_N == 32768:
                        paired_summary["position_buckets_32k"][bk] = bucket_stat

    return paired_summary


def run_long_context_experiment(config: Optional[LongContextExtrapolationConfig] = None, save_plot: bool = True):
    if config is None:
        config = LongContextExtrapolationConfig()

    device = config.device
    models: List[YaRNConfig] = [config.models] if isinstance(config.models, YaRNConfig) else list(config.models)
    seq_lengths = list(config.seq_lengths)

    # Determine use_fixed_yarn setting
    is_all_cpt = all("_cpt" in getattr(m.model, "model_name", "").lower() for m in models)
    if config.use_fixed_yarn is None:
        use_fixed_yarn = is_all_cpt
    else:
        use_fixed_yarn = config.use_fixed_yarn

    eval_target_tokens = getattr(config, 'eval_target_tokens', None)

    mode_str = "Fixed s=16.0" if use_fixed_yarn else "Dynamic YaRN"
    model_names = [m.model.model_name for m in models]
    print(f"[Extrapolation] Models: {model_names} | N: {seq_lengths} | Mode: {mode_str} | Target: {eval_target_tokens} tok/book | Device: {device}")

    if not os.path.exists(config.data_path) or not os.path.exists(config.index_path):
        raise FileNotFoundError(f"PG-19 test dataset not found: {config.data_path} or {config.index_path}")

    with open(config.index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    all_books = index_data.get("books", [])
    total_books_in_split = len(all_books)

    req_len = max(max(seq_lengths), eval_target_tokens or max(seq_lengths))
    eligible_books = [b for b in all_books if b["length"] >= req_len + 1]
    max_b = getattr(config, 'max_books', None)
    target_eligible = eligible_books if max_b is None else eligible_books[:max_b]
    n_books = len(target_eligible)
    print(f"Eligible books (>= {req_len+1} tokens): {n_books}/{total_books_in_split}\n")

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

        # Check if all sequence lengths are already fully evaluated for this model
        lengths_to_run = []
        for N in seq_lengths:
            cached_entry = results.get(model_name, {}).get(str(N), {})
            num_eval = cached_entry.get('num_books_evaluated', 0)
            if config.force_rerun or num_eval < n_books:
                lengths_to_run.append(N)
            else:
                ppl = cached_entry['ppl_mean']
                ci = cached_entry.get('ppl_ci_95', [ppl, ppl])
                print(f"[{model_name}] N={N:5d} | Cached result: PPL={ppl:6.2f} [{ci[0]:5.2f}, {ci[1]:5.2f}] ({num_eval}/{n_books} books)")

        if not lengths_to_run:
            print(f"[{model_name}] All requested lengths complete. Skipping checkpoint load.")
            continue

        print(f"[{model_name}] Initializing...")
        t_init = time.time()
        yarn_model = YaRNModel(yarn_cfg, device=device, dtype=torch.bfloat16)
        print(f"[{model_name}] Ready ({time.time() - t_init:.1f}s)")

        try:
            for N in lengths_to_run:
                cached_entry = results.get(model_name, {}).get(str(N), {})
                cached_losses = cached_entry.get("per_book_losses") if not config.force_rerun else None
                cached_indices = cached_entry.get("book_indices") if not config.force_rerun else None
                raw_cached_buckets = cached_entry.get("position_buckets") or cached_entry.get("position_buckets_32k", {})
                cached_buckets = {k: v.get("per_book_losses", []) for k, v in raw_cached_buckets.items()} if (not config.force_rerun and raw_cached_buckets) else None

                t0 = time.time()
                if use_fixed_yarn:
                    scale_val = max(16.0, float(N) / float(yarn_model.original_max_seq_len))
                else:
                    scale_val = max(1.0, float(N) / float(yarn_model.original_max_seq_len))
                ymode = getattr(yarn_cfg, 'yarn_mode', None) or YaRNMode.default()
                n_patched = yarn_model.set_context_length(N, scale=scale_val, yarn_mode=ymode)
                s_eff = scale_val if scale_val is not None else (N / yarn_model.original_max_seq_len)

                def save_progress(metrics_dict):
                    results[model_name][str(N)] = metrics_dict
                    with open(cache_file, "w") as f:
                        json.dump(results, f, indent=2)

                ymode_str = ymode.value if hasattr(ymode, "value") else str(ymode)
                print(f"[{model_name}] N={N:5d} (s={s_eff:.1f}x {ymode_str}) | Running {n_books} books (cached {len(cached_losses or [])})...")

                metrics = evaluate_books_extrapolation(
                    yarn_model,
                    data_path=config.data_path,
                    books=target_eligible,
                    seq_len=N,
                    device=device,
                    eval_target_tokens=eval_target_tokens,
                    cached_book_losses=cached_losses,
                    cached_book_indices=cached_indices,
                    cached_bucket_losses=cached_buckets,
                    compute_buckets=getattr(config, 'compute_position_buckets', True),
                    cache_save_fn=save_progress,
                )
                results[model_name][str(N)] = metrics

                ppl = metrics['ppl_mean']
                ppl_low, ppl_high = metrics['ppl_ci_95']
                n_bks = metrics['num_books_evaluated']
                n_toks = metrics['total_tokens']
                elapsed = time.time() - t0

                print(f"[{model_name}] N={N:5d} | {n_bks:2d} books ({n_toks:7d} tok) | Loss: {metrics['loss_mean']:.4f} | PPL: {ppl:6.2f} [{ppl_low:6.2f}, {ppl_high:6.2f}] ({elapsed:.1f}s)")

                with open(cache_file, "w") as f:
                    json.dump(results, f, indent=2)

        except torch.cuda.OutOfMemoryError:
            print(f"[{model_name}] CUDA OutOfMemory encountered. Stopping sweep.")
            torch.cuda.empty_cache()
        finally:
            del yarn_model
            gc.collect()
            torch.cuda.empty_cache()

    print("\n=== Paired Summary (HOFA vs. MHA) ===")

    for scale in ["70M", "125M", "350M"]:
        paired_dict = compute_paired_book_statistics(results, scale_prefix=scale)
        if paired_dict and "lengths" in paired_dict:
            results[f"paired_{scale}"] = paired_dict
            print(f"\n--- Scale: {scale} ---")
            for N in seq_lengths:
                sN = str(N)
                if sN in paired_dict["lengths"]:
                    pd = paired_dict["lengths"][sN]
                    print(
                        f"  N={N:5d} | HOFA PPL={pd['hofa_ppl']:6.2f} vs MHA PPL={pd['mha_ppl']:6.2f} | "
                        f"dPPL={pd['delta_ppl']:+6.2f} ({pd['rel_ppl_reduction_pct']:+5.1f}%) | "
                        f"Paired dLoss={pd['paired_loss_diff_mean']:+.4f} [{pd['paired_loss_diff_ci_95'][0]:+.4f}, {pd['paired_loss_diff_ci_95'][1]:+.4f}] | "
                        f"Won: {pd['hofa_books_won']}/{pd['total_books']}"
                    )

            p_buckets = paired_dict.get("position_buckets") or paired_dict.get("position_buckets_32k", {})
            if p_buckets:
                print(f"  [Position Depth Buckets ({scale})]")
                for bk, pb in p_buckets.items():
                    print(
                        f"    Bucket {bk:8s} | HOFA={pb['hofa_ppl']:6.2f} vs MHA={pb['mha_ppl']:6.2f} | "
                        f"dPPL={pb['delta_ppl']:+6.2f} | "
                        f"Paired dLoss={pb['paired_loss_diff_mean']:+.4f} [{pb['paired_loss_diff_ci_95'][0]:+.4f}, {pb['paired_loss_diff_ci_95'][1]:+.4f}] | "
                        f"Won: {pb['hofa_books_won']}/{pb['total_books']}"
                    )

    # Check for 70M_HOFA_geom comparison
    if "70M_HOFA_geom" in results and "70M_HOFA" in results:
        paired_geom = compute_paired_book_statistics(results, hofa_name="70M_HOFA_geom", mha_name="70M_HOFA")
        results["paired_70M_geom_vs_hofa"] = paired_geom
        print("\n--- 70M Geometric HOFA vs. Original HOFA ---")
        for N in seq_lengths:
            sN = str(N)
            if sN in paired_geom.get("lengths", {}):
                pd = paired_geom["lengths"][sN]
                print(
                    f"  N={N:5d} | Geom PPL={pd['hofa_ppl']:6.2f} vs Orig HOFA PPL={pd['mha_ppl']:6.2f} | "
                    f"dPPL={pd['delta_ppl']:+6.2f} ({pd['rel_ppl_reduction_pct']:+5.1f}%) | "
                    f"dLoss={pd['paired_loss_diff_mean']:+.4f} [{pd['paired_loss_diff_ci_95'][0]:+.4f}, {pd['paired_loss_diff_ci_95'][1]:+.4f}] | "
                    f"Won: {pd['hofa_books_won']}/{pd['total_books']}"
                )

    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll updated results cached to {cache_file}\n")

    if save_plot:
        plot_long_context(config, results=results)

    return results


def plot_long_context(config: Optional[LongContextExtrapolationConfig] = None, results: Optional[dict] = None):
    if config is None:
        config = LongContextExtrapolationConfig()

    cache_file = os.path.join(CACHE_PATH, config.cache_file_name)
    if results is None:
        if not os.path.exists(cache_file):
            legacy_file = os.path.join(CACHE_PATH, "long_context_extrapolation_results.json")
            if os.path.exists(legacy_file):
                cache_file = legacy_file
            else:
                raise FileNotFoundError(f"Cache file {cache_file} not found.")
        with open(cache_file, "r") as f:
            results = json.load(f)

    plot_dir = "data/plots/length_extrapolation"
    os.makedirs(plot_dir, exist_ok=True)
    base_name = getattr(config, 'plot_file_name', None) or (config.cache_file_name.replace('.json', '') if config and config.cache_file_name else "yarn_length_extrapolation")

    # If all three scales are present, generate the publication 3-panel Figures (Figures 1, 2, 3)
    has_cross_scale = all(f"{s}_HOFA" in results and f"{s}_MHA" in results for s in ["70M", "125M", "350M"])

    if has_cross_scale:
        _plot_cross_scale_figures(results, plot_dir, base_name)
    else:
        _plot_standard_extrapolation(results, config, plot_dir, base_name)


def _plot_cross_scale_figures(results: dict, plot_dir: str, base_name: str):
    scales = ["70M", "125M", "350M"]
    color_hofa = "#1f77b4"
    color_mha = "#d62728"

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
    })

    # Figure 1: 3-Panel Cross-Scale Extrapolation (Corpus PPL vs N)
    fig1, axes1 = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    for col, scale in enumerate(scales):
        ax = axes1[col]
        h_data = results[f"{scale}_HOFA"]
        m_data = results[f"{scale}_MHA"]
        common_lengths = sorted([int(k) for k in h_data if k.isdigit() and k in m_data])
        x_idx = np.arange(len(common_lengths))
        x_labels = [format_tokens(N) for N in common_lengths]

        h_ppls = [h_data[str(N)]["ppl_mean"] for N in common_lengths]
        m_ppls = [m_data[str(N)]["ppl_mean"] for N in common_lengths]
        h_lows = [h_data[str(N)]["ppl_ci_95"][0] for N in common_lengths]
        h_highs = [h_data[str(N)]["ppl_ci_95"][1] for N in common_lengths]
        m_lows = [m_data[str(N)]["ppl_ci_95"][0] for N in common_lengths]
        m_highs = [m_data[str(N)]["ppl_ci_95"][1] for N in common_lengths]

        ax.errorbar(x_idx, m_ppls, yerr=[np.array(m_ppls)-np.array(m_lows), np.array(m_highs)-np.array(m_ppls)],
                    marker="s", linestyle="--", color=color_mha, linewidth=2.0, capsize=3.5, label=f"{scale} MHA")
        ax.errorbar(x_idx, h_ppls, yerr=[np.array(h_ppls)-np.array(h_lows), np.array(h_highs)-np.array(h_ppls)],
                    marker="o", linestyle="-", color=color_hofa, linewidth=2.2, capsize=3.5, label=f"{scale} HOFA")

        ax.set_title(f"{scale} Base Extrapolation", fontweight="bold")
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Context Length (Tokens)")
        if col == 0:
            ax.set_ylabel("Perplexity on PG-19 Test")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", frameon=True)

    fig1_pdf = os.path.join(plot_dir, "Figure1_cross_scale_extrapolation.pdf")
    fig1.savefig(fig1_pdf, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved Figure 1 to {fig1_pdf}")

    # Figure 2: Position-Dependent Loss Differences
    fig2, axes2 = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    for col, scale in enumerate(scales):
        ax = axes2[col]
        paired_dict = results.get(f"paired_{scale}", {})
        paired_p = paired_dict.get("position_buckets") or paired_dict.get("position_buckets_32k", {})
        if not paired_p:
            continue
        buckets = list(paired_p.keys())
        b_idx = np.arange(len(buckets))
        d_losses = [paired_p[bk]["paired_loss_diff_mean"] for bk in buckets]
        d_lows = [paired_p[bk]["paired_loss_diff_ci_95"][0] for bk in buckets]
        d_highs = [paired_p[bk]["paired_loss_diff_ci_95"][1] for bk in buckets]

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.errorbar(b_idx, d_losses,
                    yerr=[np.array(d_losses)-np.array(d_lows), np.array(d_highs)-np.array(d_losses)],
                    marker="o", color="#0b3c5d", linewidth=2.0, capsize=4.0, label="Paired dLoss (HOFA - MHA)")

        ax.set_title(f"{scale} Context-Depth Tradeoff", fontweight="bold")
        ax.set_xticks(b_idx)
        ax.set_xticklabels(buckets, rotation=20)
        ax.set_xlabel("Position Depth Bucket")
        if col == 0:
            ax.set_ylabel("Paired dLoss (Nats)\n(< 0 means HOFA better)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", frameon=True)

    fig2_pdf = os.path.join(plot_dir, "Figure2_position_dependent_loss.pdf")
    fig2.savefig(fig2_pdf, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved Figure 2 to {fig2_pdf}")

    # Figure 3: Per-Book Robustness Distributions
    fig3, axes3 = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    for col, scale in enumerate(scales):
        ax = axes3[col]
        paired_l = results.get(f"paired_{scale}", {}).get("lengths", {})
        if not paired_l:
            continue
        common_lengths = sorted([int(k) for k in paired_l])
        x_idx = np.arange(len(common_lengths))
        x_labels = [format_tokens(N) for N in common_lengths]

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        rng = np.random.RandomState(42 + col)
        means, lows, highs = [], [], []

        for i, N in enumerate(common_lengths):
            diffs = np.array(paired_l[str(N)]["per_book_loss_diffs"])
            jitter = rng.uniform(-0.14, 0.14, size=len(diffs))
            ax.scatter(np.full_like(diffs, i, dtype=float) + jitter, diffs, color=color_hofa, alpha=0.45, s=20)
            means.append(paired_l[str(N)]["paired_loss_diff_mean"])
            lows.append(paired_l[str(N)]["paired_loss_diff_ci_95"][0])
            highs.append(paired_l[str(N)]["paired_loss_diff_ci_95"][1])

        ax.errorbar(x_idx, means, yerr=[np.array(means)-np.array(lows), np.array(highs)-np.array(means)],
                    fmt="o-", color="#0b3c5d", linewidth=2.0, capsize=4.0, label="Mean dLoss [95% CI]")

        ax.set_title(f"{scale} Per-Book Paired dLoss", fontweight="bold")
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Context Length (Tokens)")
        if col == 0:
            ax.set_ylabel("Per-Book dLoss (Nats)\n(< 0 means HOFA better)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", frameon=True)
        ax.set_xlabel("Context Length (Tokens)")
        if col == 0:
            ax.set_ylabel("Per-Book dLoss (Nats)\n(< 0 means HOFA better)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", frameon=True)

    fig3_pdf = os.path.join(plot_dir, "Figure3_per_book_robustness.pdf")
    fig3.savefig(fig3_pdf, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved Figure 3 to {fig3_pdf}")


def _plot_standard_extrapolation(results: dict, config: LongContextExtrapolationConfig, plot_dir: str, base_name: str):
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)

    target_models = []
    if config and config.models:
        model_list = config.models if isinstance(config.models, list) else [config.models]
        for m in model_list:
            if isinstance(m, YaRNConfig):
                target_models.append(getattr(m.model, "model_name", str(m)))
            else:
                target_models.append(getattr(m, "model_name", str(m)))

    # Determine scale for title/labels if available
    scale_label = ""
    if base_name and any(s in base_name for s in ["70M", "125M", "350M"]):
        for s in ["70M", "125M", "350M"]:
            if s in base_name:
                scale_label = f" ({s})"
                break

    def model_sort_key(m):
        m_lower = m.lower()
        if "mha" in m_lower:
            return 0
        elif "alpha" in m_lower:
            return 2
        return 1

    sorted_models = sorted(results.keys(), key=model_sort_key)

    plotted_count = 0
    for idx, m_name in enumerate(sorted_models):
        m_results = results[m_name]
        if not isinstance(m_results, dict) or m_name.startswith("paired_"):
            continue
        if target_models and m_name not in target_models:
            continue
        lengths = sorted([int(k) for k in m_results.keys() if k.isdigit()])
        if not lengths:
            continue

        # Skip uninitialized / diverged runs (e.g. PPL > 500)
        first_len = lengths[0]
        if m_results.get(str(first_len), {}).get("ppl_mean", 0) > 500:
            continue

        vals = [m_results[str(n)]['ppl_mean'] for n in lengths]
        lows = [m_results[str(n)]['ppl_ci_95'][0] for n in lengths]
        highs = [m_results[str(n)]['ppl_ci_95'][1] for n in lengths]

        # Style based on model identity
        if "alpha" in m_name.lower():
            color = "#1f77b4"  # Primary blue
            marker = "o-"
            label = "HOFA (Learnable $\\alpha$)"
        elif "hofa" in m_name.lower():
            color = "#2ca02c"  # Distinct green for base HOFA
            marker = "^-."
            label = "HOFA"
        elif "mha" in m_name.lower():
            color = "#d62728"  # Crimson red
            marker = "s--"
            label = "MHA Baseline"
        else:
            palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
            color = palette[idx % len(palette)]
            marker = "o-"
            label = m_name

        ax.plot(lengths, vals, marker, color=color, linewidth=2.0, markersize=7, label=label, zorder=3)
        plotted_count += 1

    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 2048, 4096, 8192, 16384, 32768, 65536])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1024)}k" if x >= 1024 else str(int(x))))
    ax.set_xlabel("Context Length (Tokens)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Perplexity on PG-19 Test", fontsize=11, fontweight="bold")
    ax.set_title(f"PG-19 Context Extrapolation{scale_label}", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=10.5, loc="upper left")
    fig.tight_layout()

    plot_pdf = os.path.join(plot_dir, f"{base_name}.pdf")
    plot_png = os.path.join(plot_dir, f"{base_name}.png")
    fig.savefig(plot_pdf, bbox_inches="tight")
    fig.savefig(plot_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_pdf} and {plot_png}")


if __name__ == "__main__":
    run_long_context_experiment()
