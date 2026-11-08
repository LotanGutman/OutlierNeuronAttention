"""Position-segment pre-softmax temperature scaling benchmark."""

import os
import sys

# Ensure repository root is on sys.path when script is executed directly
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import gc
import json
import math
import time
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np

from src.yarn import YaRNModel, YaRNMode
from training.training_config import YaRNConfig
from benchmarks.benchmarks_configs import TemperatureYaRNConfig, CACHE_PATH
from benchmarks.benchmark_long_context import compute_chunked_loss


def evaluate_segments_for_scale(
    model: YaRNModel,
    data_path: str,
    books: List[dict],
    seq_len: int,
    chunk_size: int,
    scale_factor: float,
    scale_val: float,
    ymode: YaRNMode,
    device: str = "cuda",
    max_books: Optional[int] = None,
) -> Dict:
    """Evaluates 1/sqrt(t) scale factor across non-overlapping token chunks."""
    bin_data = np.memmap(data_path, dtype=np.uint16, mode='r')
    target_books = books if max_books is None else books[:max_books]
    num_chunks = seq_len // chunk_size

    # Set temperature scaling (mscale = 1/sqrt(t)) directly into RoPE buffers
    model.set_context_length(seq_len, scale=scale_val, yarn_mode=ymode, mscale=scale_factor)

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
    print("Position-Segment Pre-Softmax Temperature Scaling Benchmark")
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
                scale_val=scale_val,
                ymode=ymode,
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


if __name__ == "__main__":
    from benchmarks.plotting.plot_temperature_YaRN import plot_temperature_yarn
    cfg = TemperatureYaRNConfig()
    if "--plot" in sys.argv:
        plot_temperature_yarn(cfg)
    else:
        run_temperature_yarn_experiment(cfg)
        plot_temperature_yarn(cfg)

