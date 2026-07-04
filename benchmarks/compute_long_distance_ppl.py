"""
Long-distance perplexity evaluation (Child et al. 2019).

Flags tokens that depend on context further back than a distance threshold,
then computes perplexity separately for those long-distance tokens vs. all
tokens. A model with weak long-range retrieval will show a gap between the
two numbers.

Usage:
    python benchmarks/compute_long_distance_ppl.py

Output:
    data/experiments_cache/long_distance_ppl_results.pt
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.HybridOutlierFactorizedAttentionTrain import SubwordLM
from training.training_config import LanguageModelingExperimentConfig
from benchmarks.benchmarks_configs import LongDistancePPLConfig, CACHE_PATH


def get_long_distance_mask(input_ids, distance_threshold=512, rare_token_threshold=500):
    """
    Flags tokens that appear at distance > threshold AND are not highly common.
    GPT2 vocab puts most common punctuation, ' a', ' the', etc. in the first few hundred IDs.
    """
    b, n = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for batch_idx in range(b):
        for i in range(distance_threshold, n):
            token = input_ids[batch_idx, i].item()
            if token > rare_token_threshold:
                past_window = input_ids[batch_idx, :i - distance_threshold]
                if token in past_window:
                    mask[batch_idx, i] = True
    return mask


def run_long_distance_ppl(
    train_cfg: LanguageModelingExperimentConfig,
    config: LongDistancePPLConfig = LongDistancePPLConfig(), 
    force_rerun=False
):
    device = torch.device(config.device)
    model_name = train_cfg.model_name
    cache_path_out = os.path.join(CACHE_PATH, f"{model_name}_{config.cache_file_name}")

    # skip eval if cached
    if not force_rerun and os.path.exists(cache_path_out):
        data = torch.load(cache_path_out, map_location="cpu", weights_only=False)
        print(f"Cached results loaded from {cache_path_out}")
        _print_results(data["ppl_all"], data["ppl_long"], data["total_tokens_long"])
        return

    # ── data cache ───────────────────────────────────────────────────
    model_size = model_name.split('_')[0]
    data_cache = f"data/datasets/data_{model_size}_val_cache.bin"
    if not os.path.exists(data_cache):
        raise FileNotFoundError(f"Data cache not found at {data_cache}")

    # ── build model ──────────────────────────────────────────────────
    model = SubwordLM(train_cfg.vocab_size, train_cfg.model_config).to(device)

    ckpt_path = f"data/training/{model_name}/checkpoint.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = {k: v for k, v in ckpt["model_state_dict"].items() if "_cached" not in k}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}")

    # ── evaluate ─────────────────────────────────────────────────────
    batch_size = config.batch_size
    seq_len = config.seq_len
    num_batches = config.num_batches

    total_loss_all = 0.0
    total_tokens_all = 0
    total_loss_long = 0.0
    total_tokens_long = 0
    start_idx = 0

    print(f"Running Long-Distance Perplexity Evaluation on {model_name}...")

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for b_idx in range(num_batches):
                file_size = os.path.getsize(data_cache)
                needed = batch_size * (seq_len + 1)
                if start_idx + needed > file_size // 2:
                    start_idx = 0

                with open(data_cache, "rb") as f:
                    f.seek(start_idx * 2)
                    chunk = np.frombuffer(f.read(needed * 2), dtype=np.uint16).astype(np.int64)
                data = torch.from_numpy(chunk).view(batch_size, seq_len + 1)
                x, y = data[:, :-1].to(device), data[:, 1:].to(device)
                start_idx += needed

                logits, _ = model(x)
                loss_unreduced = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), reduction='none'
                ).view(batch_size, seq_len)

                total_loss_all += loss_unreduced.sum().item()
                total_tokens_all += loss_unreduced.numel()

                mask = get_long_distance_mask(
                    x, distance_threshold=config.distance_threshold,
                    rare_token_threshold=config.rare_token_threshold
                )
                if mask.any():
                    total_loss_long += loss_unreduced[mask].sum().item()
                    total_tokens_long += mask.sum().item()

                if (b_idx + 1) % max(1, num_batches // 10) == 0:
                    print(f"  [{b_idx + 1}/{num_batches}]  long tokens so far: {total_tokens_long}")

    ppl_all = torch.exp(torch.tensor(total_loss_all / total_tokens_all)).item()
    ppl_long = torch.exp(torch.tensor(total_loss_long / max(total_tokens_long, 1))).item() \
        if total_tokens_long > 0 else float('nan')

    # ── cache ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(cache_path_out), exist_ok=True)
    torch.save({
        "ppl_all": ppl_all,
        "ppl_long": ppl_long,
        "total_tokens_long": total_tokens_long,
        "model_name": model_name,
        "config": config,
    }, cache_path_out)
    print(f"\nResults cached to {cache_path_out}")

    _print_results(ppl_all, ppl_long, total_tokens_long)


def _print_results(ppl_all, ppl_long, total_tokens_long):
    print("-" * 45)
    print(f"Standard Perplexity (All Tokens):   {ppl_all:.2f}")
    if not np.isnan(ppl_long):
        print(f"Long-Distance PPL (d > 512):        {ppl_long:.2f}  (Evaluated on {total_tokens_long} tokens)")
    else:
        print(f"Long-Distance PPL:                   N/A  (0 long-distance tokens found)")
    print("-" * 45)


if __name__ == "__main__":
    from training.training_config import make_125M_hofa
    run_long_distance_ppl(make_125M_hofa())