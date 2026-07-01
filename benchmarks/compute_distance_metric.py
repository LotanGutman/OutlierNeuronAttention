import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.HybridOutlierFactorizedAttentionTrain import SubwordLM
from src.modules.modules import apply_rotary_pos_emb
from training.training_config import LanguageModelingExperimentConfig
from benchmarks.benchmarks_configs import DistanceMetricConfig, CACHE_PATH
from benchmarks.plotting.plot_distance_metric import plot_distance_metric


# ── helpers ──────────────────────────────────────────────────────────

def load_batch(cache_path, batch_size, seq_len, start_idx):
    """Read a batch of tokenised text from the binary cache."""
    file_size = os.path.getsize(cache_path)
    total_tokens = file_size // 2
    needed = batch_size * (seq_len + 1)
    if start_idx + needed > total_tokens:
        start_idx = 0
    with open(cache_path, "rb") as f:
        f.seek(start_idx * 2)
        data = np.frombuffer(f.read(needed * 2), dtype=np.uint16).astype(np.int64)
    data = torch.from_numpy(data).view(batch_size, seq_len + 1)
    return data[:, :-1].contiguous(), data[:, 1:].contiguous(), start_idx + needed


def compute_effective_distances(model, x, device):
    """
    Run a forward pass with hooks that reconstruct the full N x N implicit 
    attention distributions for both the exact pathway and the GLA pathway,
    mixing them using the alpha gate to compute true effective distance.
    """
    results = [None] * len(model.layers)

    def make_hook(layer_idx):
        attn = model.layers[layer_idx].attn
        r = attn.r
        d_head = attn.d_head

        def hook(module, inp, _out):
            x_in = inp[0]
            b, n, _d = x_in.shape
            scale_factor = d_head ** 0.25

            # 1. Project and scale inputs
            q_unscaled = module.W_q(x_in).view(b, n, module.num_heads, d_head)
            k_unscaled = module.W_k(x_in).view(b, n, module.num_heads, d_head)
            
            Q = q_unscaled / scale_factor
            K = k_unscaled / scale_factor
            
            Q_trans = Q.transpose(1, 2) # (B, H, N, D)
            K_trans = K.transpose(1, 2)

            if r == 0 or r == d_head:
                return

            # ---------------------------------------------------------
            # PATHWAY 1: EXACT OUTLIER ATTENTION
            # ---------------------------------------------------------
            Q_O = Q_trans[..., :r].contiguous()
            K_O = K_trans[..., :r].contiguous()

            if getattr(module, "rotary_emb", None) is not None:
                cos, sin = module.rotary_emb(n)
                Q_O, K_O = apply_rotary_pos_emb(Q_O, K_O, cos, sin)

            sm_scale = (d_head / r) ** 0.5
            attn_logits = torch.einsum("bhnd,bhmd->bhnm", Q_O, K_O) * sm_scale

            mask = torch.triu(torch.full((n, n), float("-inf"), device=device), diagonal=1)
            attn_logits = attn_logits + mask[None, None, :, :]
            P_O = torch.softmax(attn_logits, dim=-1) # (B, H, N, N)

            # ---------------------------------------------------------
            # PATHWAY 2: INLIER GATED LINEAR ATTENTION (GLA)
            # ---------------------------------------------------------
            Q_I = Q_trans[..., r:].contiguous()
            K_I = K_trans[..., r:].contiguous()
            
            # Reconstruct the dynamic gates robustly by scanning module params
            qk_cat = torch.cat([Q, K], dim=-1) # (B, N, H, 2*d_head)
            
            
            def extract_gate(mod, names_w, names_b):
                w, b_val = None, 0.0
                for nw in names_w:
                    if hasattr(mod, nw): w = getattr(mod, nw); break
                for nb in names_b:
                    if hasattr(mod, nb): b_val = getattr(mod, nb); break
                        
                if w is not None:
                    if isinstance(w, torch.nn.Linear):
                        # qk_cat is (B, N, H, 2*d_head). w(qk_cat) -> (B, N, H, H)
                        logits = w(qk_cat)
                        if logits.dim() == 4 and logits.shape[-1] == module.num_heads:
                            # Extract the h-th output for the h-th head
                            logits = torch.diagonal(logits, dim1=-2, dim2=-1) # (B, N, H)
                        else:
                            logits = logits.squeeze(-1)
                        return torch.sigmoid(logits).transpose(1, 2) # (B, H, N)
                    else:
                        logits = torch.einsum('bnhd,d->bnh', qk_cat, w) if w.dim() == 1 else torch.einsum('bnhd,hd->bnh', qk_cat, w)
                        logits = logits + (b_val.view(1, 1, -1) if isinstance(b_val, torch.Tensor) and b_val.dim() == 1 else b_val)
                        return torch.sigmoid(logits).transpose(1, 2) # (B, H, N)
                return torch.full((b, module.num_heads, n), 0.5, device=device)
            

            g = extract_gate(module, ['w_g', 'W_g', 'gate_proj', 'g_proj'], ['b_g', 'gate_bias', 'g_bias'])
            alpha = extract_gate(module, ['w_mix', 'W_mix', 'mix_proj', 'alpha_proj'], ['b_mix', 'mix_bias', 'alpha_bias'])

            # Construct the GLA Decay Matrix (D_ij)
            gamma = 1.0 - g
            log_gamma = torch.log(torch.clamp(gamma, min=1e-6))
            cumsum_log_gamma = torch.cumsum(log_gamma, dim=-1)
            
            # Shift cumsum by 1 to represent strictly causal state (S_{t-1})
            cumsum_shift = torch.cat([torch.zeros_like(cumsum_log_gamma[..., :1]), cumsum_log_gamma[..., :-1]], dim=-1)
            
            # Decay exponent: c_{i-1} - c_j
            decay_diff = cumsum_shift.unsqueeze(-1) - cumsum_log_gamma.unsqueeze(-2)
            
            # PREVENT inf * 0 = NaN: Mask the upper triangle with -inf BEFORE exp()
            strict_causal_mask = torch.tril(torch.ones((n, n), device=device, dtype=torch.bool), diagonal=-1)
            decay_diff = torch.where(strict_causal_mask, decay_diff, float('-inf'))
            
            D = torch.exp(decay_diff)
            
            # Implicit GLA Attention Matrix
            W_I = torch.einsum("bhnd,bhmd->bhnm", Q_I, K_I) * D
            
            # Convert to absolute probability mass (Standard for evaluating linear attention flow)
            P_I = torch.abs(W_I)
            P_I_sum = P_I.sum(dim=-1, keepdim=True) + 1e-9
            P_I = P_I / P_I_sum

            # ---------------------------------------------------------
            # FINAL COMBINATION & DISTANCE CALCULATION
            # ---------------------------------------------------------
            alpha_exp = alpha.unsqueeze(-1) # (B, H, N, 1)
            P_mixed = (alpha_exp * P_O) + ((1.0 - alpha_exp) * P_I)

            pos = torch.arange(n, device=device, dtype=torch.float32)
            i_minus_j = torch.clamp(pos.unsqueeze(1) - pos.unsqueeze(0), min=0) # (N, N)            
            
            # Calculate expected distance
            d = (P_mixed * i_minus_j).sum(dim=-1) # (B, H, N)
            results[layer_idx] = d.detach().cpu().to(torch.float32)

        return hook

    handles = []
    for i, layer in enumerate(model.layers):
        h = layer.attn.register_forward_hook(make_hook(i))
        handles.append(h)

    try:
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(x.to(device))
    finally:
        for h in handles:
            h.remove()

    return results


# ── main ─────────────────────────────────────────────────────────────

def run_distance_metric(config: DistanceMetricConfig = DistanceMetricConfig(), force_rerun=False):
    device = torch.device(config.device)
    model_name = config.model_name
    cache_path_out = os.path.join(CACHE_PATH, config.cache_file_name)

    # skip eval if cached results exist
    if not force_rerun and os.path.exists(cache_path_out):
        print(f"Cached results found at {cache_path_out}, skipping eval.")
        plot_distance_metric(model_name=model_name, cache_file=config.cache_file_name)
        return

    num_heads = None

    # ── build model ──────────────────────────────────────────────────
    train_cfg = LanguageModelingExperimentConfig()
    model_cfg = train_cfg.model_config
    num_heads = model_cfg.num_heads
    num_layers = model_cfg.num_layers
    print(f"Model: {model_name}  d_model={model_cfg.d_model}  heads={num_heads}  "
          f"layers={num_layers}  r={model_cfg.r}")

    model = SubwordLM(train_cfg.vocab_size, model_cfg).to(device)
    model.eval()

    ckpt_path = f"data/training/{model_name}/checkpoint.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = {k: v for k, v in ckpt["model_state_dict"].items()
             if "_cached_outlier_idx" not in k and "_cached_inlier_idx" not in k}
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint from {ckpt_path}")

    # ── evaluate ────────────────────────────────────────────────────
    cache_path = f"data/datasets/data_{model_name}_cache.bin"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Data cache not found at {cache_path}")

    batch_size = config.batch_size
    seq_len = config.seq_len
    num_batches = config.num_sequences // batch_size

    layer_dists = []
    start_idx = 0
    total_positions = 0
    t0 = time.time()

    for b_idx in range(num_batches):
        x, _y, start_idx = load_batch(cache_path, batch_size, seq_len, start_idx)
        dists = compute_effective_distances(model, x, device)

        for li, d in enumerate(dists):
            if d is None:
                continue
            if len(layer_dists) <= li:
                layer_dists.append([])
            layer_dists[li].append(d)

        total_positions += x.shape[0] * seq_len

        if (b_idx + 1) % max(1, num_batches // 10) == 0:
            elapsed = time.time() - t0
            tok_s = total_positions / max(elapsed, 1e-6)
            print(f"  [{b_idx + 1}/{num_batches}]  {total_positions} tokens  "
                  f"({tok_s:.0f} tok/s)")

    # ── aggregate ────────────────────────────────────────────────────
    n_layers = len(layer_dists)
    dist_means = np.zeros((n_layers, num_heads))
    dist_stds = np.zeros((n_layers, num_heads))

    for li in range(n_layers):
        if not layer_dists[li]:
            continue
        cat = torch.cat(layer_dists[li], dim=0)  # (total_B, H, N)
        dist_means[li] = cat.mean(dim=(0, 2)).numpy()
        dist_stds[li] = cat.std(dim=(0, 2)).numpy()

    # ── cache ────────────────────────────────────────────────────────
    cache_path_out = os.path.join(CACHE_PATH, config.cache_file_name)
    os.makedirs(os.path.dirname(cache_path_out), exist_ok=True)
    torch.save({
        "means": dist_means,
        "stds": dist_stds,
        "model_name": model_name,
        "num_heads": num_heads,
        "num_layers": n_layers,
        "config": config,
    }, cache_path_out)
    print(f"\nRaw data cached to {cache_path_out}")

    # quick summary
    layer_means = dist_means.mean(axis=1)
    print(f"\n  Layer  |  Mean Dist  |  Head Range")
    print("-" * 35)
    for li in range(n_layers):
        rng = f"{dist_means[li].min():.0f}–{dist_means[li].max():.0f}"
        print(f"    {li:2d}   |   {layer_means[li]:6.1f}   |  {rng}")

    # ── plot ─────────────────────────────────────────────────────────
    plot_distance_metric(model_name=model_name, cache_file=config.cache_file_name)


if __name__ == "__main__":
    config = DistanceMetricConfig()
    run_distance_metric(config, force_rerun=True)