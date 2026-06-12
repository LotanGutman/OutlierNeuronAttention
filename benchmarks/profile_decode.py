import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from benchmarks.benchmarks_configs import CACHE_PATH, DecodeExperimentConfig
from src.config import ModelConfig
from matplotlib.ticker import FuncFormatter

# 1. Force PyTorch to fuse the operations
import torch._dynamo
torch._dynamo.config.suppress_errors = True

@torch.compile(mode="reduce-overhead", fullgraph=True)
def hofa_decode_step(q_O, k_O_cache, v_cache, q_J, k_J, v_J, state_I, gamma, seq_idx):
    # Slice view (Fast in compiled graph)
    k_past = k_O_cache[:, :, :seq_idx+1, :]
    v_past = v_cache[:, :, :seq_idx+1, :]
    
    # Math
    Y_O = F.scaled_dot_product_attention(q_O, k_past, v_past, is_causal=False)
    state_I_new = gamma * state_I + k_J.unsqueeze(-1) * v_J.unsqueeze(-2)
    Y_I = (q_J.unsqueeze(-1) * state_I_new).sum(dim=2)
    return Y_O, Y_I, state_I_new

@torch.compile(mode="reduce-overhead", fullgraph=True)
def mha_decode_step(q, k_cache, v_cache, seq_idx):
    k_past = k_cache[:, :, :seq_idx+1, :]
    v_past = v_cache[:, :, :seq_idx+1, :]
    return F.scaled_dot_product_attention(q, k_past, v_past, is_causal=False)

def run_decode_profiling(config: DecodeExperimentConfig = DecodeExperimentConfig(), force_rerun=True, save_results=True):
    model_cfg = config.model_config
    device = torch.device(config.device)
    cache_path = os.path.join(CACHE_PATH, config.cache_file_name)
    
    seq_lengths = config.seq_lengths
    
    if os.path.exists(cache_path) and not force_rerun:
        print(f"Loading cached results from {cache_path}")
        data = torch.load(cache_path)
        valid_lens, times_mha, times_hyb, cache_mha_mb, cache_hyb_mb = data
    else:
        times_mha, times_hyb = [], []
        cache_mha_mb, cache_hyb_mb = [], []
        valid_lens = []

        print(f"{'Context':<8} | {'MHA Tok/s':<10} | {'MHA Cache':<10} | {'HOFA Tok/s':<10} | {'HOFA Cache':<10}")
        print("-" * 65)

        # Batch size is always 1 for decoding profiling
        B = 1
    H = model_cfg.num_heads
    D_head = model_cfg.d_head
    r = model_cfg.r
    j = D_head - r

    for sl in seq_lengths:
        torch.cuda.empty_cache()
        
        mha_gb = (2 * (B * H * sl * D_head) * 2) / (1024**3)
        hyb_gb = ((B * H * sl * r * 2) + (B * H * sl * D_head * 2) + (B * H * j * D_head * 4)) / (1024**3)
        
        try:
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                q_mha = torch.randn(B, H, 1, D_head, device=device, dtype=torch.bfloat16)
                k_mha_cache = torch.randn(B, H, sl, D_head, device=device, dtype=torch.bfloat16)
                v_mha_cache = torch.randn(B, H, sl, D_head, device=device, dtype=torch.bfloat16)
                seq_idx = sl - 1

                for _ in range(config.warmup_steps):
                    _ = mha_decode_step(q_mha, k_mha_cache, v_mha_cache, seq_idx)
                torch.cuda.synchronize()

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(config.active_steps):
                    _ = mha_decode_step(q_mha, k_mha_cache, v_mha_cache, seq_idx)
                end.record()
                torch.cuda.synchronize()
                
                # Multiply by B to get total tokens per second
                tok_s_mha = (B * 1000.0) / (start.elapsed_time(end) / config.active_steps) 
                del q_mha, k_mha_cache, v_mha_cache
        except torch.cuda.OutOfMemoryError:
            tok_s_mha = float('nan')

        try:
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                q_O = torch.randn(B, H, 1, r, device=device, dtype=torch.bfloat16)
                k_O_cache = torch.randn(B, H, sl, r, device=device, dtype=torch.bfloat16)
                v_cache = torch.randn(B, H, sl, D_head, device=device, dtype=torch.bfloat16)
                
                q_J = torch.randn(B, H, j, device=device, dtype=torch.float32)
                k_J = torch.randn(B, H, j, device=device, dtype=torch.float32)
                v_J = torch.randn(B, H, D_head, device=device, dtype=torch.float32)
                state_I = torch.randn(B, H, j, D_head, device=device, dtype=torch.float32)
                gamma = torch.randn(B, H, 1, 1, device=device, dtype=torch.float32)

                for _ in range(config.warmup_steps):
                    _ = hofa_decode_step(q_O, k_O_cache, v_cache, q_J, k_J, v_J, state_I, gamma, seq_idx)
                torch.cuda.synchronize()

                start.record()
                for _ in range(config.active_steps):
                    _ = hofa_decode_step(q_O, k_O_cache, v_cache, q_J, k_J, v_J, state_I, gamma, seq_idx)
                end.record()
                torch.cuda.synchronize()
                
                tok_s_hyb = (B * 1000.0) / (start.elapsed_time(end) / config.active_steps)
                del q_O, k_O_cache, v_cache, q_J, k_J, v_J, state_I
        except torch.cuda.OutOfMemoryError:
            tok_s_hyb = float('nan')

        times_mha.append(tok_s_mha)
        times_hyb.append(tok_s_hyb)
        cache_mha_mb.append(mha_gb)
        cache_hyb_mb.append(hyb_gb)
        valid_lens.append(sl)

        t_mha_str = f"{tok_s_mha:.1f}" if not np.isnan(tok_s_mha) else "OOM"
        t_hyb_str = f"{tok_s_hyb:.1f}" if not np.isnan(tok_s_hyb) else "OOM"
        print(f"{sl:<8d} | {t_mha_str:<10} | {mha_gb:<10.2f} | {t_hyb_str:<10} | {hyb_gb:<10.2f}")
                
        if save_results:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save((valid_lens, times_mha, times_hyb, cache_mha_mb, cache_hyb_mb), cache_path)
            
    plot_decode_results(valid_lens, times_mha, times_hyb, cache_mha_mb, cache_hyb_mb)

def plot_decode_results(lens, t_mha, t_hyb, c_mha, c_hyb):
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    formatter_x = FuncFormatter(lambda x, pos: f'{int(x/1024)}k')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    valid_mha_lens = [l for i, l in enumerate(lens) if not np.isnan(t_mha[i])]
    valid_mha_times = [t for t in t_mha if not np.isnan(t)]
    valid_hyb_lens = [l for i, l in enumerate(lens) if not np.isnan(t_hyb[i])]
    valid_hyb_times = [t for t in t_hyb if not np.isnan(t)]
    
    ax1.plot(valid_mha_lens, valid_mha_times, marker='o', color='#D55E00', lw=2.5, label='MHA')
    ax1.plot(valid_hyb_lens, valid_hyb_times, marker='s', color='#0072B2', lw=2.5, label='HOFA (Ours)')
    
    ax1.set_xscale('log', base=2)
    ax1.xaxis.set_major_formatter(formatter_x)
    ax1.set_xticks(lens)
    ax1.set_xlabel('Context Length ($N$)')
    ax1.set_ylabel('Tokens Per Second')
    ax1.set_title('Decoding Throughput')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    ax2.plot(lens, c_mha, marker='o', color='#D55E00', lw=2.5, label='MHA')
    ax2.plot(lens, c_hyb, marker='s', color='#0072B2', lw=2.5, label='HOFA (Ours)')
    
    ax2.set_xscale('log', base=2)
    ax2.xaxis.set_major_formatter(formatter_x)
    ax2.set_xticks(lens)
    ax2.set_xlabel('Context Length ($N$)')
    ax2.set_ylabel('KV Cache Size (GB)')
    ax2.set_title('KV Cache Footprint')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    os.makedirs('data/plots', exist_ok=True)
    plt.savefig('data/plots/profile_decode.pdf', bbox_inches='tight')
    print("Saved plot to data/plots/profile_decode.pdf")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--override', action='store_true', help='Force rerun instead of loading cache')
    args = parser.parse_args()
    
    run_decode_profiling(force_rerun=args.override)
