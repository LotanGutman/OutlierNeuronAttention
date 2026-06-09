"""
Fidelity vs. r, showing the RFF Approximation Gap for HOFA.
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
from src.triton_model import HybridOutlierFactorizedAttention as TritonHOFA
from src.torch_model import HybridOutlierFactorizedAttention as TorchHOFA
from src.config import ModelConfig, TrainingConfig

def cosine_similarity(out1, out2):
    return F.cosine_similarity(out1, out2, dim=-1).mean().item()

def run_fidelity_experiment():
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    device = train_cfg.device
    torch.manual_seed(train_cfg.seed)

    d_head = model_cfg.d_head
    r_values = [0, 1, 2, 4, 8, 16, 24, 32, round((32+d_head)/2), d_head]
    
    # Freeze Sequence Length
    N = 4096
    
    # Build a template attention module with exact softmax (r = d_head)
    # The exact ideal is theoretically bounded by O(N^2) TorchHOFA with r=d_head
    template_cfg = copy.deepcopy(model_cfg)
    template_cfg.r = d_head
    template = TorchHOFA(template_cfg).to(device).eval()

    # Store the template's weights
    Wq = template.W_q.weight.clone()
    Wk = template.W_k.weight.clone()
    Wv = template.W_v.weight.clone()
    Wout = template.out_proj.weight.clone()

    # Styling
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

    fig, ax = plt.subplots(figsize=(10, 6))

    x = torch.randn(1, N, model_cfg.d_model, device=device)

    # Compute exact output using the dense template (r = d_head → exact softmax)
    with torch.no_grad():
        exact_out = template(x)

    sims_dense = []
    sims_triton = []

    for r in r_values:
        # --- Torch HOFA (Dense / Exact Inlier) ---
        cfg_dense = copy.deepcopy(model_cfg)
        cfg_dense.r = r
        dense_model = TorchHOFA(cfg_dense).to(device).eval()
        dense_model.W_q.weight = torch.nn.Parameter(Wq.clone())
        dense_model.W_k.weight = torch.nn.Parameter(Wk.clone())
        dense_model.W_v.weight = torch.nn.Parameter(Wv.clone())
        dense_model.out_proj.weight = torch.nn.Parameter(Wout.clone())
        
        if r > 0:
            dense_model._maybe_update_indices()

        # --- Triton HOFA (m_O = 64) ---
        cfg_64 = copy.deepcopy(model_cfg)
        cfg_64.r = r
        cfg_64.m_O = 64
        triton_64 = TritonHOFA(cfg_64).to(device).eval()
        triton_64.W_q.weight = torch.nn.Parameter(Wq.clone())
        triton_64.W_k.weight = torch.nn.Parameter(Wk.clone())
        triton_64.W_v.weight = torch.nn.Parameter(Wv.clone())
        triton_64.out_proj.weight = torch.nn.Parameter(Wout.clone())

        if r > 0:
            triton_64._maybe_update_indices()



        with torch.no_grad():
            out_dense = dense_model(x)
            out_64 = triton_64(x)

        sim_dense = cosine_similarity(out_dense, exact_out)
        sim_64 = cosine_similarity(out_64, exact_out)
        
        sims_dense.append(sim_dense)
        sims_triton.append(sim_64)
        print(f"r={r:2d}: Dense={sim_dense:.6f}, Triton(m_O=64)={sim_64:.6f}")

    # Plot Dense
    ax.plot(r_values, sims_dense, color='black', linewidth=3, linestyle='-', label='HOFA-Dense (Exact Inlier)')
    # Plot Triton m_O=64
    ax.plot(r_values, sims_triton, color='blue', linewidth=2, linestyle='--', label=r'HOFA-Chunked ($m_O=64$)')

    # Add vertical lines and labels for key models
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.text(0, 0.46, ' Linear (r=0)', rotation=90, va='bottom', ha='left', fontsize=10, color='gray')
    
    ax.axvline(x=8, color='gray', linestyle=':', alpha=0.5)
    ax.text(8, 0.46, ' HOFA (r=8)', rotation=90, va='bottom', ha='left', fontsize=10, color='gray')
    
    ax.axvline(x=d_head, color='gray', linestyle=':', alpha=0.5)
    ax.text(d_head, 0.46, f' MHA (r={d_head})', rotation=90, va='bottom', ha='left', fontsize=10, color='gray')

    # ---- Symlog scale on x-axis to spread out the r=0→1 jump ----
    ax.set_xscale('symlog', linthresh=1, linscale=0.5)
    ax.set_xlabel('Number of outlier dimensions ($r$)', fontsize=13)
    ax.set_ylabel('Cosine similarity to exact softmax', fontsize=13)
    ax.set_title('RFF Approximation Gap in Hybrid Outlier-Factorized Attention (N=4096)')

    print(f'RFF Approximation Gap '
          f'($d_{{model}}$={model_cfg.d_model}, heads={model_cfg.num_heads}, '
          f'$d_{{head}}$={d_head}, N={N})')

    # Keep custom tick locations and labels
    ax.set_xticks(r_values)
    ax.set_xticklabels(r_values)
    ax.grid(True, linestyle=':', alpha=0.4)
    
    ax.legend(frameon=True, fancybox=False, edgecolor='gray', loc='lower right', fontsize=11)
    ax.set_ylim(0.45, 1.05)
    fig.tight_layout()

    os.makedirs('benchmarks/plots', exist_ok=True)
    fig.savefig('benchmarks/plots/fidelity.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plots saved to benchmarks/plots/")

if __name__ == '__main__':
    run_fidelity_experiment()