import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def plot_unified_trendline():
    from benchmarks.benchmarks_configs import InductionExperimentConfig
    from src.modules.benchmark_utils import GenericBenchmarkLM
    from benchmarks.benchmark_induction import evaluate_induction_accuracy

    config = InductionExperimentConfig()
    device = config.device
    seq_lengths = [1024, 512, 256, 128, 64]
    
    models_to_test = config.models_to_test
    
    cache_dir = "data/experiments_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "regular_induction_trendline_cache.pt")
    
    cached_results = {}
    if os.path.exists(cache_path):
        try:
            cached_results = torch.load(cache_path, weights_only=False)
        except Exception:
            cached_results = {}
            
    results = {name: [] for name, _, _ in models_to_test}
    updated_cache = False
    
    print("\nEvaluating held-out inference accuracy for regular induction scaling trendline...")
    
    for seq_len in seq_lengths:
        for name, attn_type, r_val in models_to_test:
            cache_key = f"{seq_len}_{name}"
            if cache_key in cached_results:
                results[name].append(cached_results[cache_key])
                continue
                
            dir_name = f"induction_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}"
            ckpt_dir = f"data/induction_models/seqlen_{seq_len}/{dir_name}"
            final_model_path = f"{ckpt_dir}/final_model.pt"
            ckpt_path = f"{ckpt_dir}/checkpoint.pt"
            target_path = final_model_path if os.path.exists(final_model_path) else ckpt_path
            
            if os.path.exists(target_path):
                ckpt = torch.load(target_path, map_location=device, weights_only=False)
                sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
                
                config.model_config.r = r_val if r_val is not None else 8
                model = GenericBenchmarkLM(
                    vocab_size=config.vocab_size,
                    d_model=config.model_config.d_model,
                    attn_type=attn_type,
                    num_heads=config.model_config.num_heads,
                    num_layers=config.model_config.num_layers,
                    model_cfg=config.model_config
                ).to(device)
                
                model.load_state_dict(sd)
                model.eval()
                
                metrics = evaluate_induction_accuracy(
                    model, config, seq_len=seq_len, eval_samples=500, eval_batch_size=32,
                    device=device, pattern_len=None, pin_to_end=False, seed=config.seed + 777
                )
                final_acc = metrics['avg_token_acc']
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                final_acc = 0.0
                
            results[name].append(final_acc)
            cached_results[cache_key] = final_acc
            updated_cache = True
            
    if updated_cache:
        torch.save(cached_results, cache_path)

    os.makedirs("data/plots/induction", exist_ok=True)
    
    # Set publication rcParams matching LaTeX scaling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 13,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 9.5,
        'figure.autolayout': True
    })
    
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    
    for name, accuracies in results.items():
        ax.plot(seq_lengths[::-1], accuracies[::-1], label=name, marker='o', markersize=5, linewidth=2)
    
    ax.set_title("Sequence Length Scaling Trendline", pad=10)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Accuracy (%)")
    
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lengths[::-1])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    ax.set_ylim(-5, 105)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9, prop={'size': 8.5})
    ax.grid(True, alpha=0.3)
    
    plot_path = "data/plots/induction/unified_seqlen_trendline.pdf"
    plt.savefig(plot_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.close()
    
    print(f"\nUnified trendline plot successfully saved to {plot_path}")

def plot_feature_norm_disparity():
    # 1. Load Checkpoints
    mha_path = "data/induction_models/seqlen_1024/induction_MHA/checkpoint.pt"
    hofa8_path = "data/induction_models/seqlen_1024/induction_HOFA_r8/checkpoint.pt"
    hofa16_path = "data/induction_models/seqlen_1024/induction_HOFA_r16/checkpoint.pt"
    
    def get_feature_importance(ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f"Missing {ckpt_path}")
            return np.zeros(32)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['model_state_dict']
        W_q = state_dict['blocks.3.attn.W_q.weight'].view(4, 32, 128)
        W_k = state_dict['blocks.3.attn.W_k.weight'].view(4, 32, 128)
        return (W_q.norm(p=2, dim=2) * W_k.norm(p=2, dim=2)).mean(dim=0).numpy()

    mha_imp = get_feature_importance(mha_path)
    hofa8_imp = get_feature_importance(hofa8_path)
    hofa16_imp = get_feature_importance(hofa16_path)

    # 2. Plotting
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 13,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    dims = np.arange(32)
    
    def plot_step(ax, data, color, title, r_val=None):
        ax.fill_between(dims, data, step="mid", color=color, alpha=0.3)
        ax.plot(dims, data, drawstyle="steps-mid", color=color, linewidth=2.5)
        ax.set_title(title, pad=10)
        ax.set_xlabel("Feature Dimension Index")
        ax.set_xlim(0, 31)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        if r_val:
            ax.axvline(x=r_val - 0.5, color='red', linestyle='--', linewidth=2, label=f'Hardware Bound (r={r_val})')
            ax.legend(loc='upper right', frameon=True)

    plot_step(axes[0], mha_imp, '#4c72b0', "MHA (Baseline)")
    plot_step(axes[1], hofa8_imp, '#dd8452', "HOFA (r=8)", r_val=8)
    plot_step(axes[2], hofa16_imp, '#55a868', "HOFA (r=16)", r_val=16)
    
    axes[0].set_ylabel("Product Norm ($||W_Q||_2 \\times ||W_K||_2$)")
    
    os.makedirs("data/plots/routing", exist_ok=True)
    plt.tight_layout()
    plt.savefig("data/plots/routing/feature_norm_disparity.pdf", bbox_inches='tight', format='pdf')
    plt.close()
    
    print("\nFeature norm disparity plot successfully saved to data/plots/routing/feature_norm_disparity.pdf")

if __name__ == "__main__":
    plot_unified_trendline()
    # plot_feature_norm_disparity()
