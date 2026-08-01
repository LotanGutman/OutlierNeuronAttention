import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def plot_unified_trendline():
    seq_lengths = [1024, 512, 256, 128, 64]
    
    models_to_test = [
        "MHA",
        "HOFA (r=8)",
        "HOFA (r=10)",
        "HOFA (r=16)",
        "Gated DeltaNet",
        "GLA",
        "Mamba"
    ]
    
    # name matching with directory mapping
    dir_mapping = {
        "MHA": "induction_MHA",
        "HOFA (r=8)": "induction_HOFA_r8",
        "HOFA (r=10)": "induction_HOFA_r10",
        "HOFA (r=16)": "induction_HOFA_r16",
        "Gated DeltaNet": "induction_Gated_DeltaNet",
        "GLA": "induction_GLA",
        "Mamba": "induction_Mamba"
    }

    results = {name: [] for name in models_to_test}

    for seq_len in seq_lengths:
        for name in models_to_test:
            dir_name = dir_mapping[name]
            ckpt_path = f"data/induction_models/seqlen_{seq_len}/{dir_name}/checkpoint.pt"
            
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location='cpu')
                history = ckpt.get('metadata', {}).get('history', {})
                accs = history.get('acc', [])
                if len(accs) > 0:
                    final_acc = max(accs)
                else:
                    final_acc = 0.0
            else:
                final_acc = 0.0
                
            results[name].append(final_acc)

    os.makedirs("data/plots/induction", exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    for name, accuracies in results.items():
        # Plotting correctly left to right (64 to 1024)
        plt.plot(seq_lengths[::-1], accuracies[::-1], label=name, marker='o', markersize=6, linewidth=2)
    
    plt.title("Sequence Length Scaling Trendline (Induction Head)")
    plt.xlabel("Sequence Length")
    plt.ylabel("Final Max Accuracy (%)")
    
    # Format x-axis nicely
    plt.xscale('log', base=2)
    plt.xticks(seq_lengths[::-1], seq_lengths[::-1])
    
    plt.ylim(-5, 105)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
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
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    dims = np.arange(32)
    
    def plot_step(ax, data, color, title, r_val=None):
        ax.fill_between(dims, data, step="mid", color=color, alpha=0.3)
        ax.plot(dims, data, drawstyle="steps-mid", color=color, linewidth=2.5)
        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_xlabel("Feature Dimension Index", fontsize=11)
        ax.set_xlim(0, 31)
        if r_val:
            ax.axvline(x=r_val - 0.5, color='red', linestyle='--', linewidth=2, label=f'Hardware Bound (r={r_val})')
            ax.legend(loc='upper right', frameon=True)

    plot_step(axes[0], mha_imp, '#4c72b0', "MHA (Baseline)")
    plot_step(axes[1], hofa8_imp, '#dd8452', "HOFA (r=8)", r_val=8)
    plot_step(axes[2], hofa16_imp, '#55a868', "HOFA (r=16)", r_val=16)
    
    axes[0].set_ylabel("Product Norm ($||W_Q||_2 \\times ||W_K||_2$)", fontsize=11)
    
    os.makedirs("data/plots/routing", exist_ok=True)
    plt.tight_layout()
    plt.savefig("data/plots/routing/feature_norm_disparity.pdf", bbox_inches='tight', format='pdf')
    plt.close()
    
    print("\nFeature norm disparity plot successfully saved to data/plots/routing/feature_norm_disparity.pdf")

if __name__ == "__main__":
    plot_unified_trendline()
    # plot_feature_norm_disparity()
