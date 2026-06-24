import os
import torch
import matplotlib.pyplot as plt

def plot_unified_trendline():
    seq_lengths = [1024, 512, 256, 128, 64]
    
    models_to_test = [
        "MHA",
        "HOFA (r=8)",
        "HOFA (r=14)",
        "HOFA (r=16)",
        "Gated DeltaNet",
        "GLA",
        "Mamba"
    ]
    
    # name matching with directory mapping
    dir_mapping = {
        "MHA": "induction_MHA",
        "HOFA (r=8)": "induction_HOFA_r8",
        "HOFA (r=14)": "induction_HOFA_r14",
        "HOFA (r=16)": "induction_HOFA_r16",
        "Gated DeltaNet": "induction_Gated_DeltaNet",
        "GLA": "induction_GLA",
        "Mamba": "induction_Mamba"
    }

    results = {name: [] for name in models_to_test}

    for seq_len in seq_lengths:
        for name in models_to_test:
            dir_name = dir_mapping[name]
            ckpt_path = f"data/models/seqlen_{seq_len}/{dir_name}/checkpoint.pt"
            
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
    
    png_path = plot_path.replace('.pdf', '.png')
    plt.savefig(png_path, bbox_inches='tight', format='png', dpi=300)
    plt.close()
    
    print(f"\nUnified trendline plot successfully saved to {plot_path}")

if __name__ == "__main__":
    plot_unified_trendline()
