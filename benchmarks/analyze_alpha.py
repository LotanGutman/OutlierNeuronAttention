import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from training.data_utils import FastTokenLoader
from src.HybridOutlierFactorizedAttention import SubwordLM
from src.modules.benchmark_utils import patch_attention_for_debugging
from training.training_config import LanguageModelingExperimentConfig

def run_alpha_analysis(config: LanguageModelingExperimentConfig):
    device = torch.device(config.device)
    
    # Resolve Checkpoint
    checkpoint_dir = f"data/training/{config.model_name}"
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint_best_val.pt")
    
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            print(f"Error: Could not find checkpoint at {checkpoint_dir}")
            return
        print("Could not find best val checkpoint, loading latest")
    print(f"Loading model {config.model_name} from {ckpt_path}")
    
    # Initialize Model
    model = SubwordLM(config.vocab_size, config.model_config)
    model.to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Patch layers to intercept _last_mix_g
    for layer in model.layers:
        patch_attention_for_debugging(layer.attn)
        
    # Setup Dataloader
    model_size = config.model_name.split('_')[0]
    val_cache_path = f"data/datasets/data_{model_size}_val_cache.bin"
    
    if not os.path.exists(val_cache_path):
        print(f"Error: Validation cache not found at {val_cache_path}. Run download-data first.")
        return
        
    batch_size = 10
    seq_len = 1024
    num_batches = 10
    
    val_loader = FastTokenLoader(val_cache_path, batch_size, seq_len, 0)
    
    num_layers = len(model.layers)
    layer_gates = [[] for _ in range(num_layers)]
    
    print(f"Running forward passes over {num_batches} batches of size {batch_size}x{seq_len}...")
    
    with torch.no_grad():
        for i in range(num_batches):
            x, _, _ = val_loader.get_batch()
            x = x.to(device)
            
            # Forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = model(x)
                
            # Extract gates grouped by layer
            for l, layer in enumerate(model.layers):
                if hasattr(layer.attn, '_last_mix_g'):
                    layer_gates[l].append(layer.attn._last_mix_g.float().cpu().numpy().flatten())
                    
    if not any(layer_gates):
        print("Error: No gates were extracted. Is this a HOFA model?")
        return
        
    plot_data = [np.concatenate(gates) for gates in layer_gates]
    
    # Plot
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 13,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 11
    })

    fig_width = max(8.5, min(14.0, num_layers * 0.45))
    plt.figure(figsize=(fig_width, 4.5))
    parts = plt.violinplot(plot_data, positions=range(num_layers), showmeans=True, showextrema=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#4c72b0')
        pc.set_edgecolor('#1f4e79')
        pc.set_alpha(0.65)
        
    parts['cmeans'].set_color('#d62728')
    parts['cmeans'].set_linewidth(1.8)
    parts['cmins'].set_color('#1f4e79')
    parts['cmaxes'].set_color('#1f4e79')
    parts['cbars'].set_color('#1f4e79')
    
    plt.title(f"Mixing Gate $\\alpha_h$ Distribution per Layer ({config.plot_name})", pad=10)
    plt.xlabel("Layer Index")
    plt.ylabel("Mixing Gate Value ($\\alpha$)")
    plt.xticks(range(num_layers), [f"L{i}" for i in range(num_layers)], rotation=0 if num_layers <= 12 else 45)
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    
    out_dir = "data/plots/alpha_analysis"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{config.model_name}_alpha_violin.pdf")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"\nViolin plot saved successfully to {out_path}")
