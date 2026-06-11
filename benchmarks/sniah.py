import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import gc

from benchmarks.synthetic_recall import generate_sniah, GenericBenchmarkLM, AttentionType, adjust_learning_rate
from benchmarks.benchmarks_configs import RecallExperimentConfig, CACHE_PATH

def train_and_eval_sniah(model, seq_len, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    device = config.device
    use_autocast = config.use_mixed_precision and device == "cuda"
    warmup_steps = int(0.1 * config.train_steps)

    model.train()
    for i in range(config.train_steps):
        adjust_learning_rate(optimizer, i, config.train_steps, config.learning_rate, warmup_steps)
        optimizer.zero_grad()
        
        # random depth for training
        depth_pct = torch.rand(1).item()
        x, y = generate_sniah(config.batch_size, seq_len, config.vocab_size, depth_pct, device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            _, loss = model(x, targets=y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        if (i + 1) % config.print_every == 0:
            print(f"      Step {i + 1:4d}/{config.train_steps} | Train Loss: {loss.item():.4f}")
            # SNIAH converges very fast, early stop if loss is near zero
            if loss.item() < 0.001:
                print("      Early stopping: loss < 0.001")
                break

    # Evaluate on fixed depths
    model.eval()
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {}
    with torch.no_grad():
        for d in depths:
            correct = 0
            total = 0
            for _ in range(config.num_eval_batches):
                x, y = generate_sniah(config.batch_size, seq_len, config.vocab_size, d, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                    logits_valid, _ = model(x, targets=y, return_loss=False)

                preds = logits_valid.argmax(dim=-1)
                targets_valid = y[y != -100]
                correct += (preds == targets_valid).sum().item()
                total += targets_valid.numel()
            results[d] = correct / total
            print(f"      Eval depth {d*100:.0f}%: {results[d]*100:.1f}%")
            
    return results

def plot_sniah_heatmaps(all_results, models, seq_lens, depths, save_plot=True):
    os.makedirs("data/plots", exist_ok=True)
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]
        
    for i, model_name in enumerate(models):
        heatmap = np.zeros((len(seq_lens), len(depths)))
        for j, sl in enumerate(seq_lens):
            for k, d in enumerate(depths):
                heatmap[j, k] = all_results[model_name][sl][d] * 100
                
        ax = axes[i]
        cax = ax.matshow(heatmap, cmap='viridis', vmin=0, vmax=100)
        ax.set_title(f"{model_name}")
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([f"{d*100:.0f}%" for d in depths])
        ax.set_yticks(range(len(seq_lens)))
        ax.set_yticklabels([f"{sl//1000}k" for sl in seq_lens])
        ax.set_xlabel("Needle Depth")
        ax.set_ylabel("Context Length")
        
        for j in range(len(seq_lens)):
            for k in range(len(depths)):
                ax.text(k, j, f"{heatmap[j, k]:.0f}%", ha='center', va='center', 
                        color="black" if heatmap[j, k] > 50 else "white")

    fig.colorbar(cax, ax=axes, fraction=0.02, pad=0.04)
    if save_plot:
        plot_path = "data/plots/sniah_heatmap.png"
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
    plt.close()

def run_sniah_experiment():
    config = RecallExperimentConfig()
    # Shorter train steps for SNIAH, it converges instantly
    config.train_steps = 1000 
    config.print_every = 250
    config.batch_size = 4 # Small batch size to fit long contexts in 8GB
    
    device = config.device
    torch.manual_seed(config.seed)
    
    seq_lens = [8192, 16384, 32768, 65536, 131072] # 256k and 512k might OOM
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    model_names = ["HOFA (r=8)", "MHA", "Gated DeltaNet"]
    
    attn_type_map = {
        "MHA": AttentionType.MHA,
        "HOFA (r=8)": AttentionType.HOFA,
        "Gated DeltaNet": AttentionType.DELTA
    }
    
    all_results = {m: {sl: {} for sl in seq_lens} for m in model_names}
    
    for name in model_names:
        for sl in seq_lens:
            print(f"\nTraining {name} on SNIAH (Seq Len: {sl})...")
            model_cfg = config.model_config
            attn_type = attn_type_map[name]
            
            model = GenericBenchmarkLM(
                config.vocab_size, 
                model_cfg.d_model, 
                attn_type, 
                num_heads=model_cfg.num_heads, 
                num_layers=model_cfg.num_layers, 
                model_cfg=model_cfg
            ).to(device)
            
            results = train_and_eval_sniah(model, sl, config)
            all_results[name][sl] = results
            
            model.to('cpu')
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
    plot_sniah_heatmaps(all_results, model_names, seq_lens, depths)

if __name__ == "__main__":
    run_sniah_experiment()
