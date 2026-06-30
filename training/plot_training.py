import os
import torch
import matplotlib.pyplot as plt
from training.training_config import LanguageModelingExperimentConfig

def plot_training_metrics(config: LanguageModelingExperimentConfig):
    model_name = config.model_name
    checkpoint_path = f"data/training/{model_name}/checkpoint.pt"
    plot_dir = f"data/plots/training/{model_name}"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found at {checkpoint_path}")
        return
        
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Loading metrics from {checkpoint_path}...")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metrics = ckpt.get('metrics', None)
        
        if metrics is None or len(metrics.get('processed_tokens', [])) == 0:
            print("No metrics found in checkpoint or metrics are empty.")
            return
            
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
        
    tokens = metrics['processed_tokens']
    losses = metrics['loss']
    lrs = metrics['learning_rate']
    
    # 1. Plot Loss vs Tokens
    plt.figure(figsize=(10, 6))
    plt.plot(tokens, losses, alpha=0.8, color='blue', label='Training Loss')
    
    # Optional: Plot a smoothed trendline
    if len(losses) > 100:
        smoothed = [sum(losses[i:i+50])/50 for i in range(len(losses)-50)]
        plt.plot(tokens[25:-25], smoothed, color='red', linewidth=2, label='Smoothed (window=50)')
        
    plt.title(f"{model_name} HOFA: Loss vs. Tokens")
    plt.xlabel("Processed Tokens")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    loss_plot_path = os.path.join(plot_dir, "loss_vs_tokens.pdf")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved Loss plot to {loss_plot_path}")
    
    # 2. Plot Learning Rate vs Tokens
    plt.figure(figsize=(10, 6))
    plt.plot(tokens, lrs, color='orange', linewidth=2)
    plt.title(f"{model_name} HOFA: Learning Rate Schedule")
    plt.xlabel("Processed Tokens")
    plt.ylabel("Learning Rate")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    lr_plot_path = os.path.join(plot_dir, "lr_vs_tokens.pdf")
    plt.savefig(lr_plot_path)
    plt.close()
    print(f"Saved LR schedule plot to {lr_plot_path}")

if __name__ == "__main__":
    config = LanguageModelingExperimentConfig()
    plot_training_metrics(config)