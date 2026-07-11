import os
import math
import torch
import matplotlib.pyplot as plt
from training.training_config import LanguageModelingExperimentConfig


def moving_average(values, window=50):
    """Simple moving average."""
    if len(values) < window:
        return None
    return [
        sum(values[i:i + window]) / window
        for i in range(len(values) - window + 1)
    ]


from src.HybridOutlierFactorizedAttentionTrain import SubwordLM

def get_active_non_embedding_parameters(model):
    return sum(p.numel() for n, p in model.named_parameters() if 'token_emb' not in n and 'lm_head' not in n)

def calc_hofa_flops_per_token(P, H, d_h, r_list, N):
    dense_flops = 6 * P
    outlier_flops = 0
    inlier_flops = 0
    for r in r_list:
        outlier_flops += 12 * H * r * N
        inlier_flops += 18 * H * (d_h - r) * d_h
    return dense_flops + outlier_flops + inlier_flops

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
        metrics = ckpt.get("metrics", None)

        if metrics is None or len(metrics.get("processed_tokens", [])) == 0:
            print("No metrics found in checkpoint or metrics are empty.")
            return

    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # --- Extract training data ---
    tokens = metrics["processed_tokens"]
    losses = metrics["loss"]

    # --- Extract validation data ---
    val_tokens = metrics.get("val_tokens", [])
    val_ppl = metrics.get("val_ppl", [])

    # Drop the first validation point (often noisy / JIT artifact)
    if len(val_tokens) > 1:
        val_tokens = val_tokens[1:]
        val_ppl = val_ppl[1:]

    has_val = len(val_tokens) > 0 and len(val_ppl) > 0

    # Compute validation CE loss from perplexity: CE = log(PPL)
    if has_val:
        val_loss = [math.log(p) for p in val_ppl]

    # Calculate FLOPs per token using HOFA exact mathematical formula
    model = SubwordLM(config.vocab_size, config.model_config)
    P = get_active_non_embedding_parameters(model)
    H = config.model_config.num_heads
    d_h = config.model_config.d_model // H
    N_seq = config.seq_len
    r_list = config.model_config.r
    if not isinstance(r_list, (list, tuple)):
        r_list = [r_list] * config.model_config.num_layers
        
    flops_per_token = calc_hofa_flops_per_token(P, H, d_h, r_list, N_seq)
    print(f"Calculated Theoretical FLOPs per token: {flops_per_token:.2e}")
    
    flops_x = [t * flops_per_token for t in tokens]
    if has_val:
        val_flops_x = [t * flops_per_token for t in val_tokens]

    # ------------------------------------------------------------------
    # Compute smoothed training loss (for Plot 1 only)
    # ------------------------------------------------------------------
    smooth_window = 150
    smoothed = moving_average(losses, smooth_window)

    if smoothed is not None:
        offset = smooth_window // 2
        smooth_tokens = tokens[offset:offset + len(smoothed)]
    else:
        smooth_tokens = None

    # ------------------------------------------------------------------
    # Plot 1: Training Loss vs Tokens (Log X-axis) – RAW + SMOOTHED
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Raw training loss
    plt.plot(
        tokens,
        losses,
        color="blue",
        alpha=0.6,
        linewidth=0.8,
        label="Training Loss (raw)",
    )

    # Smoothed training loss
    if smoothed is not None:
        plt.plot(
            smooth_tokens,
            smoothed,
            color="red",
            linewidth=2,
            label=f"Training Loss (smoothed, window={smooth_window})",
        )

    plt.xscale("log")
    plt.ylim(3.0, 6.0)          # Adjust if your losses are outside this range
    plt.xlim(10**7, tokens[-1])  # Start after warmup

    plt.title(f"{model_name} HOFA: Training Loss vs. Tokens (Log X)")
    plt.xlabel("Processed Tokens (log scale)")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plot1_path = os.path.join(plot_dir, "loss_vs_tokens_logx.pdf")
    plt.savefig(plot1_path)
    plt.close()
    print(f"Saved Plot 1 (raw + smoothed loss, log-x) to {plot1_path}")

    # ------------------------------------------------------------------
    # Plot 2: RAW Training Loss + Validation CE Loss (Linear X-axis)
    # ------------------------------------------------------------------
    if has_val:
        plt.figure(figsize=(10, 6))

        # Training loss (raw) – NO smoothing
        plt.plot(
            tokens,
            losses,
            color="red",
            alpha=0.6,
            linewidth=0.8,
            label="Training Loss (raw)",
        )

        # Validation loss (log of perplexity)
        plt.plot(
            val_tokens,
            val_loss,
            color="green",
            marker="o",
            markersize=4,
            linewidth=1.5,
            linestyle="-",
            label="Validation Loss (log PPL)",
        )

        plt.title(f"{model_name} HOFA: Training & Validation Loss (Linear X)")
        plt.xlabel("Processed Tokens")
        plt.ylabel("Cross Entropy Loss")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        plot2_path = os.path.join(plot_dir, "loss_and_val_loss_vs_tokens.pdf")
        plt.savefig(plot2_path)
        plt.close()
        print(f"Saved Plot 2 (raw train loss + val loss) to {plot2_path}")

    # ------------------------------------------------------------------
    # Plot 3: Validation Perplexity only (Linear X-axis)
    # ------------------------------------------------------------------
    if has_val:
        plt.figure(figsize=(10, 6))

        plt.plot(
            val_tokens,
            val_ppl,
            color="green",
            marker="o",
            markersize=4,
            linewidth=1.5,
            linestyle="-",
            label="Validation Perplexity",
        )

        plt.title(f"{model_name} HOFA: Validation Perplexity")
        plt.xlabel("Processed Tokens")
        plt.ylabel("Perplexity")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        plot3_path = os.path.join(plot_dir, "val_ppl_vs_tokens.pdf")
        plt.savefig(plot3_path)
        plt.close()
        print(f"Saved Plot 3 (val PPL) to {plot3_path}")

    # ------------------------------------------------------------------
    # Plot 4: RAW Training Loss + Validation CE Loss (Linear FLOPs X-axis)
    # ------------------------------------------------------------------
    if has_val:
        plt.figure(figsize=(10, 6))

        plt.plot(
            flops_x,
            losses,
            color="red",
            alpha=0.6,
            linewidth=0.8,
            label="Training Loss (raw)",
        )

        plt.plot(
            val_flops_x,
            val_loss,
            color="green",
            marker="o",
            markersize=4,
            linewidth=1.5,
            linestyle="-",
            label="Validation Loss (log PPL)",
        )

        plt.title(f"{model_name} HOFA: Training & Validation Loss (Linear FLOPs)")
        plt.xlabel("Total FLOPs")
        plt.ylabel("Cross Entropy Loss")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        plot4_path = os.path.join(plot_dir, "loss_and_val_loss_vs_flops.pdf")
        plt.savefig(plot4_path)
        plt.close()
        print(f"Saved Plot 4 (raw train loss + val loss vs FLOPs) to {plot4_path}")

    print("All plots generated successfully.")


if __name__ == "__main__":
    config = LanguageModelingExperimentConfig()
    plot_training_metrics(config)