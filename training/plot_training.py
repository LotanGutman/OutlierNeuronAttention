import os
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
    lrs = metrics["learning_rate"]

    # --- Extract validation data (if it exists) ---
    val_tokens = metrics.get("val_tokens", [])
    val_ppl = metrics.get("val_ppl", [])

    # Drop the first validation point
    if len(val_tokens) > 1:
        val_tokens = val_tokens[1:]
        val_ppl = val_ppl[1:]

    has_val = len(val_tokens) > 0 and len(val_ppl) > 0

    # ------------------------------------------------------------------
    # Compute smoothed loss
    # ------------------------------------------------------------------
    smooth_window = 150
    smoothed = moving_average(losses, smooth_window)

    if smoothed is not None:
        offset = smooth_window // 2
        smooth_tokens = tokens[offset:offset + len(smoothed)]
    else:
        smooth_tokens = None

    # ------------------------------------------------------------------
    # 1. Loss vs Tokens (Linear X-axis)
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(
        tokens,
        losses,
        color="blue",
        alpha=0.8,
        linewidth=1,
        label="Training Loss",
    )

    if smoothed is not None:
        plt.plot(
            smooth_tokens,
            smoothed,
            color="red",
            linewidth=2,
            label=f"Smoothed (window={smooth_window})",
        )

    plt.title(f"{model_name} HOFA: Loss vs. Tokens")
    plt.xlabel("Processed Tokens")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    linear_loss_path = os.path.join(plot_dir, "loss_vs_tokens.pdf")
    plt.savefig(linear_loss_path)
    plt.close()

    print(f"Saved linear loss plot to {linear_loss_path}")

    # ------------------------------------------------------------------
    # 2. Loss vs Tokens (Log X-axis)
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(
        tokens,
        losses,
        color="blue",
        alpha=0.8,
        linewidth=1,
        label="Training Loss",
    )

    if smoothed is not None:
        plt.plot(
            smooth_tokens,
            smoothed,
            color="red",
            linewidth=2,
            label=f"Smoothed (window={smooth_window})",
        )

    plt.xscale("log")
    plt.ylim(3.0, 6.0)
    plt.xlim(10**7, tokens[-1])

    plt.title(f"{model_name} HOFA: Loss vs. Tokens (Log X)")
    plt.xlabel("Processed Tokens (log scale)")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    log_loss_path = os.path.join(plot_dir, "loss_vs_tokens_logx.pdf")
    plt.savefig(log_loss_path)
    plt.close()

    print(f"Saved log-x loss plot to {log_loss_path}")

    # ------------------------------------------------------------------
    # 3. Learning Rate vs Tokens
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(
        tokens,
        lrs,
        color="orange",
        linewidth=2,
    )

    plt.title(f"{model_name} HOFA: Learning Rate Schedule")
    plt.xlabel("Processed Tokens")
    plt.ylabel("Learning Rate")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    lr_plot_path = os.path.join(plot_dir, "lr_vs_tokens.pdf")
    plt.savefig(lr_plot_path)
    plt.close()

    print(f"Saved LR schedule plot to {lr_plot_path}")

    # ------------------------------------------------------------------
    # 4. NEW: Validation Perplexity (Standalone)
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
            label="Validation Perplexity",
        )

        plt.title(f"{model_name} HOFA: Validation Perplexity")
        plt.xlabel("Processed Tokens")
        plt.ylabel("Perplexity")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        val_ppl_path = os.path.join(plot_dir, "val_ppl_vs_tokens.pdf")
        plt.savefig(val_ppl_path)
        plt.close()

        print(f"Saved validation PPL plot to {val_ppl_path}")

    # ------------------------------------------------------------------
    # 5. NEW: Combined Training Loss + Validation Perplexity (Twin Axes)
    # ------------------------------------------------------------------
    if has_val and smoothed is not None:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Left y-axis: Training Loss
        color1 = "red"
        ax1.set_xlabel("Processed Tokens")
        ax1.set_ylabel("Cross Entropy Loss", color=color1)
        ax1.plot(
            smooth_tokens,
            smoothed,
            color=color1,
            linewidth=2,
            label="Training Loss (smoothed)",
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, linestyle="--", alpha=0.3)

        # Right y-axis: Validation Perplexity
        ax2 = ax1.twinx()
        color2 = "green"
        ax2.set_ylabel("Validation Perplexity", color=color2)
        ax2.plot(
            val_tokens,
            val_ppl,
            color=color2,
            marker="o",
            markersize=5,
            linewidth=1.5,
            linestyle="-",
            label="Validation Perplexity",
        )
        ax2.tick_params(axis="y", labelcolor=color2)

        # Add legends (combine both axes)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.title(f"{model_name} HOFA: Training Loss & Validation Perplexity")
        fig.tight_layout()

        combined_path = os.path.join(plot_dir, "loss_and_val_ppl_vs_tokens.pdf")
        plt.savefig(combined_path)
        plt.close()

        print(f"Saved combined Loss + Validation PPL plot to {combined_path}")

    print("All plots generated successfully.")


if __name__ == "__main__":
    config = LanguageModelingExperimentConfig()
    plot_training_metrics(config)