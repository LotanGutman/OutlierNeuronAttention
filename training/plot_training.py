import os
import math
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List
from training.training_config import LanguageModelingExperimentConfig
from src.HybridOutlierFactorizedAttentionTrain import SubwordLM


def moving_average(values, window=50):
    """Simple moving average."""
    if len(values) < window:
        return None
    return [
        sum(values[i:i + window]) / window
        for i in range(len(values) - window + 1)
    ]


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


def plot_training_metrics(configs: Union[LanguageModelingExperimentConfig, List[LanguageModelingExperimentConfig]]):
    if not isinstance(configs, list):
        configs = [configs]

    all_data = {}

    for config in configs:
        model_name = config.model_name
        checkpoint_path = f"data/training/{model_name}/checkpoint.pt"
        plot_dir = f"data/plots/training/{model_name}"

        if not os.path.exists(checkpoint_path):
            print(f"Error: No checkpoint found at {checkpoint_path}")
            continue

        os.makedirs(plot_dir, exist_ok=True)

        print(f"Loading metrics from {checkpoint_path}...")

        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            metrics = ckpt.get("metrics", None)

            if metrics is None or len(metrics.get("processed_tokens", [])) == 0:
                print(f"No metrics found in checkpoint or metrics are empty for {model_name}.")
                continue

        except Exception as e:
            print(f"Failed to load checkpoint for {model_name}: {e}")
            continue

        # --- Extract training data ---
        tokens = metrics["processed_tokens"]
        losses = metrics["loss"]

        # --- Extract validation data ---
        val_tokens = metrics.get("val_tokens", [])
        val_ppl = metrics.get("val_ppl", [])


        has_val = len(val_tokens) > 0 and len(val_ppl) > 0

        # Compute validation CE loss from perplexity: CE = log(PPL)
        val_loss = []
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
        print(f"[{model_name}] Calculated FLOPs per token: {flops_per_token:.2e}")
        
        flops_x = [t * flops_per_token for t in tokens]
        val_flops_x = [t * flops_per_token for t in val_tokens] if has_val else []

        all_data[model_name] = {
            "tokens": tokens,
            "losses": losses,
            "val_tokens": val_tokens,
            "val_loss": val_loss,
            "flops_x": flops_x,
            "val_flops_x": val_flops_x,
            "val_ppl": val_ppl,
            "has_val": has_val
        }

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
        plt.plot(tokens, losses, color="blue", alpha=0.6, linewidth=0.8, label="Training Loss (raw)")
        if smoothed is not None:
            plt.plot(smooth_tokens, smoothed, color="red", linewidth=2, label=f"Training Loss (smoothed, window={smooth_window})")
        plt.xscale("log")
        plt.ylim(3.0, 6.0)
        plt.xlim(10**7, tokens[-1])
        plt.title(f"{model_name} HOFA: Training Loss vs. Tokens (Log X)")
        plt.xlabel("Processed Tokens (log scale)")
        plt.ylabel("Cross Entropy Loss")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plot1_path = os.path.join(plot_dir, "loss_vs_tokens_logx.pdf")
        plt.savefig(plot1_path)
        plt.close()

        # ------------------------------------------------------------------
        # Plot 2: RAW Training Loss + Validation CE Loss (Linear X-axis)
        # ------------------------------------------------------------------
        if has_val:
            plt.figure(figsize=(10, 6))
            plt.plot(tokens, losses, color="red", alpha=0.6, linewidth=0.8, label="Training Loss (raw)")
            plt.plot(val_tokens, val_loss, color="green", marker="o", markersize=4, linewidth=1.5, linestyle="-", label="Validation Loss (log PPL)")
            plt.title(f"{model_name} HOFA: Training & Validation Loss (Linear X)")
            plt.xlabel("Processed Tokens")
            plt.ylabel("Cross Entropy Loss")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plot2_path = os.path.join(plot_dir, "loss_and_val_loss_vs_tokens.pdf")
            plt.savefig(plot2_path)
            plt.close()

        # ------------------------------------------------------------------
        # Plot 3: Validation Perplexity only (Linear X-axis)
        # ------------------------------------------------------------------
        if has_val:
            plt.figure(figsize=(10, 6))
            plt.plot(val_tokens, val_ppl, color="green", marker="o", markersize=4, linewidth=1.5, linestyle="-", label="Validation Perplexity")
            
            # Start x-axis at the second point to hide the step 0 gap
            plt.xscale('log')
            if len(val_tokens) > 1:
                plt.xlim(left=val_tokens[1])
            # Cap the Y-axis to just above the second point to hide the massive step 0 spike
            if len(val_ppl) > 1:
                plt.ylim(bottom=0, top=max(val_ppl[1:]) * 1.1)
                
            plt.title(f"{model_name} HOFA: Validation Perplexity")
            plt.xlabel("Processed Tokens")
            plt.ylabel("Perplexity")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plot3_path = os.path.join(plot_dir, "val_ppl_vs_tokens.pdf")
            plt.savefig(plot3_path)
            
            # Second copy with log Y
            plt.yscale('log')
            if len(val_ppl) > 1:
                plt.ylim(bottom=min(val_ppl) * 0.8) # Set a tight bottom limit
            plot3_logy_path = os.path.join(plot_dir, "val_ppl_vs_tokens_logy.pdf")
            plt.savefig(plot3_logy_path)
            
            plt.close()

        # ------------------------------------------------------------------
        # Plot 4: RAW Training Loss + Validation CE Loss (Linear FLOPs X-axis)
        # ------------------------------------------------------------------
        if has_val:
            plt.figure(figsize=(10, 6))
            plt.plot(flops_x, losses, color="red", alpha=0.6, linewidth=0.8, label="Training Loss (raw)")
            plt.plot(val_flops_x, val_loss, color="green", marker="o", markersize=4, linewidth=1.5, linestyle="-", label="Validation Loss (log PPL)")
            plt.title(f"{model_name} HOFA: Training & Validation Loss (Linear FLOPs)")
            plt.xlabel("Total FLOPs")
            plt.ylabel("Cross Entropy Loss")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plot4_path = os.path.join(plot_dir, "loss_and_val_loss_vs_flops.pdf")
            plt.savefig(plot4_path)
            plt.close()

    # ------------------------------------------------------------------
    # SHARED PLOT FOR PAPER (If multiple models provided)
    # ------------------------------------------------------------------
    if len(all_data) > 1:
        print("\nGenerating shared multi-model plot...")
        # Use academic paper theme matching profiling scripts
        plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        colors = sns.color_palette("tab10", n_colors=len(all_data))
        
        for i, (name, data) in enumerate(all_data.items()):
            color = colors[i]
            
            # Left Axis: vs Tokens
            axes[0].plot(data["tokens"], data["losses"], color=color, alpha=0.2, linewidth=0.8)
            if data["has_val"]:
                axes[0].plot(data["val_tokens"], data["val_loss"], color=color, label=name, marker='o', markersize=4, linewidth=2.5)
            
            # Right Axis: vs FLOPs
            axes[1].plot(data["flops_x"], data["losses"], color=color, alpha=0.2, linewidth=0.8)
            if data["has_val"]:
                axes[1].plot(data["val_flops_x"], data["val_loss"], color=color, label=name, marker='o', markersize=4, linewidth=2.5)
        
        # Formatting Left Axis (Tokens)
        axes[0].set_title("Validation & Train Loss vs. Tokens", pad=10)
        axes[0].set_xlabel("Processed Tokens", fontsize=11)
        axes[0].set_ylabel("Cross Entropy Loss", fontsize=11)
        axes[0].grid(True, linestyle=':', alpha=0.6)
        
        # Formatting Right Axis (FLOPs)
        axes[1].set_title("Validation & Train Loss vs. FLOPs", pad=10)
        axes[1].set_xlabel("Total FLOPs", fontsize=11)
        axes[1].grid(True, linestyle=':', alpha=0.6)
        
        # Add shared legend at the top
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=len(all_data), frameon=False, fontsize=12)
        
        # Crop the Y axis to skip the massive initial loss
        axes[0].set_ylim(top=8.0)
        axes[1].set_ylim(top=8.0)
        
        plt.tight_layout()
        
        # Generate truncated sorted name
        sorted_names = sorted(list(all_data.keys()))
        combined_name = "_".join(sorted_names)
        
        shared_dir = "data/plots/training/shared"
        os.makedirs(shared_dir, exist_ok=True)
        pdf_path = os.path.join(shared_dir, f"{combined_name}.pdf")
        plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
        
        # Second plot: exactly the same but log scale for tokens and flops
        axes[0].set_xscale('log')
        axes[1].set_xscale('log')
        
        # Set x limits to start at the first point where loss drops below the 8.0 ceiling
        min_token_visible = float('inf')
        min_flop_visible = float('inf')
        for data in all_data.values():
            for t, l, f in zip(data["tokens"], data["losses"], data["flops_x"]):
                if l <= 8.0:
                    min_token_visible = min(min_token_visible, t)
                    min_flop_visible = min(min_flop_visible, f)
                    break
                    
        if min_token_visible == float('inf'):
            min_token_visible, min_flop_visible = 1, 1
            
        axes[0].set_xlim(left=min_token_visible)
        axes[1].set_xlim(left=min_flop_visible)
        
        log_pdf_path = os.path.join(shared_dir, f"{combined_name}_log.pdf")
        plt.savefig(log_pdf_path, bbox_inches='tight', format='pdf')
        
        plt.close()
        print(f"Saved shared high-clarity plots to {pdf_path} and {log_pdf_path}")
        
        # ------------------------------------------------------------------
        # SHARED PERPLEXITY PLOT
        # ------------------------------------------------------------------
        if any(d["has_val"] for d in all_data.values()):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            
            for i, (name, data) in enumerate(all_data.items()):
                if not data["has_val"]:
                    continue
                color = colors[i]
                
                axes[0].plot(data["val_tokens"], data["val_ppl"], color=color, label=name, marker='o', markersize=4, linewidth=1.5)
                axes[1].plot(data["val_flops_x"], data["val_ppl"], color=color, label=name, marker='o', markersize=4, linewidth=1.5)
            
            axes[0].set_title("Validation Perplexity vs. Tokens", pad=10)
            axes[0].set_xlabel("Processed Tokens", fontsize=11)
            axes[0].set_ylabel("Perplexity", fontsize=11)
            axes[0].grid(True, linestyle=':', alpha=0.6)
            
            axes[1].set_title("Validation Perplexity vs. FLOPs", pad=10)
            axes[1].set_xlabel("Total FLOPs", fontsize=11)
            axes[1].grid(True, linestyle=':', alpha=0.6)
            
            axes[0].set_xscale('log')
            axes[1].set_xscale('log')
            
            min_token_step1 = min([d["val_tokens"][1] for d in all_data.values() if len(d["val_tokens"]) > 1] or [1])
            min_flop_step1 = min([d["val_flops_x"][1] for d in all_data.values() if len(d["val_flops_x"]) > 1] or [1])
            
            axes[0].set_xlim(left=min_token_step1)
            axes[1].set_xlim(left=min_flop_step1)
            
            max_ppl = 0
            min_ppl = float('inf')
            for d in all_data.values():
                if d["has_val"] and len(d["val_ppl"]) > 1:
                    max_ppl = max(max_ppl, max(d["val_ppl"][1:]))
                    min_ppl = min(min_ppl, min(d["val_ppl"]))
                    
            if max_ppl > 0:
                axes[0].set_ylim(bottom=0, top=max_ppl * 1.1)
                axes[1].set_ylim(bottom=0, top=max_ppl * 1.1)
                
            handles, labels = axes[1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=len(all_data), frameon=False, fontsize=12)
            
            plt.tight_layout()
            
            ppl_pdf_path = os.path.join(shared_dir, f"{combined_name}_ppl.pdf")
            plt.savefig(ppl_pdf_path, bbox_inches='tight', format='pdf')
            
            # Second copy with log Y
            axes[0].set_yscale('log')
            axes[1].set_yscale('log')
            if max_ppl > 0:
                axes[0].set_ylim(bottom=min_ppl * 0.8, top=max_ppl * 1.1)
                axes[1].set_ylim(bottom=min_ppl * 0.8, top=max_ppl * 1.1)
                
            ppl_logy_pdf_path = os.path.join(shared_dir, f"{combined_name}_ppl_logy.pdf")
            plt.savefig(ppl_logy_pdf_path, bbox_inches='tight', format='pdf')
            
            plt.close()
            print(f"Saved shared perplexity plots to {ppl_pdf_path} and {ppl_logy_pdf_path}")

    print("\nAll plots generated successfully.")

if __name__ == "__main__":
    config = LanguageModelingExperimentConfig()
    plot_training_metrics(config)