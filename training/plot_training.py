import os
import math
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Any
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
        outlier_flops += 6 * H * (r + d_h) * N
        inlier_flops += 18 * H * (d_h - r) * d_h
    return dense_flops + outlier_flops + inlier_flops


def plot_training_metrics(configs: Union[Any, List[Any]], subdirectory: str = ""):
    if not isinstance(configs, list):
        configs = [configs]

    all_data = {}

    for config in configs:
        if hasattr(config, "model"):
            lm_cfg = config.model
        else:
            lm_cfg = config

        model_name = lm_cfg.model_name
        plot_name = getattr(lm_cfg, "plot_name", model_name)
        plot_dir = f"data/plots/training/{model_name}"

        candidate_paths = [
            f"data/training/{model_name}/checkpoint.pt",
            f"data/training/{model_name}/checkpoint_best_val.pt",
            f"data/training/{model_name}/checkpoint_emergency.pt"
        ]
        checkpoint_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                checkpoint_path = p
                break

        if checkpoint_path is None:
            print(f"Error: No checkpoint found for {model_name} (checked: {', '.join(candidate_paths)})")
            continue

        os.makedirs(plot_dir, exist_ok=True)

        print(f"Loading metrics from {checkpoint_path}...")

        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            metrics = ckpt.get("metrics", None)

            if metrics is None or (len(metrics.get("processed_tokens", [])) == 0 and len(metrics.get("loss", [])) == 0):
                print(f"No metrics found in checkpoint or metrics are empty for {model_name}.")
                continue

        except Exception as e:
            print(f"Failed to load checkpoint for {model_name}: {e}")
            continue

        is_cpt = ("_cpt" in model_name.lower()) or ("val_pg19_ppl" in metrics)

        # Training metrics
        tokens = metrics.get("processed_tokens", [])
        losses = metrics.get("loss", [])
        steps = list(range(len(losses)))

        # Validation metrics
        if is_cpt:
            val_steps = metrics.get("val_steps", [])
            val_pg19_ppl = metrics.get("val_pg19_ppl", [])
            val_fineweb_ppl_yarn = metrics.get("val_fineweb_ppl_yarn", [])
            val_fineweb_ppl_native = metrics.get("val_fineweb_ppl_native", [])

            tokens_per_step = lm_cfg.batch_size * lm_cfg.seq_len
            val_tokens = []
            for s in val_steps:
                if s < len(tokens):
                    val_tokens.append(tokens[s])
                else:
                    val_tokens.append(s * tokens_per_step)

            val_ppl = val_pg19_ppl
            val_loss = [math.log(p) if (p is not None and p > 0) else None for p in val_pg19_ppl]
            has_val = len(val_steps) > 0 and len(val_ppl) > 0
        else:
            val_steps = []
            val_tokens = metrics.get("val_tokens", [])
            val_ppl = metrics.get("val_ppl", [])
            val_fineweb_ppl_yarn = []
            val_fineweb_ppl_native = []
            has_val = len(val_tokens) > 0 and len(val_ppl) > 0
            val_loss = [math.log(p) for p in val_ppl] if has_val else []

        # HOFA FLOPs per token
        model = SubwordLM(lm_cfg.vocab_size, lm_cfg.model_config)
        P = get_active_non_embedding_parameters(model)
        H = lm_cfg.model_config.num_heads
        d_h = lm_cfg.model_config.d_head
        N_seq = lm_cfg.seq_len
        r_list = lm_cfg.model_config.r
        if not isinstance(r_list, (list, tuple)):
            r_list = [r_list] * lm_cfg.model_config.num_layers
            
        flops_per_token = calc_hofa_flops_per_token(P, H, d_h, r_list, N_seq)
        print(f"[{model_name}] Calculated FLOPs per token: {flops_per_token:.2e}")
        
        flops_x = [t * flops_per_token for t in tokens]
        val_flops_x = [t * flops_per_token for t in val_tokens] if has_val else []

        all_data[plot_name] = {
            "model_name": model_name,
            "tokens": tokens,
            "steps": steps,
            "losses": losses,
            "val_steps": val_steps,
            "val_tokens": val_tokens,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "val_fineweb_ppl_yarn": val_fineweb_ppl_yarn,
            "val_fineweb_ppl_native": val_fineweb_ppl_native,
            "flops_x": flops_x,
            "val_flops_x": val_flops_x,
            "has_val": has_val,
            "is_cpt": is_cpt,
        }

        if is_cpt:
            # CPT step plots
            smooth_window = min(50, max(5, len(losses) // 4)) if len(losses) >= 10 else 1
            smoothed = moving_average(losses, smooth_window) if smooth_window > 1 else None

            # Unified training and validation loss vs steps
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses, color="tab:gray", alpha=0.35, linewidth=0.8, label="Training Loss")
            if smoothed is not None:
                offset = smooth_window // 2
                smooth_steps = steps[offset:offset + len(smoothed)]
                plt.plot(smooth_steps, smoothed, color="tab:blue", alpha=0.8, linewidth=1.5, label="Training Loss (smoothed)")

            if has_val:
                # PG-19 16k val loss
                plt.plot(val_steps, val_loss, color="tab:green", marker="o", markersize=5, linewidth=2.0, linestyle="-", label="PG-19 16k Val Loss")

                # FineWeb 1k loss
                if val_fineweb_ppl_yarn:
                    valid_fw = [(s, math.log(p)) for s, p in zip(val_steps, val_fineweb_ppl_yarn) if p is not None and p > 0]
                    if valid_fw:
                        fw_s, fw_l = zip(*valid_fw)
                        plt.plot(fw_s, fw_l, color="tab:purple", marker="s", markersize=5, linewidth=1.8, linestyle="--", label="FineWeb 1k Loss")

            plt.title(f"{plot_name}: Training & Validation Loss", fontsize=14, fontweight="bold")
            plt.xlabel("Steps", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend(frameon=True)
            plt.tight_layout()
            loss_plot_path = os.path.join(plot_dir, "loss_vs_steps.pdf")
            plt.savefig(loss_plot_path, bbox_inches="tight")
            plt.close()

            # Validation perplexity vs steps
            if has_val:
                plt.figure(figsize=(10, 6))
                plt.plot(val_steps, val_ppl, color="tab:green", marker="o", markersize=5, linewidth=2.0, linestyle="-", label="PG-19 16k Val PPL")
                if val_fineweb_ppl_yarn:
                    valid_fw_ppl = [(s, p) for s, p in zip(val_steps, val_fineweb_ppl_yarn) if p is not None and p > 0]
                    if valid_fw_ppl:
                        fw_s, fw_p = zip(*valid_fw_ppl)
                        plt.plot(fw_s, fw_p, color="tab:purple", marker="s", markersize=5, linewidth=1.8, linestyle="--", label="FineWeb 1k PPL")

                plt.title(f"{plot_name}: Validation Perplexity", fontsize=14, fontweight="bold")
                plt.xlabel("Steps", fontsize=12)
                plt.ylabel("Perplexity", fontsize=12)
                plt.grid(True, linestyle=":", alpha=0.6)
                plt.legend(frameon=True)
                plt.tight_layout()
                ppl_plot_path = os.path.join(plot_dir, "val_ppl_vs_steps.pdf")
                plt.savefig(ppl_plot_path, bbox_inches="tight")
                plt.close()
        else:
            # Pretraining token plots
            smooth_window = 150
            smoothed = moving_average(losses, smooth_window)

            if smoothed is not None:
                offset = smooth_window // 2
                smooth_tokens = tokens[offset:offset + len(smoothed)]
            else:
                smooth_tokens = None

            # Plot 1: Training Loss vs Tokens (Log X-axis) – RAW + SMOOTHED
            plt.figure(figsize=(10, 6))
            plt.plot(tokens, losses, color="blue", alpha=0.6, linewidth=0.8, label="Training Loss (raw)")
            if smoothed is not None:
                plt.plot(smooth_tokens, smoothed, color="red", linewidth=2, label=f"Training Loss (smoothed, window={smooth_window})")
            plt.xscale("log")
            plt.ylim(3.0, 6.0)
            plt.xlim(10**7, tokens[-1])
            plt.title(f"{plot_name}: Training Loss vs. Tokens (Log X)")
            plt.xlabel("Processed Tokens (log scale)")
            plt.ylabel("Cross Entropy Loss")
            plt.grid(True, which="both", linestyle="--", alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plot1_path = os.path.join(plot_dir, "loss_vs_tokens_logx.pdf")
            plt.savefig(plot1_path)
            plt.close()

            # Plot 2: RAW Training Loss + Validation CE Loss (Linear X-axis)
            if has_val:
                plt.figure(figsize=(10, 6))
                plt.plot(tokens, losses, color="red", alpha=0.6, linewidth=0.8, label="Training Loss (raw)")
                plt.plot(val_tokens, val_loss, color="green", marker="o", markersize=4, linewidth=1.5, linestyle="-", label="Validation Loss (log PPL)")
                plt.title(f"{plot_name}: Training & Validation Loss (Linear X)")
                plt.xlabel("Processed Tokens")
                plt.ylabel("Cross Entropy Loss")
                plt.grid(True, linestyle="--", alpha=0.4)
                plt.legend()
                plt.tight_layout()
                plot2_path = os.path.join(plot_dir, "loss_and_val_loss_vs_tokens.pdf")
                plt.savefig(plot2_path)
                plt.close()

            # Plot 3: Validation Perplexity only (Linear X-axis)
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
                    
                plt.title(f"{plot_name}: Validation Perplexity")
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

            # Plot 4: RAW Training Loss + Validation CE Loss (Linear FLOPs X-axis)
            if has_val:
                plt.figure(figsize=(10, 6))
                plt.plot(flops_x, losses, color="red", alpha=0.6, linewidth=0.8, label="Training Loss (raw)")
                plt.plot(val_flops_x, val_loss, color="green", marker="o", markersize=4, linewidth=1.5, linestyle="-", label="Validation Loss (log PPL)")
                plt.title(f"{plot_name}: Training & Validation Loss (Linear FLOPs)")
                plt.xlabel("Total FLOPs")
                plt.ylabel("Cross Entropy Loss")
                plt.grid(True, linestyle="--", alpha=0.4)
                plt.legend()
                plt.tight_layout()
                plot4_path = os.path.join(plot_dir, "loss_and_val_loss_vs_flops.pdf")
                plt.savefig(plot4_path)
                plt.close()
    if len(all_data) > 1:
        print("\nGenerating shared multi-model plot...")
        # Use academic paper theme matching publication rcParams
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 13,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.labelsize': 13,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11
        })
        
        colors = sns.color_palette("tab10", n_colors=len(all_data))
        
        shared_dir = "data/plots/training/shared"
        if subdirectory:
            import re
            safe_subdir = re.sub(r'[^a-zA-Z0-9_\-]', '', subdirectory)
            shared_dir = os.path.join(shared_dir, safe_subdir)
        os.makedirs(shared_dir, exist_ok=True)
        
        sorted_model_names = sorted([d["model_name"] for d in all_data.values()])
        combined_name = "_".join(sorted_model_names)

        is_any_cpt = any(d.get("is_cpt", False) for d in all_data.values())

        if is_any_cpt:
            # CPT shared curves
            for x_axis in ["flops", "tokens"]:
                # PPL vs x
                plt.figure(figsize=(10, 6))
                for i, (name, data) in enumerate(all_data.items()):
                    color = colors[i]
                    if data["has_val"]:
                        x_vals = data["val_flops_x"] if x_axis == "flops" else data["val_tokens"]
                        y_vals = data["val_ppl"]
                        plt.plot(x_vals, y_vals, color=color, label=name, marker='o', markersize=5, linewidth=2.5)

                title_x = "Total FLOPs" if x_axis == "flops" else "Processed Tokens"
                plt.title(f"CPT PG-19 16k Validation Perplexity vs. {title_x}", fontsize=14, fontweight='bold', pad=10)
                plt.xlabel("Total FLOPs" if x_axis == "flops" else "Processed Tokens", fontsize=12)
                plt.ylabel("Validation Perplexity (PG-19 16k)", fontsize=12)
                plt.grid(True, linestyle=':', alpha=0.5)
                plt.legend(frameon=True, framealpha=0.9, fontsize=11)
                plt.tight_layout()
                out_path = os.path.join(shared_dir, f"{combined_name}_ppl_vs_{x_axis}.pdf")
                plt.savefig(out_path, bbox_inches='tight', format='pdf')
                plt.close()

                # PPL vs x (log)
                plt.figure(figsize=(10, 6))
                for i, (name, data) in enumerate(all_data.items()):
                    color = colors[i]
                    if data["has_val"]:
                        x_vals = data["val_flops_x"] if x_axis == "flops" else data["val_tokens"]
                        y_vals = data["val_ppl"]
                        plt.plot(x_vals, y_vals, color=color, label=name, marker='o', markersize=5, linewidth=2.5)

                plt.title(f"CPT PG-19 16k Validation Perplexity vs. {title_x} (Log X)", fontsize=14, fontweight='bold', pad=10)
                plt.xlabel(f"{title_x} (log scale)", fontsize=12)
                plt.ylabel("Validation Perplexity (PG-19 16k)", fontsize=12)
                plt.xscale('log')
                plt.grid(True, which="both", linestyle=':', alpha=0.5)
                plt.legend(frameon=True, framealpha=0.9, fontsize=11)
                plt.tight_layout()
                out_path = os.path.join(shared_dir, f"{combined_name}_ppl_vs_{x_axis}_log.pdf")
                plt.savefig(out_path, bbox_inches='tight', format='pdf')
                plt.close()

                # Loss vs x
                plt.figure(figsize=(10, 6))
                for i, (name, data) in enumerate(all_data.items()):
                    color = colors[i]
                    if data["has_val"]:
                        x_vals = data["val_flops_x"] if x_axis == "flops" else data["val_tokens"]
                        y_vals = data["val_loss"]
                        plt.plot(x_vals, y_vals, color=color, label=name, marker='o', markersize=5, linewidth=2.5)

                plt.title(f"CPT PG-19 16k Validation Loss vs. {title_x}", fontsize=14, fontweight='bold', pad=10)
                plt.xlabel("Total FLOPs" if x_axis == "flops" else "Processed Tokens", fontsize=12)
                plt.ylabel("Cross Entropy Loss (PG-19 16k)", fontsize=12)
                plt.grid(True, linestyle=':', alpha=0.5)
                plt.legend(frameon=True, framealpha=0.9, fontsize=11)
                plt.tight_layout()
                out_path = os.path.join(shared_dir, f"{combined_name}_loss_vs_{x_axis}.pdf")
                plt.savefig(out_path, bbox_inches='tight', format='pdf')
                plt.close()

            print(f"Saved CPT shared plots to {shared_dir}")
        else:
            # Pretraining shared curves
            # Pre-calculate global limits
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

            min_token_step1 = min([d["val_tokens"][1] for d in all_data.values() if d["has_val"] and len(d["val_tokens"]) > 1] or [1])
            min_flop_step1 = min([d["val_flops_x"][1] for d in all_data.values() if d["has_val"] and len(d["val_flops_x"]) > 1] or [1])

            max_ppl = 0
            min_ppl = float('inf')
            has_any_val = any(d["has_val"] for d in all_data.values())
            if has_any_val:
                for d in all_data.values():
                    if d["has_val"] and len(d["val_ppl"]) > 1:
                        max_ppl = max(max_ppl, max(d["val_ppl"][1:]))
                        min_ppl = min(min_ppl, min(d["val_ppl"]))

            def _plot_shared(metric, x_axis, is_log, is_zoomed=False):
                figsize = (12.5, 5.0)
                plt.figure(figsize=figsize)
                
                for i, (name, data) in enumerate(all_data.items()):
                    color = colors[i]
                    
                    if metric == "loss":
                        if data["has_val"]:
                            x_data_val = data["val_tokens"] if x_axis == "tokens" else data["val_flops_x"]
                            plt.plot(x_data_val, data["val_loss"], color=color, label=name, marker='o', markersize=4, linewidth=3.0)
                    elif metric == "ppl":
                        if data["has_val"]:
                            x_data_val = data["val_tokens"] if x_axis == "tokens" else data["val_flops_x"]
                            if is_zoomed:
                                window = min(5, len(data["val_ppl"]))
                                smoothed_ppl = moving_average(data["val_ppl"], window=window)
                                if smoothed_ppl is not None:
                                    plt.plot(x_data_val[window-1:], smoothed_ppl, color=color, label=name, linewidth=3.0)
                                else:
                                    plt.plot(x_data_val, data["val_ppl"], color=color, label=name, linewidth=3.0)
                            else:
                                plt.plot(x_data_val, data["val_ppl"], color=color, label=name, marker='o', markersize=4, linewidth=3.0)

                title_metric = "Validation & Train Loss" if metric == "loss" else "Validation Perplexity"
                title_x = "Tokens" if x_axis == "tokens" else "FLOPs"
                plt.title(f"{title_metric} vs. {title_x}", pad=10, fontsize=15, fontweight='bold')
                
                plt.xlabel("Processed Tokens" if x_axis == "tokens" else "Total FLOPs")
                plt.ylabel("Cross Entropy Loss" if metric == "loss" else "Perplexity")
                plt.grid(True, linestyle=':', alpha=0.3, color='#e0e0e0')
                
                plt.legend(frameon=True, framealpha=0.9, edgecolor='#d0d0d0', fontsize=11)
                
                if is_log:
                    plt.xscale('log')
                    if metric == "loss":
                        plt.xlim(left=min_token_visible if x_axis == "tokens" else min_flop_visible)
                    else:
                        plt.xlim(left=min_token_step1 if x_axis == "tokens" else min_flop_step1)
                
                if metric == "loss":
                    plt.ylim(top=8.0)
                elif metric == "ppl" and max_ppl > 0:
                    if is_log:
                        plt.yscale('log')
                        plt.ylim(bottom=min_ppl * 0.8, top=max_ppl * 1.1)
                    else:
                        plt.ylim(bottom=0, top=max_ppl * 1.1)
                        
                if is_zoomed:
                    max_x = max([data["tokens"][-1] if x_axis == "tokens" else data["flops_x"][-1] for data in all_data.values() if len(data["tokens"]) > 0] or [1])
                    zoom_start = max_x / 20.0
                    plt.xlim(left=zoom_start, right=max_x * 1.05)
                    
                    # Dynamically compute max_ppl and min_ppl in this window
                    window_max_ppl = 0
                    window_min_ppl = float('inf')
                    for d in all_data.values():
                        if d["has_val"]:
                            x_arr = d["val_tokens"] if x_axis == "tokens" else d["val_flops_x"]
                            ppl_arr = d["val_ppl"]
                            window = min(5, len(ppl_arr))
                            smoothed_ppl = moving_average(ppl_arr, window=window)
                            if smoothed_ppl is not None:
                                x_arr = x_arr[window-1:]
                                ppl_arr = smoothed_ppl
                                
                            for x_val, p_val in zip(x_arr, ppl_arr):
                                if x_val >= zoom_start:
                                    window_max_ppl = max(window_max_ppl, p_val)
                                    window_min_ppl = min(window_min_ppl, p_val)
                    if window_max_ppl > 0 and window_min_ppl != float('inf'):
                        plt.ylim(bottom=window_min_ppl * 0.95, top=window_max_ppl * 1.05)
                        
                plt.tight_layout()
                
                suffix_str = "_log" if is_log else ""
                if is_zoomed:
                    suffix_str += "_zoomed"
                filename = f"{combined_name}_{metric}_vs_{x_axis}{suffix_str}.pdf"
                pdf_path = os.path.join(shared_dir, filename)
                plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
                plt.close()

            # Generate the plots
            for metric in ["loss", "ppl"]:
                if metric == "ppl" and not has_any_val:
                    continue
                for x_axis in ["tokens", "flops"]:
                    for is_log in [False, True]:
                        _plot_shared(metric, x_axis, is_log, is_zoomed=False)
                        # Create the extra zoomed plots for log-log perplexity
                        if metric == "ppl" and is_log:
                            _plot_shared(metric, x_axis, is_log, is_zoomed=True)

            # Generate an extra 1x2 grid specifically for Loss vs FLOPs (Linear | Log)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)
            for i, (name, data) in enumerate(all_data.items()):
                color = colors[i]
                if data["has_val"]:
                    # Left side (Linear)
                    axes[0].plot(data["val_flops_x"], data["val_loss"], color=color, label=name, marker='o', markersize=4, linewidth=2.5)
                    # Right side (Log)
                    axes[1].plot(data["val_flops_x"], data["val_loss"], color=color, label=name, marker='o', markersize=4, linewidth=2.5)

            for ax, is_log in zip(axes, [False, True]):
                title_suffix = "(Log Scale)" if is_log else "(Linear Scale)"
                ax.set_title(f"Validation & Train Loss vs. FLOPs\n{title_suffix}", pad=10)
                ax.set_xlabel("Total FLOPs")
                if not is_log:
                    ax.set_ylabel("Cross Entropy Loss")
                ax.grid(True, linestyle=':', alpha=0.4)
                if is_log:
                    ax.set_xscale('log')
                    ax.set_xlim(left=min_flop_visible)
                ax.set_ylim(top=8.0)
                
            handles, labels = axes[1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=len(all_data), frameon=False)
            plt.tight_layout()
            combined_pdf_path = os.path.join(shared_dir, f"{combined_name}_loss_vs_flops_combined.pdf")
            plt.savefig(combined_pdf_path, bbox_inches='tight', format='pdf')
            plt.close()

            print(f"Saved 8 single shared plots and 1 combined 1x2 plot to {shared_dir}")

    print("\nAll plots generated successfully.")

if __name__ == "__main__":
    config = LanguageModelingExperimentConfig()
    plot_training_metrics(config)