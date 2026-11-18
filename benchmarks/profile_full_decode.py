"""End-to-end 350M decoding profiling benchmark (B=1 top row, B=4 bottom row) with CUDA Graphs."""
import os
from dataclasses import replace
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

from benchmarks.benchmarks_configs import CACHE_PATH, FullDecodeProfilingConfig
from src.HybridOutlierFactorizedAttention import SubwordLM
from training.training_config import make_350M_HOFA, make_350M_MHA


def _allocate_decode_state(model: SubwordLM, batch_size: int, sl: int, device: torch.device):
    cache_O_list, state_I_list = [], []
    for layer in model.layers:
        attn = layer.attn
        H, d_head, r, j = attn.num_heads, attn.d_head, attn.r, attn.j
        c_O = (
            torch.randn(batch_size, H, sl, r, device=device, dtype=torch.bfloat16),
            torch.randn(batch_size, H, sl, d_head, device=device, dtype=torch.bfloat16),
        ) if r > 0 else None
        s_I = (
            torch.randn(batch_size, H, j, d_head, device=device, dtype=torch.float32),
            torch.empty(batch_size, H, j, d_head, device=device, dtype=torch.float32),
        ) if (j > 0 and r < d_head) else None
        cache_O_list.append(c_O)
        state_I_list.append(s_I)
    return cache_O_list, state_I_list


def _profile_decode_model(cfg, config: FullDecodeProfilingConfig, batch_size: int, device: torch.device):
    mcfg = replace(cfg.model_config, device=str(device))
    model = SubwordLM(config.vocab_size, mcfg).to(device=device, dtype=torch.bfloat16).eval()
    torch.cuda.synchronize()
    base_bytes = torch.cuda.memory_allocated(device)

    tok_s_list, mem_gb_list = [], []
    token = torch.randint(0, config.vocab_size, (batch_size, 1), device=device)

    for sl in config.seq_lengths:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        g = None
        try:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                cache_O, state_I = _allocate_decode_state(model, batch_size, sl, device)
                s = torch.cuda.Stream(device=device)
                s.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(s):
                    for _ in range(config.warmup_steps):
                        _ = model.forward_step(token, cache_O, state_I, cache_seq_len=sl - 1)
                torch.cuda.current_stream(device).wait_stream(s)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                _ = model.forward_step(token, cache_O, state_I, cache_seq_len=sl - 1)
                torch.cuda.synchronize()
                kv_gb = (torch.cuda.max_memory_allocated(device) - base_bytes) / (1024 ** 3)

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    _ = model.forward_step(token, cache_O, state_I, cache_seq_len=sl - 1)
                torch.cuda.synchronize()

                g.replay()
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(config.active_steps):
                    g.replay()
                end.record()
                torch.cuda.synchronize()

                tok_s = (batch_size * 1000.0) / (start.elapsed_time(end) / config.active_steps)
                del g, cache_O, state_I
                g = None
        except torch.cuda.OutOfMemoryError:
            if g is not None:
                del g
            tok_s, kv_gb = float("nan"), float("nan")
        tok_s_list.append(tok_s)
        mem_gb_list.append(kv_gb)
        print(f"  [Decode {cfg.model_name} CUDAGraph B={batch_size}] N={sl:<7d} | tok/s={tok_s:8.1f} | KV+state={kv_gb:.3f} GB")

    del model, token
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return tok_s_list, mem_gb_list


def _plot_decode_row(axs_row, lens, dec_mha, dec_hofa, mem_mha, mem_hofa, show_titles: bool, show_xlabel: bool, vram_gb: float = 32.0):
    s_mha = {"marker": "o", "color": "#D55E00", "linewidth": 2.5, "markersize": 6}
    s_hyb = {"marker": "s", "color": "#0072B2", "linewidth": 2.5, "markersize": 6}
    fmt_x = FuncFormatter(lambda x, pos: f"{int(x/1024)}k" if x >= 1024 else f"{int(x)}")

    mha_oom_indices = [i for i, v in enumerate(dec_mha) if np.isnan(v)]
    first_oom_idx = mha_oom_indices[0] if mha_oom_indices else None

    def _annotate_pct(ax, x_val, y_a, y_b, fmt_str, side="right"):
        ax.annotate("", xy=(x_val, y_b), xytext=(x_val, y_a), arrowprops=dict(arrowstyle="<->", color="black", lw=1.4))
        if side == "left":
            ax.text(x_val * 0.84, 0.5 * (y_a + y_b), fmt_str, color="black", va="center", ha="right", fontsize=11, fontweight="bold", clip_on=False)
        else:
            ax.text(x_val * 1.12, 0.5 * (y_a + y_b), fmt_str, color="black", va="center", ha="left", fontsize=11, fontweight="bold", clip_on=False)

    # 1. Decoding Throughput
    ax = axs_row[0]
    ax.plot(lens, dec_mha, label="350M MHA", **s_mha)
    ax.plot(lens, dec_hofa, label="350M HOFA (U-Shaped)", **s_hyb)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(fmt_x)
    ax.set_xticks(lens)
    if show_xlabel:
        ax.set_xlabel("Context Length ($N$)")
    ax.set_ylabel("Tokens / sec")
    if show_titles:
        ax.set_title("Decoding Throughput")
    ax.grid(True, linestyle=":", alpha=0.3, color="#e0e0e0")

    valid_both = [i for i, (m, h) in enumerate(zip(dec_mha, dec_hofa)) if not np.isnan(m) and not np.isnan(h) and m > 0]
    if valid_both:
        i_last = valid_both[-1]
        pct_spd = (dec_hofa[i_last] - dec_mha[i_last]) / dec_mha[i_last] * 100.0
        pct_side = "left" if first_oom_idx is not None else "right"
        _annotate_pct(ax, lens[i_last], dec_mha[i_last], dec_hofa[i_last], f"+{pct_spd:.1f}%", side=pct_side)
    if first_oom_idx is not None and first_oom_idx > 0:
        x_prev, y_prev = lens[first_oom_idx - 1], dec_mha[first_oom_idx - 1]
        x_oom = lens[first_oom_idx]
        if not np.isnan(y_prev):
            y_hofa_oom = dec_hofa[first_oom_idx]
            y_oom = 45.0 if np.isnan(y_hofa_oom) else max(20.0, y_hofa_oom * 0.25)
            ax.set_ylim(bottom=0.0)
            ax.annotate("", xy=(x_oom, y_oom), xytext=(x_prev, y_prev),
                        arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.5, ls="--", shrinkA=5, shrinkB=6))
            ax.plot(x_oom, y_oom, marker="X", color="#D55E00", markersize=9, markeredgecolor="white", markeredgewidth=0.8, zorder=5)
            ax.annotate("OOM", xy=(x_oom, y_oom), xytext=(-8, 0), textcoords="offset points",
                        color="#D55E00", fontsize=9.5, fontweight="bold", ha="right", va="center")

    # 2. KV Cache & State Footprint
    ax = axs_row[1]
    ax.plot(lens, mem_mha, **s_mha)
    ax.plot(lens, mem_hofa, **s_hyb)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(fmt_x)
    ax.set_xticks(lens)
    if show_xlabel:
        ax.set_xlabel("Context Length ($N$)")
    ax.set_ylabel("Attention State + KV Cache (GB)")
    if show_titles:
        ax.set_title("KV Cache & State Footprint")
    ax.grid(True, linestyle=":", alpha=0.3, color="#e0e0e0")

    valid_mem = [i for i, (m, h) in enumerate(zip(mem_mha, mem_hofa)) if not np.isnan(m) and not np.isnan(h) and m > 0]
    if valid_mem:
        i_last = valid_mem[-1]
        pct_mem = (mem_mha[i_last] - mem_hofa[i_last]) / mem_mha[i_last] * 100.0
        _annotate_pct(ax, lens[i_last], mem_hofa[i_last], mem_mha[i_last], f"-{pct_mem:.1f}%", side="left")
    if first_oom_idx is not None and first_oom_idx > 0:
        x_prev, y_prev = lens[first_oom_idx - 1], mem_mha[first_oom_idx - 1]
        x_oom = lens[first_oom_idx]
        if not np.isnan(y_prev) and y_prev > 0:
            y_extrap = y_prev * (x_oom / x_prev)
            y_cap = vram_gb * 1.12
            if y_extrap <= y_cap:
                x_tip, y_tip = x_oom, y_extrap
            else:
                # Follow exact O(N) linear KV scaling up to y_cap
                y_tip = y_cap
                x_tip = x_prev * (y_tip / y_prev)
            ax.set_ylim(top=max(y_tip, vram_gb) * 1.07)
            ax.axhline(vram_gb, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
            ax.text(lens[0], vram_gb * 1.015, f"{int(round(vram_gb))} GB VRAM", color="gray", fontsize=9, va="bottom", ha="left")
            ax.annotate("", xy=(x_tip, y_tip), xytext=(x_prev, y_prev),
                        arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.5, ls="--", shrinkA=5, shrinkB=2))
            ax.annotate("OOM", xy=(x_tip, y_tip), xytext=(-4, 2), textcoords="offset points",
                        color="#D55E00", fontsize=9.5, fontweight="bold", ha="right", va="bottom")

    # 3. Decoding Speedup
    ax = axs_row[2]
    dec_spd = [h / m if (not np.isnan(h) and not np.isnan(m) and m > 0) else float("nan") for h, m in zip(dec_hofa, dec_mha)]
    ax.plot(lens, dec_spd, marker="^", color="#009E73", linewidth=2.5, markersize=7)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(fmt_x)
    ax.set_xticks(lens)
    if show_xlabel:
        ax.set_xlabel("Context Length ($N$)")
    ax.set_ylabel(r"Speedup ($\times$ over MHA)")
    if show_titles:
        ax.set_title("Decoding Speedup over MHA")
    ax.grid(True, linestyle=":", alpha=0.3, color="#e0e0e0")

    for i in range(len(dec_spd) - 1):
        s0, s1 = dec_spd[i], dec_spd[i + 1]
        if not np.isnan(s0) and not np.isnan(s1) and s0 < 1.0 <= s1:
            t = (1.0 - s0) / (s1 - s0)
            cross_x = lens[i] * ((lens[i + 1] / lens[i]) ** t)
            cross_k = cross_x / 1024.0
            cross_str = f"~{cross_k:.1f}k" if cross_k < 10 else f"~{int(round(cross_k))}k"
            ax.annotate(f"Approx. crossover\n{cross_str}", xy=(cross_x, 1.0), xytext=(cross_x * 1.4, 0.90),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.1), fontsize=9.5, fontweight="bold", ha="left", va="top")
            break


def plot_full_decode(config: Optional[FullDecodeProfilingConfig] = None, res_b1=None, res_b4=None):
    """Render the 2x3 End-to-End 350M Decoding Profiling PDF from caches or provided tuples."""
    if config is None:
        config = FullDecodeProfilingConfig()
    if res_b1 is None:
        res_b1 = torch.load(os.path.join(CACHE_PATH, config.cache_b1_name), weights_only=False)
    if res_b4 is None:
        res_b4 = torch.load(os.path.join(CACHE_PATH, config.cache_b4_name), weights_only=False)

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 13,
        "axes.spines.top": True, "axes.spines.right": True, "font.family": "serif",
    })
    fig, axs = plt.subplots(2, 3, figsize=(18.5, 10.5))

    _plot_decode_row(axs[0], *res_b1, show_titles=True, show_xlabel=False, vram_gb=config.vram_gb)
    _plot_decode_row(axs[1], *res_b4, show_titles=False, show_xlabel=True, vram_gb=config.vram_gb)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.99), ncol=2, frameon=False, fontsize=13)
    plt.subplots_adjust(left=0.08, top=0.92, hspace=0.16, wspace=0.25)

    for row_idx, row_label in [(0, "B = 1"), (1, "B = 4")]:
        pos = axs[row_idx, 0].get_position()
        y_mid = 0.5 * (pos.y0 + pos.y1)
        fig.text(
            0.02, y_mid, row_label,
            fontsize=14, fontweight="bold", ha="center", va="center",
        )

    os.makedirs(os.path.dirname(config.out_pdf), exist_ok=True)
    fig.savefig(config.out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved full 350M 2x3 decoding plot to {config.out_pdf}")


def _load_or_run(cache_name: str, batch_size: int, cfg_mha, cfg_hofa, config: FullDecodeProfilingConfig, device, force_rerun: bool, save_results: bool):
    cache_path = os.path.join(CACHE_PATH, cache_name)
    if not force_rerun and os.path.exists(cache_path):
        print(f"Loading B={batch_size} decode cache from {cache_path}")
        return torch.load(cache_path, weights_only=False)
    print(f"\n=== Full 350M Decode Profiling (B={batch_size}) ===")
    dec_mha, mem_mha = _profile_decode_model(cfg_mha, config, batch_size, device)
    dec_hofa, mem_hofa = _profile_decode_model(cfg_hofa, config, batch_size, device)
    data = (list(config.seq_lengths), dec_mha, dec_hofa, mem_mha, mem_hofa)
    if save_results:
        os.makedirs(CACHE_PATH, exist_ok=True)
        torch.save(data, cache_path)
        print(f"Saved B={batch_size} decode cache to {cache_path}")
    return data


def run_full_decode_profiling(config: Optional[FullDecodeProfilingConfig] = None, force_rerun: bool = False, save_results: bool = True, plot_only: bool = False):
    """Run (or load cached) end-to-end 350M decoding profiling for B=1 and B=4 and generate the 2x3 PDF."""
    if config is None:
        config = FullDecodeProfilingConfig()
    if plot_only:
        plot_full_decode(config)
        return

    device = torch.device(config.device)
    cfg_mha = make_350M_MHA()
    cfg_hofa = make_350M_HOFA()

    res_b1 = _load_or_run(config.cache_b1_name, config.batch_sizes[0], cfg_mha, cfg_hofa, config, device, force_rerun, save_results)
    res_b4 = _load_or_run(config.cache_b4_name, config.batch_sizes[1], cfg_mha, cfg_hofa, config, device, force_rerun, save_results)
    plot_full_decode(config, res_b1=res_b1, res_b4=res_b4)
