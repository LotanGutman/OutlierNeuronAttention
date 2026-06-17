import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import gc
import os
import matplotlib.pyplot as plt
import numpy as np

from src.HybridOutlierFactorizedAttentionTrain import HybridOutlierFactorizedAttention
from benchmarks.benchmarks_configs import RecallExperimentConfig, CACHE_PATH
from fla.layers import DeltaNet, GatedLinearAttention as GLA
from mamba_ssm import Mamba2


def generate_zoology_mqar(batch_size, seq_len, vocab_size, num_kv_pairs, device):
    key_vocab_end = vocab_size // 2 # keys: 1..64
    val_vocab_start = vocab_size // 2 + 1 # values: 65..128
    val_vocab_end = vocab_size - 1

    context_size = num_kv_pairs * 2
    x = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    for b in range(batch_size):
        keys = torch.randperm(key_vocab_end, device=device)[:num_kv_pairs] + 1
        values = torch.randint(val_vocab_start, val_vocab_end + 1, (num_kv_pairs,), device=device)

        # KV pairs at start — preserves causal constraint
        x[b, 0:context_size:2] = keys
        x[b, 1:context_size:2] = values

        # Queries strictly after context
        query_positions = torch.randperm(seq_len - context_size, device=device)[:num_kv_pairs] + context_size
        query_order = torch.randperm(num_kv_pairs, device=device)
        x[b, query_positions] = keys[query_order]
        y[b, query_positions] = values[query_order]

    mask = (x == 0)
    x[mask] = torch.randint(1, val_vocab_start, (mask.sum().item(),), device=device)
    return x, y

def generate_sniah(batch_size, seq_len, vocab_size, depth_pct, device):
    num_keys = 128
    num_vals = 128
    seq = torch.randint(num_keys + num_vals + 1, vocab_size, (batch_size, seq_len), device=device)
    
    needle_key = torch.randint(1, num_keys + 1, (batch_size, 1), device=device)
    needle_value = torch.randint(num_keys + 1, num_keys + num_vals + 1, (batch_size, 1), device=device)
    
    needle_idx = int(depth_pct * (seq_len // 2 - 2)) * 2
    seq[:, needle_idx:needle_idx+1] = needle_key
    seq[:, needle_idx+1:needle_idx+2] = needle_value
    
    x = seq.clone()
    x[:, -1:] = needle_key
    y = torch.full_like(x, -100)
    y[:, -1:] = needle_value
    return x, y


class StandardMHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        q = self.W_q(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(y)

class FLAWrapper(nn.Module):
    def __init__(self, fla_layer):
        super().__init__()
        self.layer = fla_layer

    def forward(self, x):
        out = self.layer(x)
        return out[0] if isinstance(out, tuple) else out

class AttentionType(Enum):
    MHA = "mha"
    HOFA = "hofa"
    DELTA = "delta"
    GLA = "gla"
    MAMBA = "mamba"

def build_attention(attn_type, model_cfg):
    d_model = model_cfg.d_model
    num_heads = model_cfg.num_heads
    if attn_type == AttentionType.MHA:
        return StandardMHA(d_model, num_heads)
    if attn_type == AttentionType.HOFA:
        return HybridOutlierFactorizedAttention(model_cfg)
    if attn_type == AttentionType.DELTA:
        return FLAWrapper(DeltaNet(hidden_size=d_model, num_heads=num_heads))
    if attn_type == AttentionType.GLA:
        return FLAWrapper(GLA(hidden_size=d_model, num_heads=num_heads))
    if attn_type == AttentionType.MAMBA:
        return Mamba2(
            d_model=d_model,
            d_state=2 ** int(np.log2(d_model // num_heads)),
            d_conv=4,
            expand=2
        )
    raise ValueError(f"Unknown attention type: {attn_type}")

class GenericBenchmarkLM(nn.Module):
    def __init__(self, vocab_size, d_model, attn_type, num_heads=8, num_layers=2, model_cfg=None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(131072, d_model)
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln_1': nn.LayerNorm(d_model),
                'attn': build_attention(attn_type, model_cfg),
                'ln_2': nn.LayerNorm(d_model),
                'mlp': nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
            }) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight

    def forward(self, x, targets=None, return_loss=True):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        
        x = self.token_embedding(x) + self.pos_embedding(positions)
        
        for block in self.blocks:
            x = x + block['attn'](block['ln_1'](x))
            x = x + block['mlp'](block['ln_2'](x))
            
        x = self.ln_f(x)
        
        if targets is not None:
            valid_mask = (targets != -100)
            x_valid = x[valid_mask]
            logits_valid = self.lm_head(x_valid)
            
            if return_loss:
                targets_valid = targets[valid_mask]
                loss = F.cross_entropy(logits_valid, targets_valid)
                return None, loss
            else:
                return logits_valid, None
        else:
            logits = self.lm_head(x)
            return logits, None


def adjust_learning_rate(optimizer, step, total_steps, base_lr, warmup_steps):
    lr = base_lr * min(1.0, step / warmup_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_and_eval(model, gen_func, gen_kwargs, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    model.train()
    device = gen_kwargs["device"]
    use_autocast = config.use_mixed_precision and device == "cuda"
    warmup_steps = int(0.1 * config.train_steps)
    
    from collections import defaultdict
    alpha_history = defaultdict(list)
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    consecutive_perfect_acc = 0

    for i in range(config.train_steps):
        adjust_learning_rate(optimizer, i, config.train_steps, config.learning_rate, warmup_steps)
        optimizer.zero_grad()
        x, y = gen_func(**gen_kwargs)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            _, loss = model(x, targets=y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        train_loss_history.append((i + 1, loss.item()))

        if (i + 1) % config.print_every == 0 or i == 0 or (i + 1) == config.train_steps:
            model.eval()
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                    x_val, y_val = gen_func(**gen_kwargs)
                    logits_val, _ = model(x_val, targets=y_val, return_loss=False)
                
                targets_valid = y_val[y_val != -100]
                val_loss = F.cross_entropy(logits_val, targets_valid).item()
                val_acc = (logits_val.argmax(dim=-1) == targets_valid).float().mean().item()
                
                val_loss_history.append((i + 1, val_loss))
                val_acc_history.append((i + 1, val_acc))
                
                query_mask = (y_val != -100)
                context_mask = (y_val == -100)
    
                layer_stats = ""
                for block_idx, block in enumerate(model.blocks):
                    if hasattr(block['attn'], 'last_mix_g'):
                        mix_g = block['attn'].last_mix_g.squeeze(-1) # (B, H, N)
                        
                        query_val = mix_g[query_mask.unsqueeze(1).expand(-1, mix_g.shape[1], -1)].mean().item()
                        context_val = mix_g[context_mask.unsqueeze(1).expand(-1, mix_g.shape[1], -1)].mean().item()
                        
                        alpha_history.setdefault(f'layer_{block_idx}_query', []).append((i + 1, query_val))
                        alpha_history.setdefault(f'layer_{block_idx}_context', []).append((i + 1, context_val))
                        
                        layer_stats += f" | L{block_idx} Q:{query_val:.2f} C:{context_val:.2f}"

            model.train()
            print(f"\r      Step {i + 1:4d}/{config.train_steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%{layer_stats}")

            # a simple early stopping mechanism to save time on models that converge quickly
            if val_acc >= 0.995:
                consecutive_perfect_acc += 1
            else:
                consecutive_perfect_acc = 0
            
            if consecutive_perfect_acc >= 2:
                print("\n      Early stopping: achieved >99.5% accuracy for 2 consecutive evaluations.")
                break
        else:
            # Print dynamic progress bar
            print(f"\r      Step {i + 1:4d}/{config.train_steps}", end="", flush=True)

    print()
    # Print final gate status if HOFA
    for i, block in enumerate(model.blocks):
        if hasattr(block['attn'], 'last_mix_g'):
            
            import matplotlib.pyplot as plt
            import os
            # Grab sequence gate values for the first batch, first head
            mix_g_snapshot = block['attn'].last_mix_g[0, 0, :].detach().cpu().numpy().squeeze()
            
            plt.style.use('seaborn-v0_8-paper')
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(mix_g_snapshot, color='#1f77b4', linewidth=1.5, label='Mix Gate (α)')
            
            ax.set_title(f'Outlier Routing Snapshot: Layer {i} Head 0 Activation', fontsize=12, fontweight='bold')
            ax.set_xlabel('Sequence Position', fontsize=11)
            ax.set_ylabel('Mix Gate α', fontsize=11)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            
            os.makedirs("data/plots/mqar", exist_ok=True)
            plt.tight_layout()
            plt.savefig(f"data/plots/mqar/outlier_snapshot_L{i}.pdf", format='pdf', dpi=300)
            plt.close()
            
    model.eval()
    correct = 0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(config.num_eval_batches):
            x, y = gen_func(**gen_kwargs)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                logits_valid, _ = model(x, targets=y, return_loss=False)

            preds = logits_valid.argmax(dim=-1)
            targets_valid = y[y != -100]
            correct += (preds == targets_valid).sum().item()
            total_tokens += targets_valid.numel()

    return {
        'accuracy': correct / total_tokens,
        'correct_tokens': correct,
        'total_tokens': total_tokens,
        'alpha_history': alpha_history,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'val_acc_history': val_acc_history,
        'final_val_loss': val_loss if 'val_loss' in locals() else None,
        'consecutive_perfect_acc': consecutive_perfect_acc
    }

def plot_alphas(alpha_histories, save_plot=True):
    if not alpha_histories or not alpha_histories.get(list(alpha_histories.keys())[0]):
        return
        
    os.makedirs("data/plots/mqar", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # We will just plot alpha evolution for the max density to avoid clutter
    max_density = max(alpha_histories.keys())
    history = alpha_histories[max_density]
    
    for key, records in history.items():
        if not records: continue
        steps, alphas = zip(*records)
        linestyle = '-' if 'query' in str(key) else '--'
        ax.plot(steps, alphas, marker='o', linestyle=linestyle, label=f'{key}')
        
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mix Gate Value')
    ax.set_title(f'Dynamic Mix Gate Evolution (Density: {max_density})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if save_plot:
        plot_path = "data/plots/mqar/alpha_evolution.pdf"
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    plt.close()


def plot_results(results, save_plot=True):
    os.makedirs("data/plots/mqar", exist_ok=True)
    densities = sorted(list(results.keys()))
    if not densities:
        return
    models = list(results[densities[0]].keys())
    
    # Use an academic style for better default font weights
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Distinct markers for each model to make it readable in black & white
    markers = {'HOFA (r=34)': 'o', 'HOFA (r=32)': 'x', 'MHA': 's', 'Gated DeltaNet': '^', 'GLA': 'D'}
    colors = {'HOFA (r=34)': '#1f77b4', 'HOFA (r=32)': '#9467bd', 'MHA': '#ff7f0e', 'Gated DeltaNet': '#2ca02c', 'GLA': '#d62728'}
    
    for model in models:
        accs = []
        for d in densities:
            res = results[d][model]
            acc = res['accuracy'] if isinstance(res, dict) else res
            accs.append(acc * 100)
            
        ax.plot(
            densities, 
            accs, 
            marker=markers.get(model, 'o'), 
            color=colors.get(model, '#000000'),
            linewidth=2.5, 
            markersize=8,
            label=model
        )
        
    ax.set_xlabel('Number of KV Pairs in Context (Density)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Exact-Match Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('High-Density MQAR: Capacity Collapse', fontsize=14, fontweight='bold')
    
    # Crucial: Log scale for X-axis since densities grow exponentially
    ax.set_xscale('log', base=2)
    ax.set_xticks(densities)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()) # Prevent scientific notation
    
    # Lock Y-axis from 0 to 105 so the 100% lines don't hit the absolute roof
    ax.set_ylim(-5, 105)
    
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.7, which='both')
    
    plt.tight_layout()
    if save_plot:
        plot_path = "data/plots/mqar/recall_accuracy.pdf"
        plt.savefig(plot_path, format='pdf', dpi=300)
        print(f"Plot saved to {plot_path}")
    plt.close()

def run_recall_experiment(config: RecallExperimentConfig = RecallExperimentConfig(), force_rerun=True, save_results=True, continue_from_cache=False):
    cache_path = os.path.join(CACHE_PATH, config.cache_file_name)
    interm_cache_path = os.path.join(CACHE_PATH, "intermediate_" + config.cache_file_name)
    device = config.device
    torch.manual_seed(config.seed)
    print("--- Starting Zoology Exact-Match MQAR Sweep ---")
    
    model_names = ["HOFA (r=34)", "HOFA (r=32)", "MHA", "Gated DeltaNet", "GLA", "Mamba"]
    
    if os.path.exists(cache_path) and not force_rerun:
        print(f"Loading cached results from {cache_path}")
        results = torch.load(cache_path, weights_only=False)
    else:
        results = {d: {} for d in config.densities}
        
        if continue_from_cache and os.path.exists(interm_cache_path):
            print(f"Loading intermediate cached results from {interm_cache_path}")
            loaded_results = torch.load(interm_cache_path, weights_only=False)
            for d in loaded_results:
                if d in results:
                    results[d].update(loaded_results[d])

        all_alpha_histories = {}
        # Start from the hardest problem (highest density) to fail fast on capacity limits
        for density in reversed(config.densities):
            for name in model_names:
                if continue_from_cache and name in results[density]:
                    print(f"\nSkipping {name} (Density: {density}) - already in intermediate cache.")
                    continue
                    
                print(f"\nTraining {name} (Density: {density})...")
                
                # Resetting the seed here ensures every model sees the *exact same* 
                # sequence of training data, providing the fairest possible comparison.
                torch.manual_seed(config.seed)
                np.random.seed(config.seed)
                
                from dataclasses import replace
                model_cfg = config.model_config
                if name == "HOFA (r=34)":
                    model_cfg = replace(model_cfg, r=34)
                elif name == "HOFA (r=32)":
                    model_cfg = replace(model_cfg, r=32)
                
                attn_type_map = {
                    "MHA": AttentionType.MHA,
                    "HOFA (r=34)": AttentionType.HOFA,
                    "HOFA (r=32)": AttentionType.HOFA,
                    "Gated DeltaNet": AttentionType.DELTA,
                    "GLA": AttentionType.GLA,
                    "Mamba": AttentionType.MAMBA
                }
                attn_type = attn_type_map[name]
                
                model = GenericBenchmarkLM(
                    config.vocab_size, 
                    model_cfg.d_model, 
                    attn_type, 
                    num_heads=model_cfg.num_heads, 
                    num_layers=model_cfg.num_layers, 
                    model_cfg=model_cfg
                ).to(device)
                
                info = train_and_eval(
                    model,
                    generate_zoology_mqar,
                    {
                        "batch_size": config.batch_size,
                        "seq_len": config.seq_len,
                        "vocab_size": config.vocab_size,
                        "num_kv_pairs": density,
                        "device": device,
                    },
                    config,
                )
                
                acc = info['accuracy']
                alpha_hist = info['alpha_history']
                print(f">>> MQAR {density} | {name} Final Accuracy: {acc*100:.1f}%")
                
                from dataclasses import asdict
                info['model_config'] = asdict(model_cfg)
                results[density][name] = info
                
                if name == "HOFA (r=32)":
                    all_alpha_histories[density] = alpha_hist
                
                model.to('cpu')
                del model
                torch.cuda.empty_cache()
                gc.collect()
                
                if save_results:
                    os.makedirs(os.path.dirname(interm_cache_path), exist_ok=True)
                    torch.save(results, interm_cache_path)
                
        if save_results:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(results, cache_path)
            
    # Re-extract all_alpha_histories for plotting (runs for both fresh runs and full cache loads)
    all_alpha_histories = {}
    for density in config.densities:
        if density in results:
            if "HOFA (r=34)" in results[density] and 'alpha_history' in results[density]["HOFA (r=34)"]:
                all_alpha_histories[density] = results[density]["HOFA (r=34)"]['alpha_history']
            elif "HOFA (r=32)" in results[density] and 'alpha_history' in results[density]["HOFA (r=32)"]:
                all_alpha_histories[density] = results[density]["HOFA (r=32)"]['alpha_history']

    plot_results(results, save_plot=save_results)
    plot_alphas(all_alpha_histories, save_plot=save_results)
    return results

if __name__ == "__main__":
    config = RecallExperimentConfig()
    run_recall_experiment(config, force_rerun=True, continue_from_cache=True)