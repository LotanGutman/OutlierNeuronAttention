import torch
import torch.nn.functional as F
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from src.config import ModelConfig
from src.modules.benchmark_utils import GenericBenchmarkLM, AttentionType, adjust_learning_rate
from benchmarks.benchmarks_configs import CopyingExperimentConfig
from src.modules.checkpointing import save_checkpoint, load_checkpoint
from interrupt_util.interrupts import GracefulInterruptHandler

def generate_copying_batch(batch_size, pattern_len, gap_len, vocab_size, delim_token, device):
    """
    Returns (input_ids, target_ids, loss_mask), each shape (B, total_len - 1).
    Sequence layout before shifting: [pattern (P)] [noise (G)] [delim (1)] [pattern again (P)]
    Loss is computed ONLY on the P positions after the delimiter (next-token prediction
    of the repeated pattern).
    """
    usable_vocab = vocab_size - 1  # reserve delim_token = vocab_size - 1

    pattern = torch.randint(0, usable_vocab, (batch_size, pattern_len), device=device)
    noise = torch.randint(0, usable_vocab, (batch_size, gap_len), device=device)
    delim = torch.full((batch_size, 1), delim_token, device=device)

    full_seq = torch.cat([pattern, noise, delim, pattern], dim=1)

    input_ids = full_seq[:, :-1].contiguous()
    target_ids = full_seq[:, 1:].contiguous()

    loss_mask = torch.zeros_like(target_ids, dtype=torch.bool)
    copy_start = pattern_len + gap_len + 1 - 1  # -1 accounts for the shift
    loss_mask[:, copy_start:copy_start + pattern_len] = True

    return input_ids, target_ids, loss_mask

def exact_sequence_accuracy(logits, target_ids, loss_mask):
    preds = logits.argmax(dim=-1)
    # gather only the masked (copy-target) positions per sequence
    correct_per_seq = []
    for b in range(preds.shape[0]):
        mask_b = loss_mask[b]
        pred_seq = preds[b][mask_b]
        tgt_seq = target_ids[b][mask_b]
        correct_per_seq.append(torch.equal(pred_seq, tgt_seq))
    return sum(correct_per_seq) / len(correct_per_seq)

def per_token_accuracy(logits, target_ids, loss_mask):
    preds = logits.argmax(dim=-1)
    correct = (preds == target_ids) & loss_mask
    return correct.sum().item() / max(loss_mask.sum().item(), 1)

def train_copying(model, config, model_name, pattern_len, gap_len, checkpoint_dir=None):
    print(f"\n--- Training {model_name} on Copying (Gap={gap_len}) ---")
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        # The exact pathway Q/K projections remain unregularized (wd=0.0) 
        # to allow infinite gamma scaling per theoretical requirement.
        if config.disable_weight_decay_for_attention and ('W_q' in name or 'W_k' in name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
    ]
    if len(no_decay_params) > 0:
        optim_groups.append({'params': no_decay_params, 'weight_decay': 0.0})

    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, fused=True)
    device = config.device
    use_autocast = config.use_mixed_precision and device == "cuda"
    warmup_steps = int(0.05 * config.train_steps)
    delim_token = config.vocab_size - 1

    model.train()
    history = {'loss': [], 'exact_seq_acc': [], 'per_token_acc': [], 'step': [], 'mix_bias': []}
    start_step = 0
    
    if checkpoint_dir is not None:
        metadata = load_checkpoint(model, optimizer, model_name, -1, checkpoint_dir)
        if metadata is not None:
            print(f"\n      Resuming {model_name} from step {metadata['step']}")
            start_step = metadata['step']
            history = metadata.get('history', {'loss': [], 'exact_seq_acc': [], 'per_token_acc': [], 'step': [], 'mix_bias': []})
            
            if start_step >= config.train_steps:
                print(f"      {model_name} already completed or was skipped previously. Skipping.")
                return history

    early_stopper = GracefulInterruptHandler()
    early_stopper.attach()
    
    for i in range(start_step, config.train_steps):
        adjust_learning_rate(optimizer, i, config.train_steps, config.learning_rate, warmup_steps)
        optimizer.zero_grad()
        
        x, y, loss_mask = generate_copying_batch(config.batch_size, pattern_len, gap_len, config.vocab_size, delim_token, device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            # We must get full logits to run exact_sequence_accuracy correctly
            logits, _ = model(x, targets=None)
            logits_valid = logits[loss_mask]
            targets_valid = y[loss_mask]
            loss = F.cross_entropy(logits_valid, targets_valid)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        if (i + 1) % config.print_every == 0 or i == 0 or (i + 1) == config.train_steps:
            model.eval()
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                    logits_eval, _ = model(x, targets=None)
                
                exact_acc = exact_sequence_accuracy(logits_eval, y, loss_mask) * 100.0
                pt_acc = per_token_accuracy(logits_eval, y, loss_mask) * 100.0
            model.train()
            
            bias_str = ""
            norm_str = ""
            if hasattr(model, 'blocks') and len(model.blocks) > 0:
                attn = model.blocks[0]['attn']
                if hasattr(attn, 'W_q') and hasattr(attn, 'W_k'):
                    try:
                        H = attn.num_heads
                        D_head = getattr(attn, 'd_head', attn.W_q.weight.shape[1] // H)
                        W_q_w = attn.W_q.weight.view(H, D_head, -1)
                        W_k_w = attn.W_k.weight.view(H, D_head, -1)
                        norm_prod = (W_q_w.norm(p=2, dim=2) * W_k_w.norm(p=2, dim=2)).mean().item()
                        norm_str = f" | WQ*WK Norm: {norm_prod:.2f}"
                    except Exception:
                        pass
                        
                if hasattr(attn, 'mix_proj') and hasattr(attn.mix_proj, 'bias') and attn.mix_proj.bias is not None:
                    bias_str = f" | Mix Bias: {attn.mix_proj.bias.mean().item():.4f}"

            print(f"\r      Step {i + 1:5d}/{config.train_steps} | Loss: {loss.item():.4f} | Seq Acc: {exact_acc:.1f}% | Tok Acc: {pt_acc:.1f}%{bias_str}{norm_str}")
            history['loss'].append(loss.item())
            history['exact_seq_acc'].append(exact_acc)
            history['per_token_acc'].append(pt_acc)
            history['step'].append(i + 1)
            
            if hasattr(model, 'blocks'):
                layer_biases = []
                for b in model.blocks:
                    attn_layer = b['attn']
                    if hasattr(attn_layer, 'mix_proj') and hasattr(attn_layer.mix_proj, 'bias') and attn_layer.mix_proj.bias is not None:
                        layer_biases.append(attn_layer.mix_proj.bias.detach().cpu().numpy().tolist())
                if layer_biases:
                    history.setdefault('mix_bias', []).append(layer_biases)
            
            if checkpoint_dir is not None:
                metadata = {
                    'model_name': model_name,
                    'density': -1,
                    'step': i + 1,
                    'history': history,
                }
                save_checkpoint(model, optimizer, metadata, checkpoint_dir)
                
            if early_stopper.stop_requested:
                break
        else:
            print(f"\r      Step {i + 1:5d}/{config.train_steps}", end="", flush=True)
            if early_stopper.stop_requested:
                break

    early_stopper.detach()
    print()
    
    if checkpoint_dir is not None:
        final_metadata = {
            'model_name': model_name,
            'density': -1,
            'step': i + 1 if 'i' in locals() else 0,
            'history': history,
            'stopped_early': early_stopper.stop_requested
        }
        save_checkpoint(model, optimizer, final_metadata, checkpoint_dir)

    return history

def run_copying_experiment(base_config=None):    
    if base_config is None:
        base_config = CopyingExperimentConfig()
        
    device = base_config.device
    torch.manual_seed(base_config.seed)
    
    models_to_test = base_config.models_to_test
    pattern_len = base_config.pattern_len
    gap_lengths = base_config.gap_lengths

    for gap_len in gap_lengths:
        print(f"\n{'='*50}\nEVALUATING GAP LENGTH: {gap_len}\n{'='*50}")
        seq_len = 2 * pattern_len + gap_len + 1
        
        for name, attn_type, r in models_to_test:
            if r is not None:
                base_config.model_config.r = r
            
            base_config.seq_len = seq_len
            
            # Resetting the seed here ensures every model sees the *exact same* sequence of training data
            torch.manual_seed(base_config.seed)
            np.random.seed(base_config.seed)
            
            model = GenericBenchmarkLM(
                vocab_size=base_config.vocab_size,
                d_model=base_config.model_config.d_model,
                attn_type=attn_type,
                num_heads=base_config.model_config.num_heads,
                num_layers=base_config.model_config.num_layers,
                model_cfg=base_config.model_config
            ).to(device)
            
            checkpoint_dir = f"data/copying_models/gap_{gap_len}/copying_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            history = train_copying(
                model=model, 
                config=base_config, 
                model_name=name, 
                pattern_len=pattern_len, 
                gap_len=gap_len, 
                checkpoint_dir=checkpoint_dir
            )
            
            # Save final model state dict for easy loading later
            torch.save(model.state_dict(), f"{checkpoint_dir}/final_model.pt")
            
            del model
            torch.cuda.empty_cache()

def plot_copying_experiment(base_config=None):
    if base_config is None:
        base_config = CopyingExperimentConfig()
        
    models_to_test = base_config.models_to_test
    gap_lengths = base_config.gap_lengths
    trendline_results = {name: [] for name, _, _ in models_to_test}
    
    plot_dir = "data/plots/copying"
    os.makedirs(plot_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    
    for gap_len in gap_lengths:
        all_histories = {}
        for name, attn_type, r in models_to_test:
            dir_name = f"copying_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}"
            ckpt_path = f"data/copying_models/gap_{gap_len}/{dir_name}/checkpoint.pt"
            
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                history = ckpt.get('metadata', {}).get('history', {})
                exact_accs = history.get('exact_seq_acc', [])
                max_acc = max(exact_accs) if len(exact_accs) > 0 else 0.0
                all_histories[name] = history
            else:
                max_acc = 0.0
                all_histories[name] = {'exact_seq_acc': []}
                
            trendline_results[name].append(max_acc)
            
        # Plot side-by-side Accuracy and Loss convergence for this gap length
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        has_conv_data = False
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_to_test)))
        
        random_loss = np.log(base_config.vocab_size)
        
        for i, (name, _, _) in enumerate(models_to_test):
            history = all_histories.get(name, {})
            accs = history.get('exact_seq_acc', [])
            losses = history.get('loss', [])
            if len(accs) == 0: continue
            
            steps = np.arange(1, len(accs) + 1) * base_config.print_every
            steps[0] = 1
            
            ax1.plot(steps, accs, label=name, color=colors[i], marker='o', markersize=3, linewidth=2)
            ax2.plot(steps, losses, label=name, color=colors[i], marker='o', markersize=3, linewidth=2)
            has_conv_data = True
            
        if has_conv_data:
            # Panel A: Accuracy
            ax1.set_title(f"(a) Exact Sequence Accuracy (Gap = {gap_len})", fontsize=12, pad=10)
            ax1.set_xlabel("Training Steps", fontsize=11)
            ax1.set_ylabel("Accuracy (%)", fontsize=11)
            ax1.set_ylim(-2, 103)
            ax1.grid(True, alpha=0.3)
            
            # Panel B: Loss
            ax2.axhline(y=random_loss, color='black', linestyle='--', linewidth=1.5, label=fr'Random Guess ($\ln V \approx {random_loss:.2f}$)')
            ax2.set_title(f"(b) Training Loss (Gap = {gap_len})", fontsize=12, pad=10)
            ax2.set_xlabel("Training Steps", fontsize=11)
            ax2.set_ylabel("Cross Entropy Loss", fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            # Shared legend at top center matching profiling style
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            by_label = dict(zip(labels1 + labels2, handles1 + handles2))
            fig.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False, fontsize=11)
            
            plt.tight_layout()
            plot_path = f"{plot_dir}/convergence_gap_{gap_len}.pdf"
            plt.savefig(plot_path, bbox_inches='tight', format='pdf', dpi=300)
            plt.close()
            print(f"Convergence plot (Accuracy & Loss with shared legend) saved to {plot_path}")
        else:
            plt.close()

    # --- TRENDLINE PLOT ---
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_to_test)))
    
    has_trend_data = False
    for i, (name, _, _) in enumerate(models_to_test):
        accs = trendline_results[name]
        if any(a > 0 for a in accs):
            has_trend_data = True
        plt.plot(gap_lengths, accs, marker='o', label=name, color=colors[i], linewidth=2, markersize=8)
    
    if has_trend_data:
        plt.title("Sequential Copying Accuracy vs Gap Length", fontsize=13, pad=12)
        plt.xlabel("Gap Length", fontsize=11)
        plt.ylabel("Exact Sequence Accuracy (%)", fontsize=11)
        plt.xticks(gap_lengths)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='best', frameon=True)
        
        plot_path = f"{plot_dir}/accuracy_vs_gap.pdf"
        plt.savefig(plot_path, bbox_inches='tight', format='pdf', dpi=300)
        plt.close()
        print(f"\nSequential Copying trendline plot successfully saved to {plot_path}")
    else:
        print("No copying checkpoint data found to plot.")
        plt.close()

    # --- CLI SUMMARY TABLE ---
    print("\n" + "="*60)
    print("FINAL EXACT SEQUENCE ACCURACY (%)")
    print("="*60)
    header = f"{'Model':<18} | " + " | ".join([f"Gap {g:<4}" for g in gap_lengths])
    print(header)
    print("-" * len(header))
    for name, _, _ in models_to_test:
        accs = trendline_results[name]
        row_str = f"{name:<18} | " + " | ".join([f"{a:>8.1f}" for a in accs])
        print(row_str)
    print("="*60)

if __name__ == "__main__":
    run_copying_experiment()
