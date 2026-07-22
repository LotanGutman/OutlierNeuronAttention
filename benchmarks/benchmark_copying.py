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
        {'params': decay_params, 'weight_decay': config.weight_decay},  # Backbone gets regularized for stability
    ]
    if len(no_decay_params) > 0:
        optim_groups.append({'params': no_decay_params, 'weight_decay': 0.0}) # Exact pathway stays pure

    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, fused=True)
    device = config.device
    use_autocast = config.use_mixed_precision and device == "cuda"
    warmup_steps = int(0.05 * config.train_steps)
    delim_token = config.vocab_size - 1

    model.train()
    history = {'loss': [], 'exact_seq_acc': [], 'per_token_acc': [], 'step': [], 'mix_bias': []}
    consecutive_perfect_acc = 0
    start_step = 0
    
    if checkpoint_dir is not None:
        metadata = load_checkpoint(model, optimizer, model_name, -1, checkpoint_dir)
        if metadata is not None:
            print(f"\n      Resuming {model_name} from step {metadata['step']}")
            start_step = metadata['step']
            history = metadata.get('history', {'loss': [], 'exact_seq_acc': [], 'per_token_acc': [], 'step': [], 'mix_bias': []})
            consecutive_perfect_acc = metadata.get('consecutive_perfect_acc', 0)
            
            # If we achieved near perfect seq accuracy early, we stopped
            if start_step >= config.train_steps or consecutive_perfect_acc >= 2 or metadata.get('stopped_early', False):
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
            
            print(f"\r      Step {i + 1:5d}/{config.train_steps} | Loss: {loss.item():.4f} | Seq Acc: {exact_acc:.1f}% | Tok Acc: {pt_acc:.1f}%")
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
            
            if exact_acc >= 99.5 and pt_acc >= 99.5 and (i > warmup_steps):
                consecutive_perfect_acc += 1
            else:
                consecutive_perfect_acc = 0
                
            if consecutive_perfect_acc >= 2:
                print("\n      Early stopping: achieved >99.5% exact sequence accuracy for 2 consecutive evaluations.")
                break
                
            if checkpoint_dir is not None:
                metadata = {
                    'model_name': model_name,
                    'density': -1,
                    'step': i + 1,
                    'history': history,
                    'consecutive_perfect_acc': consecutive_perfect_acc
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
            'step': config.train_steps if early_stopper.stop_requested else (i + 1 if 'i' in locals() else 0),
            'history': history,
            'consecutive_perfect_acc': consecutive_perfect_acc,
            'stopped_early': early_stopper.stop_requested
        }
        save_checkpoint(model, optimizer, final_metadata, checkpoint_dir)

    return history

def run_copying_experiment(base_config = CopyingExperimentConfig()):    
    device = base_config.device
    torch.manual_seed(base_config.seed)
    
    models_to_test = base_config.models_to_test
    
    pattern_len = 8
    gap_lengths = [128, 512, 1024]
    
    trendline_results = {name: [] for name, _, _ in models_to_test}
    
    convergence_data_1024 = {}

    for gap_len in gap_lengths:
        print(f"\n{'='*50}\nEVALUATING GAP LENGTH: {gap_len}\n{'='*50}")
        
        seq_len = 2 * pattern_len + gap_len + 1
        
        all_histories = {}
        for name, attn_type, r in models_to_test:
            if r is not None:
                base_config.model_config.r = r
            
            base_config.seq_len = seq_len
            
            # Resetting the seed here ensures every model sees the *exact same* sequence of training data, providing the fairest possible comparison.
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
            all_histories[name] = history
            
            # Save final model state dict for easy loading later
            torch.save(model.state_dict(), f"{checkpoint_dir}/final_model.pt")
            
            del model
            torch.cuda.empty_cache()
            
            max_acc = max(history['exact_seq_acc']) if len(history['exact_seq_acc']) > 0 else 0.0
            trendline_results[name].append(max_acc)
            
        # Plotting Convergence for this gap length
        os.makedirs("data/plots/copying", exist_ok=True)
        plt.figure(figsize=(10, 6))
        for name, history in all_histories.items():
            if len(history['exact_seq_acc']) == 0: continue
            steps = np.arange(1, len(history['exact_seq_acc']) + 1) * base_config.print_every
            steps[0] = 1
            plt.plot(steps, history['exact_seq_acc'], label=name, marker='o', markersize=3)
            
        plt.title(f"Copying Task Convergence (Gap Len = {gap_len})")
        plt.xlabel("Training Steps")
        plt.ylabel("Exact Sequence Accuracy (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = f"data/plots/copying/convergence_gap_{gap_len}.pdf"
        plt.savefig(plot_path, bbox_inches='tight', format='pdf', dpi=300)
        plt.close()
        print(f"\nConvergence plot saved to {plot_path}")

    # --- PLOTTING ---
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    plot_dir = "data/plots/benchmarks/copying"
    os.makedirs(plot_dir, exist_ok=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_to_test)))

    # 2. Trendline plot
    plt.figure(figsize=(10, 6))
    for i, (name, _, _) in enumerate(models_to_test):
        accs = trendline_results[name]
        plt.plot(gap_lengths, accs, marker='o', label=name, color=colors[i], linewidth=2, markersize=8)
    
    plt.title("Final Exact Sequence Accuracy vs Gap Length")
    plt.xlabel("Gap Length")
    plt.ylabel("Exact Sequence Accuracy (%)")
    plt.xticks(gap_lengths)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "accuracy_vs_gap.pdf"))
    plt.close()

    # --- CLI SUMMARY TABLE ---
    print("\n" + "="*60)
    print("FINAL EXACT SEQUENCE ACCURACY (%)")
    print("="*60)
    header = f"{'Model':<15} | " + " | ".join([f"Gap {g:<4}" for g in gap_lengths])
    print(header)
    print("-" * len(header))
    for name, _, _ in models_to_test:
        accs = trendline_results[name]
        row_str = f"{name:<15} | " + " | ".join([f"{a:>8.1f}" for a in accs])
        print(row_str)
    print("="*60)

if __name__ == "__main__":
    run_copying_experiment()
