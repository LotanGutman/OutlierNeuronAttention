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

from dataclasses import dataclass, field

from src.config import ModelConfig
from src.modules.benchmark_utils import GenericBenchmarkLM, AttentionType, adjust_learning_rate
from benchmarks.benchmarks_configs import InductionExperimentConfig
from src.modules.checkpointing import save_checkpoint, load_checkpoint
from interrupt_util.interrupts import GracefulInterruptHandler

def generate_induction_seqs(batch_size, seq_len, vocab_size, device, pattern_len=None, pin_to_end=False):
    """
    Generates sequences for the Induction Head task: [A, B, ..., A, B]
    A pattern of random tokens is placed at index 0 and repeated later in the sequence.
    """
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    
    if pattern_len is None:
        max_len = min(128, max(2, seq_len // 4))
        min_len = min(32, max_len)
    else:
        min_len = pattern_len
        max_len = pattern_len
    
    for b in range(batch_size):
        p_len = torch.randint(min_len, max_len + 1, (1,)).item() if max_len > min_len else max_len
        pattern = torch.randint(0, vocab_size, (p_len,), device=device)
        
        # Place pattern at start
        x[b, :p_len] = pattern
        
        # Place repeat in second half or pinned to end
        if pin_to_end:
            start_idx = seq_len - p_len
        else:
            low_idx = seq_len // 2
            high_idx = seq_len - p_len
            start_idx = torch.randint(low_idx, high_idx + 1, (1,)).item() if high_idx >= low_idx else high_idx
            
        x[b, start_idx:start_idx+p_len] = pattern
        
        # Set targets for the copied pattern
        y[b, start_idx:start_idx+p_len-1] = pattern[1:]
        
    return x, y

def evaluate_induction_accuracy(
    model, 
    config, 
    seq_len=None, 
    eval_samples=500, 
    eval_batch_size=32, 
    device=None,
    pattern_len=32,
    pin_to_end=True,
    seed=None
):
    """
    Evaluates model accuracy on held-out induction sequences.
    Returns:
        dict with:
            'avg_token_acc': float,   # Average per-token accuracy across all target tokens (%)
            'first_token_acc': float, # Accuracy on the first target token (%)
            'full_match_acc': float   # Exact match accuracy across the full target block (%)
    """
    if seq_len is None:
        seq_len = config.seq_len
    if device is None:
        device = getattr(config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
    use_autocast = getattr(config, 'use_mixed_precision', True) and device == "cuda"
    
    # Store and set RNG if seed specified
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() and device == "cuda" else None
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    was_training = model.training
    model.eval()
    
    total_tokens = 0
    correct_tokens = 0
    first_token_correct = 0
    full_match_correct = 0
    total_seqs = 0
    
    with torch.no_grad():
        for b_start in range(0, eval_samples, eval_batch_size):
            sub_size = min(eval_batch_size, eval_samples - b_start)
            x_sub, y_sub = generate_induction_seqs(
                sub_size, seq_len, config.vocab_size, device,
                pattern_len=pattern_len, pin_to_end=pin_to_end
            )
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                logits_sub, _ = model(x_sub)
                
            preds = logits_sub.argmax(dim=-1)
            
            for b in range(sub_size):
                mask_b = (y_sub[b] != -100)
                if mask_b.any():
                    target_tokens = y_sub[b][mask_b]
                    pred_tokens = preds[b][mask_b]
                    
                    is_correct = (pred_tokens == target_tokens)
                    correct_tokens += is_correct.sum().item()
                    total_tokens += len(target_tokens)
                    
                    first_token_correct += is_correct[0].item()
                    if is_correct.all().item():
                        full_match_correct += 1
                total_seqs += 1
                
    if was_training:
        model.train()
        
    if seed is not None:
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)
            
    avg_token_acc = (correct_tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0
    first_token_acc = (first_token_correct / total_seqs * 100.0) if total_seqs > 0 else 0.0
    full_match_acc = (full_match_correct / total_seqs * 100.0) if total_seqs > 0 else 0.0
    
    return {
        'avg_token_acc': avg_token_acc,
        'first_token_acc': first_token_acc,
        'full_match_acc': full_match_acc
    }

def train_induction(model, config, model_name, checkpoint_dir=None):
    print(f"\n--- Training {model_name} on Induction Head ---")
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
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
    assert config.use_mixed_precision, "use_mixed_precision MUST be True for HOFA models."
    use_autocast = config.use_mixed_precision and device == "cuda"
    warmup_steps = int(0.05 * config.train_steps)

    model.train()
    history = {'loss': [], 'acc': [], 'mix_bias': []}
    start_step = 0
    
    if checkpoint_dir is not None:
        metadata = load_checkpoint(model, optimizer, model_name, -1, checkpoint_dir)
        if metadata is not None:
            print(f"\n      Resuming {model_name} from step {metadata['step']}")
            start_step = metadata['step']
            history = metadata.get('history', {'loss': [], 'acc': [], 'mix_bias': []})
            
            if start_step >= config.train_steps:
                print(f"      {model_name} already completed or was skipped previously. Skipping.")
                return history

    early_stopper = GracefulInterruptHandler()
    early_stopper.attach()
    
    for i in range(start_step, config.train_steps):
        adjust_learning_rate(optimizer, i, config.train_steps, config.learning_rate, warmup_steps)
        optimizer.zero_grad()
        
        x, y = generate_induction_seqs(config.batch_size, config.seq_len, config.vocab_size, device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            logits, loss = model(x, targets=y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        if (i + 1) % config.print_every == 0 or i == 0 or (i + 1) == config.train_steps:
            val_metrics = evaluate_induction_accuracy(
                model, config, seq_len=config.seq_len, eval_samples=config.batch_size,
                eval_batch_size=config.batch_size, device=device,
                pattern_len=None, pin_to_end=False, seed=config.seed + 10000 + i
            )
            acc = val_metrics['avg_token_acc']
            
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
            
            print(f"\r      Step {i + 1:5d}/{config.train_steps} | Train Loss: {loss.item():.4f} | Train Acc: {acc:.1f}%{bias_str}{norm_str}")
            history['loss'].append(loss.item())
            history['acc'].append(acc)
            
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

def run_induction_experiment():
    config = InductionExperimentConfig()
    device = config.device
    torch.manual_seed(config.seed)
    
    models_to_test = config.models_to_test
    
    seq_lengths = config.seq_lengths
    trendline_results = {name: [] for name, _, _ in models_to_test}
    
    for seq_len in seq_lengths:
        config.seq_len = seq_len
        print(f"\n========================================")
        print(f" Starting sequence length: {seq_len}")
        print(f"========================================")
        
        all_histories = {}
        for name, attn_type, r_val in models_to_test:
            # Resetting the seed here ensures every model sees the *exact same* 
            # sequence of training data, providing the fairest possible comparison.
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            
            if r_val is not None:
                config.model_config.r = r_val
            model = GenericBenchmarkLM(
                vocab_size=config.vocab_size,
                d_model=config.model_config.d_model,
                attn_type=attn_type,
                num_heads=config.model_config.num_heads,
                num_layers=config.model_config.num_layers,
                model_cfg=config.model_config
            ).to(device)

            checkpoint_dir = f"data/induction_models/seqlen_{seq_len}/induction_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            history = train_induction(model, config, name, checkpoint_dir=checkpoint_dir)
            all_histories[name] = history
            
            # Save final model state dict for easy loading later
            torch.save(model.state_dict(), f"{checkpoint_dir}/final_model.pt")
            
            del model
            torch.cuda.empty_cache()
            
            max_acc = max(history['acc']) if len(history['acc']) > 0 else 0
            trendline_results[name].append(max_acc)
            
        # Plotting Convergence for this sequence length
        os.makedirs("data/plots/induction", exist_ok=True)
        plt.figure(figsize=(10, 6))
        for name, history in all_histories.items():
            if len(history['acc']) == 0: continue
            steps = np.arange(1, len(history['acc']) + 1) * config.print_every
            steps[0] = 1
            plt.plot(steps, history['acc'], label=name, marker='o', markersize=3)
            
        plt.title(f"Induction Head Task Convergence (Seq Len = {seq_len})")
        plt.xlabel("Training Steps")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = f"data/plots/induction/convergence_seqlen_{seq_len}.pdf"
        plt.savefig(plot_path, bbox_inches='tight', format='pdf', dpi=300)
        plt.close()
        print(f"\nConvergence plot saved to {plot_path}")

    # Final Trendline Plot is now handled by benchmarks.plotting.plot_induction

if __name__ == "__main__":
    run_induction_experiment()
    
    # Automatically generate the unified trendline plot when fully finished
    from benchmarks.plotting.plot_induction import plot_unified_trendline
    plot_unified_trendline()
