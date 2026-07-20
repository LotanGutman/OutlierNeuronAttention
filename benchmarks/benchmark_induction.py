import torch
import torch.nn.functional as F
import os
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

def generate_induction_seqs(batch_size, seq_len, vocab_size, device):
    """
    Generates a sequence for the Induction Head task.
    A random pattern is placed at the start of the sequence, and then repeated 
    at a random position in the second half. The model must predict the 
    subsequent tokens of the pattern upon seeing its prefix in the second half.
    """
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        max_len = min(128, max(2, seq_len // 4))
        pattern_len = torch.randint(min(32, max_len), max_len + 1, (1,)).item()
        pattern = torch.randint(0, vocab_size, (pattern_len,), device=device)
        
        # Place pattern at the start
        x[b, :pattern_len] = pattern
        
        # Place pattern at a random location in the second half
        start_idx = torch.randint(seq_len // 2, seq_len - pattern_len, (1,)).item()
        x[b, start_idx:start_idx+pattern_len] = pattern
        
        # The targets are the next tokens of the pattern.
        # e.g., if pattern is [A, B, C], and it starts at index i,
        # predicting at i gives B, predicting at i+1 gives C.
        y[b, start_idx:start_idx+pattern_len-1] = pattern[1:]
        
    return x, y

def train_induction(model, config, model_name, checkpoint_dir=None):
    print(f"\n--- Training {model_name} on Induction Head ---")
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
    assert config.use_mixed_precision, "use_mixed_precision MUST be True for HOFA models."
    use_autocast = config.use_mixed_precision and device == "cuda"
    warmup_steps = int(0.05 * config.train_steps)

    model.train()
    history = {'loss': [], 'acc': [], 'mix_bias': []}
    consecutive_perfect_acc = 0
    start_step = 0
    
    if checkpoint_dir is not None:
        metadata = load_checkpoint(model, optimizer, model_name, -1, checkpoint_dir)
        if metadata is not None:
            print(f"\n      Resuming {model_name} from step {metadata['step']}")
            start_step = metadata['step']
            history = metadata.get('history', {'loss': [], 'acc': [], 'mix_bias': []})
            consecutive_perfect_acc = metadata.get('consecutive_perfect_acc', 0)
            
            if start_step >= config.train_steps or consecutive_perfect_acc >= 2 or metadata.get('stopped_early', False):
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
            model.eval()
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                    logits_valid, _ = model(x, targets=y, return_loss=False)
                preds = logits_valid.argmax(dim=-1)
                valid_mask = y[y != -100] # y is targets, valid_mask is the valid targets themselves
                # Wait, the valid targets are returned, so we compare preds to valid targets
                acc = (preds == valid_mask).float().mean().item() * 100
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

            if acc >= 99.5:
                consecutive_perfect_acc += 1
            else:
                consecutive_perfect_acc = 0
                
            if consecutive_perfect_acc >= 2:
                print("\n      Early stopping: achieved >99.5% accuracy for 2 consecutive evaluations.")
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
        # Save a final checkpoint indicating it's done or was intentionally skipped
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

def run_induction_experiment():
    config = InductionExperimentConfig()
    device = config.device
    torch.manual_seed(config.seed)
    
    models_to_test = config.models_to_test
    
    seq_lengths = [1024, 512, 256, 128, 64]
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
