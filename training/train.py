import os
from src.utils.platform_utils import configure_pytorch_allocator
configure_pytorch_allocator()
import time
import math
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
import numpy as np
from src.HybridOutlierFactorizedAttentionTrain import SubwordLM
from training.training_config import LanguageModelingExperimentConfig
from src.utils.interrupts import GracefulInterruptHandler
from src.modules.benchmark_utils import adjust_learning_rate
from training.modules.eval import compute_val_ppl_from_cache

from training.data_utils import FastTokenLoader

def train(config: LanguageModelingExperimentConfig):
    model_size = config.model_name.split('_')[0]
    cache_path: str = config.train_cache_path or f"data/datasets/data_{model_size}_cache.bin"
    val_cache_path: str = config.val_cache_path or f"data/datasets/data_{model_size}_val_cache.bin"
    checkpoint_dir = f"data/training/{config.model_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if getattr(config, 'activation_checkpointing_interval', None) is not None:
        print(f"[Notice] activation_checkpointing_interval={config.activation_checkpointing_interval} is ignored during regular pretraining (designed for long-context continual pretraining).")
    
    if not os.path.exists(cache_path) or not os.path.exists(val_cache_path):
        raise FileNotFoundError(f"Dataset cache not found at {cache_path} or {val_cache_path}. Run download script first.")
        
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    device = torch.device(config.device)
    model = SubwordLM(config.vocab_size, config.model_config)
    model.to(device)
    
    assert config.use_mixed_precision, "use_mixed_precision MUST be True for HOFA models."
    
    # Build optimizer parameter groups with zero weight decay for biases and log_alpha
    gate_multiplier = getattr(config, 'gate_lr_multiplier', None)
    if gate_multiplier is None:
        gate_multiplier = 1.0

    decay_params = []
    nodecay_params = []
    gate_params = []

    gate_param_ids = set()

    for name, module in model.named_modules():
        if getattr(module, '_is_gate', False) or name.endswith('gate_proj') or name.endswith('mix_proj'):
            for p_name, p in module.named_parameters(recurse=False):
                gate_param_ids.add(id(p))
                gate_params.append(p)
        if hasattr(module, 'fixed_mix') and isinstance(module.fixed_mix, torch.nn.Parameter):
            p = module.fixed_mix
            if id(p) not in gate_param_ids:
                gate_param_ids.add(id(p))
                gate_params.append(p)
        if hasattr(module, 'gla_scale') and isinstance(module.gla_scale, torch.nn.Parameter):
            p = module.gla_scale
            if id(p) not in gate_param_ids:
                gate_param_ids.add(id(p))
                gate_params.append(p)
        if hasattr(module, 'log_alpha') and isinstance(module.log_alpha, torch.nn.Parameter):
            p = module.log_alpha
            if id(p) not in gate_param_ids:
                gate_param_ids.add(id(p))
                gate_params.append(p)

    for name, p in model.named_parameters():
        if id(p) in gate_param_ids:
            continue
        if p.dim() < 2 or name.endswith('.bias') or 'norm' in name.lower() or 'scale' in name.lower():
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {'params': decay_params, 'lr': config.learning_rate, 'weight_decay': config.weight_decay, 'lr_multiplier': 1.0},
        {'params': nodecay_params, 'lr': config.learning_rate, 'weight_decay': 0.0, 'lr_multiplier': 1.0},
        {'params': gate_params, 'lr': config.learning_rate * gate_multiplier, 'weight_decay': 0.0, 'lr_multiplier': gate_multiplier},
    ]

    print(f"[Optimizer] Parameter groups initialized: {len(decay_params)} decay tensors (wd={config.weight_decay}), "
          f"{len(nodecay_params)} no-decay tensors, {len(gate_params)} gate/alpha tensors (mult={gate_multiplier:.1f}x, wd=0.0).")

    optimizer = torch.optim.AdamW(
        param_groups, 
        betas=(0.9, 0.95),
        fused=True
    )
    
    metrics = {
        'processed_tokens': [],
        'loss': [],
        'learning_rate': [],
        'gate_alpha': [],
        'gate_decay': [],
        'alpha_min': [],
        'alpha_max': [],
        'alpha_mean': [],
        'alpha_std': [],
        'alpha_grad_norm': [],
        'alpha_update_norm': [],
        'alpha_per_layer_head': {},
        'step_times': [],
        # validation keys
        'val_steps': [],
        'val_tokens': [],
        'val_ppl': []
    }
    
    start_step = 0
    start_idx = 0
    total_processed_tokens = 0
    best_val_ppl = float('inf')
    
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        if 'metrics' in ckpt:
            old_metrics = ckpt['metrics']
            for key, value in old_metrics.items():
                if key in metrics:  # Only update keys that already exist in our template
                    metrics[key] = value
        
        if 'step' in ckpt:
            start_step = ckpt['step']
            
        if 'rng_state' in ckpt:
            torch.set_rng_state(ckpt['rng_state'].cpu())
        if 'cuda_rng_state' in ckpt:
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt['cuda_rng_state']])
        
        if 'total_processed_tokens' in ckpt:
            total_processed_tokens = ckpt['total_processed_tokens']
        else:
            total_processed_tokens = start_step * config.batch_size * config.seq_len
            
        if 'start_idx' in ckpt:
            start_idx = ckpt['start_idx']
        else:
            start_idx = total_processed_tokens
            
        if 'best_val_ppl' in ckpt:
            best_val_ppl = ckpt['best_val_ppl']
            
        print(f"Resumed successfully from step {start_step}.")
    
    model = torch.compile(model, dynamic=True)
    # Initialize interrupt handler
    interrupt_handler = GracefulInterruptHandler()
    interrupt_handler.attach()
    
    if start_step > 0:
        print(f"Resuming {config.model_name} HOFA training from step {start_step} to {config.train_steps} steps...")
    else:
        print(f"Starting {config.model_name} HOFA training for {config.train_steps} steps...")
    
    model.train()
    
    # Initialize FastTokenLoader for training
    global_batch_size = config.micro_batch_size * config.gradient_accumulation_steps
    train_loader = FastTokenLoader(cache_path, global_batch_size, config.seq_len, start_idx)
    
    is_crash = False
    try:
        for step in range(start_step, config.train_steps):
            t0 = time.time()
            
            adjust_learning_rate(optimizer, step, config.train_steps, config.learning_rate, config.warmup_steps)
            optimizer.zero_grad(set_to_none=True)
            
            step_loss = 0.0
            
            # Fetch global batch and transfer to GPU asynchronously
            x_global, y_global, next_idx = train_loader.get_batch()
            x_global = x_global.to(device, non_blocking=True)
            y_global = y_global.to(device, non_blocking=True)
            
            # Update start_idx tracking for checkpointing
            start_idx = next_idx
            
            for micro_step in range(config.gradient_accumulation_steps):
                # Slice the global batch into micro batches entirely on the GPU
                start_m = micro_step * config.micro_batch_size
                end_m = start_m + config.micro_batch_size
                
                x = x_global[start_m:end_m]
                y = y_global[start_m:end_m]
                
                if config.use_mixed_precision:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        _, loss = model(x, targets=y)
                        loss = loss / config.gradient_accumulation_steps
                    loss.backward()
                else:
                    _, loss = model(x, targets=y)
                    loss = loss / config.gradient_accumulation_steps
                    loss.backward()
                    
                step_loss += loss.item()
                
            # Telemetry for alpha gradients before clipping
            log_alpha_params = [p for name, p in model.named_parameters() if name.endswith('.log_alpha')]
            if log_alpha_params:
                total_alpha_grad_sq = sum(p.grad.pow(2).sum().item() for p in log_alpha_params if p.grad is not None)
                alpha_grad_norm = math.sqrt(total_alpha_grad_sq)
                old_alpha_values = [p.detach().clone() for p in log_alpha_params]
            else:
                alpha_grad_norm = None
                old_alpha_values = None

            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            if log_alpha_params and old_alpha_values:
                total_update_sq = sum((p.detach() - old_p).pow(2).sum().item() for p, old_p in zip(log_alpha_params, old_alpha_values))
                alpha_update_norm = math.sqrt(total_update_sq)
            else:
                alpha_update_norm = None
                
            t1 = time.time()
            dt = t1 - t0
            total_processed_tokens += config.batch_size * config.seq_len
            
            # Log metrics
            metrics['processed_tokens'].append(total_processed_tokens)
            metrics['loss'].append(step_loss)
            metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])
            metrics['step_times'].append(dt)

            # Gate and Alpha telemetry
            mean_alpha = None
            mean_decay = None
            a_min, a_max, a_mean, a_std = None, None, None, None
            is_pure_mha = "MHA" in config.model_name or (hasattr(config.model_config, 'r') and config.model_config.r == config.model_config.d_head)
            if not is_pure_mha:
                mix_biases = [layer.attn.mix_proj.bias for layer in model.layers if hasattr(layer, 'attn') and hasattr(layer.attn, 'mix_proj') and layer.attn.mix_proj.bias is not None]
                decay_biases = [layer.attn.gate_proj.bias for layer in model.layers if hasattr(layer, 'attn') and hasattr(layer.attn, 'gate_proj') and layer.attn.gate_proj.bias is not None]
                if mix_biases and decay_biases:
                    mean_alpha = torch.sigmoid(torch.cat(mix_biases)).mean().item()
                    mean_decay = torch.sigmoid(-torch.cat(decay_biases)).mean().item()

                if log_alpha_params:
                    all_alphas = torch.cat([p.detach().exp().view(-1) for p in log_alpha_params])
                    a_min = all_alphas.min().item()
                    a_max = all_alphas.max().item()
                    a_mean = all_alphas.mean().item()
                    a_std = all_alphas.std().item()

            metrics['gate_alpha'].append(mean_alpha)
            metrics['gate_decay'].append(mean_decay)
            metrics['alpha_min'].append(a_min)
            metrics['alpha_max'].append(a_max)
            metrics['alpha_mean'].append(a_mean)
            metrics['alpha_std'].append(a_std)
            metrics['alpha_grad_norm'].append(alpha_grad_norm)
            metrics['alpha_update_norm'].append(alpha_update_norm)

            if log_alpha_params and ((step > 0 and step % config.val_every == 0) or (step + 1 == config.train_steps)):
                eval_step_key = str(step)
                layer_head_dict = {}
                for l_idx, layer in enumerate(model.layers):
                    if hasattr(layer, 'attn') and hasattr(layer.attn, 'log_alpha') and layer.attn.log_alpha is not None:
                        a_lh = layer.attn.log_alpha.detach().exp().squeeze(0).squeeze(1)
                        for h_idx in range(a_lh.shape[0]):
                            vals = a_lh[h_idx]
                            layer_head_dict[f"L{l_idx}_H{h_idx}"] = {
                                "min": vals.min().item(),
                                "mean": vals.mean().item(),
                                "max": vals.max().item(),
                                "std": vals.std().item(),
                            }
                metrics['alpha_per_layer_head'][eval_step_key] = layer_head_dict

            # Validation perplexity
            if step % config.val_every == 0:
                val_ppl = compute_val_ppl_from_cache(
                    val_cache_path, model,
                    batch_size=config.micro_batch_size,   # same micro batch for consistency
                    seq_len=config.seq_len,
                    num_batches=config.val_num_batches,
                    device=device
                )
                if val_ppl is not None:
                    metrics['val_steps'].append(step)
                    metrics['val_tokens'].append(total_processed_tokens)
                    metrics['val_ppl'].append(val_ppl)
                    print(f"\n[Val @ step {step}] Perplexity: {val_ppl:.3f}")
                    
                    if val_ppl < best_val_ppl:
                        best_val_ppl = val_ppl
                        
                        best_save_path = os.path.join(checkpoint_dir, "checkpoint_best_val.pt")
                        best_temp_path = best_save_path + ".tmp"
                        
                        raw_state_dict = model.state_dict()
                        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
                        

                        torch.save({
                            'model_state_dict': clean_state_dict,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'config': config,
                            'metrics': metrics,
                            'step': step,
                            'start_idx': start_idx,
                            'total_processed_tokens': total_processed_tokens,
                            'best_val_ppl': best_val_ppl,
                            'rng_state': torch.get_rng_state(),
                            'cuda_rng_state': torch.cuda.get_rng_state_all()
                        }, best_temp_path)
                        os.replace(best_temp_path, best_save_path)
                        print(f"*** New best validation perplexity! Saved to {best_save_path} ***")
            
            
            if step > 0 and step % config.print_every == 0:
                print(f"\r\033[KStep {step:4d} | Loss: {step_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {dt:.3f}s | Tokens: {total_processed_tokens}")
            else:
                print(f"\rStep {step}/{config.train_steps} - loss: {step_loss:.4f}", end="", flush=True)
                
            if step > 0 and step % config.save_every == 0:
                save_path = os.path.join(checkpoint_dir, "checkpoint.pt")
                temp_path = save_path + ".tmp"
                
                # cleanup model state dict in case we're using the compiled model (save as "uncompiled")
                raw_state_dict = model.state_dict()
                clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}

                torch.save({
                    'model_state_dict': clean_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'metrics': metrics,
                    'step': step,
                    'start_idx': start_idx,
                    'total_processed_tokens': total_processed_tokens,
                    'best_val_ppl': best_val_ppl,
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all()
                }, temp_path)
                os.replace(temp_path, save_path)
                print(f"Saved intermediate checkpoint at step {step}.")
                
            if interrupt_handler.stop_requested:
                print(f"Graceful stop requested. Saving checkpoint and exiting...")
                break
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        error_str = str(e).lower()
        if "device not ready" in error_str:
            print("\n" + "="*80)
            print("If you are running this on WSL, than it may not support GPU level linux control which may cause you to OOM (and hence this crash). Please check your VRAM :)")
            print("="*80 + "\n")
        else:
            print(f"\nCRASH DETECTED: {e}")
            
        print("Saving emergency checkpoint before failing...")
        is_crash = True
    finally:
        # Final save on normal exit, crash, or interrupt
        interrupt_handler.detach()
        
        # If we crashed, save to a distinct emergency file to avoid corrupting the main checkpoint tmp file
        final_filename = "checkpoint_emergency.pt" if is_crash else "checkpoint.pt"
        save_path = os.path.join(checkpoint_dir, final_filename)
        temp_path = save_path + ".tmp"
        
        # same thing, cleanup model
        raw_state_dict = model.state_dict()
        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
        
        try:
            torch.save({
                'model_state_dict': clean_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'step': step,
                'start_idx': start_idx,
                'total_processed_tokens': total_processed_tokens,
                'best_val_ppl': best_val_ppl,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all()
            }, temp_path)
            os.replace(temp_path, save_path)
            print(f"Final checkpoint successfully saved to {save_path}.")
        except Exception as save_err:
            print(f"CRITICAL: Failed to save final checkpoint to {save_path}! Error: {save_err}")
            if is_crash:
                print("This usually means the disk is completely full.")

if __name__ == "__main__":
    config = LanguageModelingExperimentConfig()
    train(config)
