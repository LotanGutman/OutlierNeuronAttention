import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import math
import torch
torch.set_float32_matmul_precision('high')
import numpy as np
from src.HybridOutlierFactorizedAttentionTrain import SubwordLM
from training.training_config import LanguageModelingExperimentConfig
from interrupt_util.interrupts import GracefulInterruptHandler
from src.modules.benchmark_utils import adjust_learning_rate
from training.modules.eval import compute_val_ppl_from_cache

def load_batch(cache_path, batch_size, seq_len, start_idx):
    # Returns (x, y), next_start_idx
    # cache_path contains np.uint16 tokens
    file_size_bytes = os.path.getsize(cache_path)
    total_tokens = file_size_bytes // 2
    
    tokens_needed = batch_size * (seq_len + 1)
    
    if start_idx + tokens_needed > total_tokens:
        # Loop around or just stop. For smoke test, wrap around.
        start_idx = 0
        
    # Read the required chunk
    with open(cache_path, 'rb') as f:
        f.seek(start_idx * 2)
        chunk_bytes = f.read(tokens_needed * 2)
        
    data = np.frombuffer(chunk_bytes, dtype=np.uint16).astype(np.int64)
    data = torch.from_numpy(data).view(batch_size, seq_len + 1)
    
    x = data[:, :-1].contiguous()
    y = data[:, 1:].contiguous()
    
    return x, y, start_idx + tokens_needed

def train(config: LanguageModelingExperimentConfig):
    model_size = config.model_name.split('_')[0]
    cache_path: str = f"data/datasets/data_{model_size}_cache.bin"
    val_cache_path: str = f"data/datasets/data_{model_size}_val_cache.bin"
    checkpoint_dir = f"data/training/{config.model_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if not os.path.exists(cache_path) or not os.path.exists(val_cache_path):
        raise FileNotFoundError(f"Dataset cache not found at {cache_path} or {val_cache_path}. Run download script first.")
        
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    device = torch.device(config.device)
    model = SubwordLM(config.vocab_size, config.model_config)
    model.to(device)
    
    if config.use_mixed_precision:
        pass    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    metrics = {
        'processed_tokens': [],
        'loss': [],
        'learning_rate': [],
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
    
    print(f"Starting {config.model_name} HOFA training for {config.train_steps} steps...")
    
    model.train()
    
    try:
        for step in range(start_step, config.train_steps):
            t0 = time.time()
            
            adjust_learning_rate(optimizer, step, config.train_steps, config.learning_rate, config.warmup_steps)
            optimizer.zero_grad(set_to_none=True)
            
            step_loss = 0.0
            
            for micro_step in range(config.gradient_accumulation_steps):
                x, y, start_idx = load_batch(cache_path, config.micro_batch_size, config.seq_len, start_idx)
                x, y = x.to(device), y.to(device)
                
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
                
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
                
            t1 = time.time()
            dt = t1 - t0
            total_processed_tokens += config.batch_size * config.seq_len
            
            # Log metrics
            metrics['processed_tokens'].append(total_processed_tokens)
            metrics['loss'].append(step_loss)
            metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])
            metrics['step_times'].append(dt)

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
        print(f"\nCRASH DETECTED: {e}")
        print("Saving emergency checkpoint before failing...")
    finally:
        # Final save on normal exit, crash, or interrupt
        interrupt_handler.detach()
        save_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        temp_path = save_path + ".tmp"
        
        # same thing, cleanup model
        raw_state_dict = model.state_dict()
        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
        
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
        print(f"Final checkpoint saved to {save_path}.")

if __name__ == "__main__":
    config = LanguageModelingExperimentConfig()
    train(config)
