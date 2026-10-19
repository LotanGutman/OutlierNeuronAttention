"""Continual pretraining (CPT) on PG19 with YaRN RoPE scaling."""

import os
from src.utils.platform_utils import configure_pytorch_allocator
configure_pytorch_allocator()
import gc
import time
import math
import json
import hashlib
from typing import Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from src.HybridOutlierFactorizedAttentionTrain import SubwordLM
from src.yarn import patch_model_yarn, swap_rotary_to_native, restore_rotary_from_saved
from training.training_config import LanguageModelingExperimentConfig, YaRNConfig, get_model_config
from interrupt_util.interrupts import GracefulInterruptHandler
from training.data_utils import FastTokenLoader


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def get_cpt_lr(step: int, total_steps: int, base_lr: float, min_lr: float = 1e-6, warmup_steps: int = 30) -> float:
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def evaluate_cache(
    model: SubwordLM,
    cache_path: str,
    batch_size: int,
    seq_len: int,
    num_batches: int,
    device: torch.device,
    chunk_size: int = 4096,
    index_path: Optional[str] = None,
) -> Optional[float]:
    if not os.path.exists(cache_path):
        return None

    file_size_bytes = os.path.getsize(cache_path)
    total_tokens = file_size_bytes // 2
    window_tokens = seq_len + 1

    target_windows = []
    if index_path is None:
        candidate_idx = cache_path.replace(".bin", "_index.json")
        if os.path.exists(candidate_idx):
            index_path = candidate_idx

    if index_path and os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
                books = idx_data.get("books", [])
                total_windows = total_tokens // window_tokens
                for b in books:
                    fw = b.get("first_window", 0)
                    if fw < total_windows:
                        target_windows.append(fw)
        except Exception:
            target_windows = []

    total_loss = 0.0
    total_tokens_eval = 0

    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        if target_windows:
            mmap_data = np.memmap(cache_path, dtype=np.uint16, mode='r')
            for w_idx in target_windows:
                start_token = w_idx * window_tokens
                raw = mmap_data[start_token : start_token + window_tokens]
                if len(raw) < window_tokens:
                    continue
                tensor = torch.from_numpy(raw.astype(np.int64)).unsqueeze(0).to(device, non_blocking=True)
                x = tensor[:, :-1].contiguous()
                y = tensor[:, 1:].contiguous()

                h = model.token_emb(x)
                for layer in model.layers:
                    h = layer(h)
                h = model.ln_f(h)

                h_flat = h.view(-1, h.shape[-1])
                y_flat = y.view(-1)
                for i in range(0, h_flat.shape[0], chunk_size):
                    end = min(i + chunk_size, h_flat.shape[0])
                    sub_logits = model.lm_head(h_flat[i:end])
                    sub_loss = F.cross_entropy(sub_logits, y_flat[i:end], reduction='sum')
                    total_loss += sub_loss.item()
                    total_tokens_eval += (end - i)
        else:
            tokens_needed = num_batches * batch_size * window_tokens
            if total_tokens < tokens_needed:
                return None
            loader = FastTokenLoader(cache_path, batch_size, seq_len, start_idx=0)
            for _ in range(num_batches):
                x, y, _ = loader.get_batch()
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                h = model.token_emb(x)
                for layer in model.layers:
                    h = layer(h)
                h = model.ln_f(h)

                h_flat = h.view(-1, h.shape[-1])
                y_flat = y.view(-1)
                for i in range(0, h_flat.shape[0], chunk_size):
                    end = min(i + chunk_size, h_flat.shape[0])
                    sub_logits = model.lm_head(h_flat[i:end])
                    sub_loss = F.cross_entropy(sub_logits, y_flat[i:end], reduction='sum')
                    total_loss += sub_loss.item()
                    total_tokens_eval += (end - i)

    model.train()
    return math.exp(total_loss / total_tokens_eval) if total_tokens_eval > 0 else None


def forward_and_backward_chunked(
    model: SubwordLM,
    x: torch.Tensor,
    targets: torch.Tensor,
    grad_accum_steps: int = 1,
    chunk_size: int = 4096,
    checkpointing_interval: Optional[int] = None,
) -> float:
    # Chunked CE loss calculation (identical to direct CE) to prevent VRAM explosion at long contexts
    B, N = x.shape
    total_tokens = B * N

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        h = model.token_emb(x)
        for i, layer in enumerate(model.layers):
            if checkpointing_interval is not None and (i % checkpointing_interval == 0):
                h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        h = model.ln_f(h)

        h_flat = h.view(total_tokens, -1)
        y_flat = targets.view(total_tokens)
        grad_h = torch.empty_like(h_flat)
        total_loss = 0.0

        for i in range(0, total_tokens, chunk_size):
            end = min(i + chunk_size, total_tokens)
            sub_h = h_flat[i:end].detach().clone().requires_grad_(True)
            sub_y = y_flat[i:end]

            sub_logits = model.lm_head(sub_h)
            sub_loss = (F.cross_entropy(sub_logits, sub_y, reduction='sum') / total_tokens) / grad_accum_steps
            total_loss += sub_loss.item() * grad_accum_steps
            sub_loss.backward()
            grad_h[i:end] = sub_h.grad

    h_flat.backward(grad_h)

    return total_loss


def train_cpt(config: Union[YaRNConfig, LanguageModelingExperimentConfig]):
    if isinstance(config, YaRNConfig):
        yarn_cfg = config
        lm_cfg = config.model
    else:
        yarn_cfg = None
        lm_cfg = config

    model_size = lm_cfg.model_name.split('_')[0]
    base_model_name = lm_cfg.model_name.replace("_cpt", "")
    checkpoint_dir = f"data/training/{lm_cfg.model_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    cache_path = lm_cfg.train_cache_path or "data/datasets/pg19_16k_train.bin"
    val_cache_path = lm_cfg.val_cache_path or "data/datasets/pg19_16k_val.bin"
    fineweb_val_cache_path = f"data/datasets/data_{model_size}_val_cache.bin"
    if os.path.exists(fineweb_val_cache_path):
        print(f"FineWeb validation cache found: {fineweb_val_cache_path}. Retention will be evaluated every {lm_cfg.val_every} steps.")
    else:
        print(f"FineWeb validation cache '{fineweb_val_cache_path}' " + rf"not found. FineWeb retention evaluation will be skipped (N/A) (you can download it via python main.py train --scale (13M\70M\...\350M) --download-data --val-only).")

    if not os.path.exists(cache_path) or not os.path.exists(val_cache_path):
        raise FileNotFoundError(f"PG19 dataset cache not found at {cache_path} or {val_cache_path}. Run download script first.")

    torch.manual_seed(lm_cfg.seed)
    torch.cuda.manual_seed_all(lm_cfg.seed)
    device = torch.device(lm_cfg.device)

    model = SubwordLM(lm_cfg.vocab_size, lm_cfg.model_config)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lm_cfg.learning_rate,
        weight_decay=lm_cfg.weight_decay,
        betas=(0.9, 0.95),
        fused=True
    )

    metrics = {
        'processed_tokens': [],
        'loss': [],
        'learning_rate': [],
        'step_times': [],
        'val_steps': [],
        'val_pg19_ppl': [],
        'val_fineweb_ppl_yarn': [],
        'val_fineweb_ppl_native': [],
    }

    start_step = 0
    start_idx = 0
    total_processed_tokens = 0
    best_val_ppl = float('inf')
    starting_base_ckpt_info = {}
    base_seq_len = 1024

    cpt_ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    if os.path.exists(cpt_ckpt_path):
        print(f"Resuming CPT from existing checkpoint: {cpt_ckpt_path}")
        ckpt = torch.load(cpt_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'starting_base_ckpt_info' in ckpt:
            starting_base_ckpt_info = ckpt['starting_base_ckpt_info']
            base_seq_len = starting_base_ckpt_info.get("base_seq_len", base_seq_len)
        if 'metrics' in ckpt:
            metrics.update(ckpt['metrics'])
        if 'step' in ckpt:
            start_step = ckpt['step']
        if 'start_idx' in ckpt:
            start_idx = ckpt['start_idx']
        if 'total_processed_tokens' in ckpt:
            total_processed_tokens = ckpt['total_processed_tokens']
        if 'best_val_ppl' in ckpt:
            best_val_ppl = ckpt['best_val_ppl']
        if 'rng_state' in ckpt:
            torch.set_rng_state(ckpt['rng_state'].cpu())
        if 'cuda_rng_state' in ckpt:
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt['cuda_rng_state']])
        print(f"Resumed successfully from step {start_step}.")
    else:
        base_dir = f"data/training/{base_model_name}"
        candidate_paths = []
        if yarn_cfg and yarn_cfg.base_checkpoint_path:
            candidate_paths.append(yarn_cfg.base_checkpoint_path)
        candidate_paths.extend([
            os.path.join(base_dir, "checkpoint_best_val.pt"),
            os.path.join(base_dir, "checkpoint.pt"),
        ])

        chosen_base_ckpt = None
        for p in candidate_paths:
            if os.path.exists(p):
                chosen_base_ckpt = p
                break

        if chosen_base_ckpt is None:
            raise FileNotFoundError(
                f"No pretrained checkpoint found for base model '{base_model_name}'. "
                f"Checked: {candidate_paths}. CPT requires a valid base checkpoint."
            )

        print(f"Initializing CPT from pretrained base checkpoint: {chosen_base_ckpt}")
        ckpt = torch.load(chosen_base_ckpt, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)

        if isinstance(ckpt, dict) and 'config' in ckpt:
            saved_cfg = ckpt['config']
            base_seq_len = getattr(saved_cfg, 'seq_len', getattr(getattr(saved_cfg, 'model_config', None), 'block_size', 1024))
        elif yarn_cfg and yarn_cfg.original_max_seq_len is not None:
            base_seq_len = yarn_cfg.original_max_seq_len
        else:
            try:
                base_seq_len = get_model_config(base_model_name).seq_len
            except Exception:
                base_seq_len = 1024

        starting_base_ckpt_info = {
            "path": chosen_base_ckpt,
            "step": ckpt.get("step", 0) if isinstance(ckpt, dict) else 0,
            "sha256": sha256_file(chosen_base_ckpt),
            "base_seq_len": base_seq_len,
        }
        print(f"Base checkpoint SHA256: {starting_base_ckpt_info['sha256'][:16]}... (base step {starting_base_ckpt_info['step']}, base_seq_len {base_seq_len})")
        del ckpt, state_dict
        gc.collect()

    beta_fast = getattr(yarn_cfg, 'beta_fast', 32.0) if yarn_cfg else 32.0
    beta_slow = getattr(yarn_cfg, 'beta_slow', 1.0) if yarn_cfg else 1.0

    n_patched = patch_model_yarn(
        model,
        target_seq_len=lm_cfg.seq_len,
        original_seq_len=base_seq_len,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        dtype=torch.bfloat16,
    )
    s = lm_cfg.seq_len / base_seq_len
    print(f"Patched {n_patched} layers with canonical YaRN RoPE (s={s:.1f}x, {base_seq_len} -> {lm_cfg.seq_len}).")

    interrupt_handler = GracefulInterruptHandler()
    interrupt_handler.attach()

    global_batch_size = lm_cfg.micro_batch_size * lm_cfg.gradient_accumulation_steps
    train_loader = FastTokenLoader(cache_path, lm_cfg.micro_batch_size, lm_cfg.seq_len, start_idx)

    fw_batch_size = max(1, min(4, lm_cfg.micro_batch_size * 4))
    tokens_per_fw_batch = fw_batch_size * (base_seq_len + 1)
    fineweb_batches = max(1, math.ceil(500_000 / tokens_per_fw_batch))

    def run_evaluations(current_step: int):
        pg19_ppl = evaluate_cache(
            model=model,
            cache_path=val_cache_path,
            batch_size=lm_cfg.micro_batch_size,
            seq_len=lm_cfg.seq_len,
            num_batches=lm_cfg.val_num_batches,
            device=device,
        )

        fw_ppl_yarn = None
        fw_ppl_native = None
        if os.path.exists(fineweb_val_cache_path):
            fw_ppl_yarn = evaluate_cache(
                model=model,
                cache_path=fineweb_val_cache_path,
                batch_size=fw_batch_size,
                seq_len=base_seq_len,
                num_batches=fineweb_batches,
                device=device,
            )

            saved_yarn = swap_rotary_to_native(model, native_seq_len=base_seq_len, dtype=torch.bfloat16)
            try:
                fw_ppl_native = evaluate_cache(
                    model=model,
                    cache_path=fineweb_val_cache_path,
                    batch_size=fw_batch_size,
                    seq_len=base_seq_len,
                    num_batches=fineweb_batches,
                    device=device,
                )
            finally:
                restore_rotary_from_saved(model, saved_yarn)

        metrics['val_steps'].append(current_step)
        metrics['val_pg19_ppl'].append(pg19_ppl)
        metrics['val_fineweb_ppl_yarn'].append(fw_ppl_yarn)
        metrics['val_fineweb_ppl_native'].append(fw_ppl_native)

        pg19_str = f"{pg19_ppl:.2f}" if pg19_ppl is not None else "N/A"
        fw_yarn_str = f"{fw_ppl_yarn:.2f}" if fw_ppl_yarn is not None else "N/A"
        fw_native_str = f"{fw_ppl_native:.2f}" if fw_ppl_native is not None else "N/A"
        print(f"\r\033[K[Val @ step {current_step}] PG19 ({lm_cfg.seq_len // 1024}k) PPL: {pg19_str} | FineWeb Ret (YaRN): {fw_yarn_str} | FineWeb Ret (Native): {fw_native_str}")

        torch.cuda.empty_cache()
        return pg19_ppl

    min_lr = 0.1 * lm_cfg.learning_rate
    ckpt_interval = getattr(lm_cfg, 'activation_checkpointing_interval', None)
    if ckpt_interval is not None:
        print(f"[CPT] Layer activation checkpointing enabled every {ckpt_interval} layer(s).")
    print(f"\nStarting {lm_cfg.model_name} CPT from step {start_step} to {lm_cfg.train_steps}...")

    avg_step_time = None
    is_crash = False
    last_completed_step = start_step

    try:
        for step in range(start_step, lm_cfg.train_steps):
            t0 = time.time()
            lr = get_cpt_lr(step, lm_cfg.train_steps, lm_cfg.learning_rate, min_lr=min_lr, warmup_steps=lm_cfg.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _ in range(lm_cfg.gradient_accumulation_steps):
                x, y, start_idx = train_loader.get_batch()
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                step_loss = forward_and_backward_chunked(
                    model, x, y,
                    grad_accum_steps=lm_cfg.gradient_accumulation_steps,
                    chunk_size=4096,
                    checkpointing_interval=ckpt_interval,
                )
                accum_loss += step_loss / lm_cfg.gradient_accumulation_steps

            if lm_cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), lm_cfg.grad_clip_norm)

            optimizer.step()
            last_completed_step = step + 1

            step_time = time.time() - t0
            step_tokens = global_batch_size * lm_cfg.seq_len
            total_processed_tokens += step_tokens

            if avg_step_time is None:
                avg_step_time = step_time
            else:
                avg_step_time = 0.75 * avg_step_time + 0.25 * step_time

            remaining_steps = max(0, lm_cfg.train_steps - (step + 1))
            eta_seconds = remaining_steps * avg_step_time
            eta_hours = int(eta_seconds // 3600)
            eta_minutes = int((eta_seconds % 3600) // 60)

            metrics['processed_tokens'].append(total_processed_tokens)
            metrics['loss'].append(accum_loss)
            metrics['learning_rate'].append(lr)
            metrics['step_times'].append(step_time)

            is_last_step = (step + 1) == lm_cfg.train_steps
            if step % lm_cfg.val_every == 0 or is_last_step:
                eval_step = (step + 1) if is_last_step else step
                val_ppl = run_evaluations(eval_step)
                if val_ppl is not None and val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    best_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_best_val.pt")
                    best_temp_path = best_ckpt_path + ".tmp"
                    torch.save({
                        'step': eval_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': metrics,
                        'config': config,
                        'total_processed_tokens': total_processed_tokens,
                        'best_val_ppl': best_val_ppl,
                        'start_idx': start_idx,
                        'starting_base_ckpt_info': starting_base_ckpt_info,
                    }, best_temp_path)
                    os.replace(best_temp_path, best_ckpt_path)
                    print(f"*** New best validation perplexity ({best_val_ppl:.2f})! Saved to {best_ckpt_path} ***")

            print(f"\rStep {step}/{lm_cfg.train_steps} - loss: {accum_loss:.4f} ({avg_step_time:.2f}s/step, ETA: H: {eta_hours}, M: {eta_minutes})", end="", flush=True)

            if step > 0 and step % lm_cfg.save_every == 0:
                temp_path = cpt_ckpt_path + ".tmp"
                torch.save({
                    'step': step + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                    'config': config,
                    'total_processed_tokens': total_processed_tokens,
                    'best_val_ppl': best_val_ppl,
                    'start_idx': start_idx,
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all(),
                    'starting_base_ckpt_info': starting_base_ckpt_info,
                }, temp_path)
                os.replace(temp_path, cpt_ckpt_path)

            if interrupt_handler.stop_requested:
                print(f"\nGraceful stop requested. Saving checkpoint and exiting...")
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
            print(f"\n[CRASH] Training loop encountered an error: {e}")

        print("Saving emergency checkpoint before failing...")
        is_crash = True
        raise e
    finally:
        interrupt_handler.detach()

        final_filename = "checkpoint_emergency.pt" if is_crash else "checkpoint.pt"
        save_path = os.path.join(checkpoint_dir, final_filename)
        temp_path = save_path + ".tmp"

        try:
            torch.save({
                'step': last_completed_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': config,
                'total_processed_tokens': total_processed_tokens,
                'best_val_ppl': best_val_ppl,
                'start_idx': start_idx,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'starting_base_ckpt_info': starting_base_ckpt_info,
            }, temp_path)
            os.replace(temp_path, save_path)
            print(f"\nFinal checkpoint successfully saved to {save_path}.")
        except Exception as save_err:
            print(f"\nCRITICAL: Failed to save final checkpoint to {save_path}! Error: {save_err}")
            if is_crash:
                print("This usually means the disk is completely full.")

    if not is_crash and last_completed_step >= lm_cfg.train_steps:
        print(f"\n{lm_cfg.model_name} CPT finished successfully! Best PG19 Val PPL: {best_val_ppl:.2f}")
