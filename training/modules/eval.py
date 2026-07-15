import torch
import numpy as np
import os
import math
from datasets import load_dataset
import random

def compute_val_ppl_from_cache(val_cache_path, model, batch_size, seq_len, num_batches, device):
    if not os.path.exists(val_cache_path):
        print(f"Warning: Validation cache {val_cache_path} not found. Skipping val PPL.")
        return None

    file_size_bytes = os.path.getsize(val_cache_path)
    total_tokens = file_size_bytes // 2
    tokens_needed = num_batches * batch_size * (seq_len + 1)
    if total_tokens < tokens_needed:
        print(f"Validation cache has only {total_tokens} tokens; need {tokens_needed}. Skipping.")
        return None

    from training.data_utils import FastTokenLoader
    
    # Validation uses a random offset to test the whole dataset over time.
    # While random seeks technically break sequential disk locality (which is bad for throughput), 
    # we allow this inefficiency here because validation happens very rarely (e.g. once every few minutes).
    max_start = total_tokens - tokens_needed
    start_offset = random.randint(0, max_start) if max_start > 0 else 0

    val_loader = FastTokenLoader(val_cache_path, num_batches * batch_size, seq_len, start_idx=start_offset)
    data_x, data_y, _ = val_loader.get_batch()

    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            x = data_x[start:end].to(device)
            y = data_y[start:end].to(device)
            _, loss = model(x, targets=y)
            total_loss += loss.item() * seq_len
            total_tokens += seq_len
    model.train()
    return math.exp(total_loss / total_tokens)
