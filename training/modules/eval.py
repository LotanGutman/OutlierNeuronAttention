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

    # Read a fixed slice (starting at 0 for simplicity)
    max_start = total_tokens - tokens_needed
    start_offset = random.randint(0, max_start) if max_start > 0 else 0

    with open(val_cache_path, 'rb') as f:
        f.seek(start_offset * 2)
        chunk_bytes = f.read(tokens_needed * 2)

    data = np.frombuffer(chunk_bytes, dtype=np.uint16).astype(np.int64)
    data = torch.from_numpy(data).view(num_batches * batch_size, seq_len + 1)

    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch = data[start:end]  # shape (batch_size, seq_len+1)
            x = batch[:, :-1].contiguous().to(device)
            y = batch[:, 1:].contiguous().to(device)
            _, loss = model(x, targets=y)
            total_loss += loss.item() * seq_len
            total_tokens += seq_len
    model.train()
    return math.exp(total_loss / total_tokens)
