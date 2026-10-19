"""
Diagnostic script: test YaRN extrapolation with correct base_seq_len=1024 and s=16.0
across N in [1024, 2048, 4096, 8192, 16384, 32768, 65536] on pg19_test_books.bin
"""

import os
import sys
import math
import json
import torch
import torch.nn.functional as F
import numpy as np

from src.HybridOutlierFactorizedAttentionTrain import SubwordLM
from src.yarn import patch_model_yarn
from training.training_config import make_350M_HOFA_cpt, make_350M_MHA_cpt


def compute_chunked_loss(model, x, y, chunk_size=2048):
    total_tokens = x.shape[1]
    total_loss = 0.0
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        h = model.token_emb(x)
        for layer in model.layers:
            h = layer(h)
        h = model.ln_f(h)
        h_flat = h.view(-1, h.shape[-1])
        y_flat = y.view(-1)
        for i in range(0, total_tokens, chunk_size):
            end = min(i + chunk_size, total_tokens)
            sub_logits = model.lm_head(h_flat[i:end])
            sub_loss = F.cross_entropy(sub_logits, y_flat[i:end], reduction='sum')
            total_loss += sub_loss.item()
    return total_loss / total_tokens


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("VERIFYING YARN EXTRAPOLATION WITH CORRECT BASE_SEQ_LEN = 1024")
    print("=" * 80)

    test_bin = "data/datasets/pg19_test_books.bin"
    test_idx = "data/datasets/pg19_test_books_index.json"
    with open(test_idx, "r") as f:
        t_data = json.load(f)
    all_books = t_data["books"]
    mmap_test = np.memmap(test_bin, dtype=np.uint16, mode='r')

    for model_name, cpt_fn in [("350M_HOFA_cpt", make_350M_HOFA_cpt), ("350M_MHA_cpt", make_350M_MHA_cpt)]:
        print(f"\nEvaluating {model_name}...")
        cpt_config = cpt_fn()
        lm_cfg = cpt_config.model

        ckpt_path = f"data/training/{model_name}/checkpoint_best_val.pt"
        if not os.path.exists(ckpt_path):
            ckpt_path = f"data/training/{model_name}/checkpoint.pt"

        model = SubwordLM(lm_cfg.vocab_size, lm_cfg.model_config).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()

        # Test lengths up to 64k on 5 sample books
        sample_books = [b for b in all_books if b["length"] >= 65537][:5]

        for N in [1024, 2048, 4096, 8192, 16384, 32768, 65536]:
            # Patch model with target_seq_len=N, original_seq_len=1024, scale=16.0
            patch_model_yarn(
                model,
                target_seq_len=N,
                original_seq_len=1024,
                beta_fast=32.0,
                beta_slow=1.0,
                dtype=torch.bfloat16,
                scale=16.0,
            )

            book_losses = []
            for b in sample_books:
                start = b["start"]
                raw = mmap_test[start : start + N + 1]
                tensor = torch.from_numpy(raw.astype(np.int64)).unsqueeze(0).to(device)
                x = tensor[:, :-1]
                y = tensor[:, 1:]
                l = compute_chunked_loss(model, x, y)
                book_losses.append(l)

            mean_l = float(np.mean(book_losses))
            print(f"  [{model_name}] N = {N:5d} | Mean Loss: {mean_l:.4f} | PPL: {math.exp(mean_l):.2f}")


if __name__ == "__main__":
    main()
