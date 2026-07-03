import os
import sys
import types
try:
    import lzma
except ImportError:
    # Patch for Python environments missing the _lzma C extension (like custom WSL builds)
    mock_lzma = types.ModuleType('lzma')
    mock_lzma.LZMAFile = None
    mock_lzma.LZMAError = Exception
    mock_lzma.open = None
    sys.modules['lzma'] = mock_lzma

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from training.training_config import LanguageModelingExperimentConfig

def download_and_tokenize(config: LanguageModelingExperimentConfig, split: str = "train", num_batches: int = 25):
    """
        Downloads and tokenizes FineWeb-Edu in streaming mode, saving tokens into a binary file.

        - `split` can be "train" or "validation".
        - For `train` the function uses `config.max_tokens` as the target token count.
        - For `validation` `num_batches` (default 25) is used and the target token count is
            `num_batches * config.batch_size * config.seq_len`.

    """
    dataset_name = config.dataset_name
    dataset_config = config.dataset_config

    if split == "train":
        cache_path = f"data/datasets/data_{config.model_name}_cache.bin"
        max_tokens = config.max_tokens
    elif split in ("val", "validation"):
        cache_path = f"data/datasets/data_{config.model_name}_val_cache.bin"
        max_tokens = int(num_batches) * int(config.batch_size) * int(config.seq_len)
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train' or 'validation'.")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # Check if we already have enough tokens
    if os.path.exists(cache_path):
        current_size_bytes = os.path.getsize(cache_path)
        current_tokens = current_size_bytes // 2  # np.uint16 = 2 bytes
        if current_tokens >= max_tokens:
            print(f"Cache file {cache_path} already exists with {current_tokens} tokens. Skipping download.")
            return cache_path
        else:
            print(f"Found cache with {current_tokens} tokens, but requested {max_tokens}. Rebuilding...")

    desired_split = "train" if split == "train" else "validation"
    print(f"Loading {dataset_name} ({dataset_config}) split={desired_split} in streaming mode...")
    try:
        dataset = load_dataset(dataset_name, name=dataset_config, split=desired_split, streaming=True)
        load_split = desired_split
    except ValueError as e:
        print(f"Requested split '{desired_split}' not available; falling back to 'train'. ({e})")
        dataset = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
        load_split = "train (fallback)"

    enc = tiktoken.get_encoding(config.model_config.tokenizer_name)
    eot_token = enc.eot_token
    
    print(f"Tokenizing and caching up to {max_tokens} tokens to {cache_path}...")
    
    total_tokens = 0
    buffer_size = 1_000_000
    buffer = np.zeros(buffer_size, dtype=np.uint16)
    buffer_idx = 0
    
    # Write to a temporary file first for safety
    tmp_path = cache_path + ".tmp"
    
    with open(tmp_path, "wb") as f:
        with tqdm(total=max_tokens, unit="tok") as pbar:
            for example in dataset:
                # dataset examples may have different text keys; assume 'text'
                text = example["text"]
                tokens = enc.encode_ordinary(text)
                # add an explicit EOT if tokenizer provides one
                if eot_token is not None:
                    tokens.append(eot_token)

                for token in tokens:
                    buffer[buffer_idx] = token
                    buffer_idx += 1
                    total_tokens += 1

                    if buffer_idx == buffer_size:
                        f.write(buffer.tobytes())
                        buffer_idx = 0

                    if total_tokens >= max_tokens:
                        break

                pbar.update(len(tokens))

                if total_tokens >= max_tokens:
                    break
                    
        # Write any remaining tokens in buffer
        if buffer_idx > 0:
            f.write(buffer[:buffer_idx].tobytes())
            
    os.replace(tmp_path, cache_path)
    print(f"Successfully cached {total_tokens} tokens to {cache_path}.")
    return cache_path

if __name__ == "__main__":
    config = LanguageModelingExperimentConfig()
    download_and_tokenize(config)
