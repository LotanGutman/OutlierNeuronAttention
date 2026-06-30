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

def download_and_tokenize(config: LanguageModelingExperimentConfig):
    """
    Downloads and tokenizes FineWeb-Edu, saving the tokens into a binary file.
    If the binary file already exists and has the required size, it skips downloading.
    """
    cache_path = f"data/datasets/data_{config.model_name}_cache.bin"
    max_tokens = config.max_tokens
    dataset_name = config.dataset_name
    dataset_config = config.dataset_config

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

    print(f"Loading {dataset_name} ({dataset_config}) in streaming mode...")
    dataset = load_dataset(dataset_name, name=dataset_config, split=split, streaming=True)
    
    enc = tiktoken.get_encoding(tokenizer_name)
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
                text = example["text"]
                tokens = enc.encode_ordinary(text)
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
