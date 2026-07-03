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
        Downloads and tokenizes FineWeb-Edu in streaming mode.
        It processes validation and training sets in a single pass to guarantee disjoint sets
        without needing any metadata tracking files.
    """
    dataset_name = config.dataset_name
    dataset_config = config.dataset_config
    
    model_size = config.model_name.split('_')[0]

    val_cache_path = f"data/datasets/data_{model_size}_val_cache.bin"
    train_cache_path = f"data/datasets/data_{model_size}_cache.bin"

    # Evaluation needs seq_len + 1 tokens per sequence to form input and target pairs
    val_max_tokens = int(config.val_num_batches) * int(config.batch_size) * (int(config.seq_len) + 1)
    train_max_tokens = config.max_tokens

    os.makedirs(os.path.dirname(train_cache_path), exist_ok=True)
    
    val_current = os.path.getsize(val_cache_path) // 2 if os.path.exists(val_cache_path) else 0
    train_current = os.path.getsize(train_cache_path) // 2 if os.path.exists(train_cache_path) else 0

    if val_current >= val_max_tokens and train_current >= train_max_tokens:
        print("Both validation and training caches already exist with enough tokens. Skipping download.")
        return val_cache_path, train_cache_path
    
    if val_current > 0 and val_current < val_max_tokens:
        print(f"Warning: Validation cache has only {val_current} tokens; need {val_max_tokens}. Will append missing tokens.")
    if train_current > 0 and train_current < train_max_tokens:
        print(f"Warning: Training cache has only {train_current} tokens; need {train_max_tokens}. Will append missing tokens.")

    print(f"Loading {dataset_name} ({dataset_config}) split='train' in streaming mode...")
    dataset = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
    dataset = dataset.shuffle(seed=config.seed, buffer_size=10000)

    dataset_iter = iter(dataset)

    enc = tiktoken.get_encoding(config.model_config.tokenizer_name)
    eot_token = enc.eot_token

    def process_split(target_path, max_tokens, split_name):
        current_tokens = os.path.getsize(target_path) // 2 if os.path.exists(target_path) else 0
        if current_tokens >= max_tokens:
            print(f"[{split_name}] Cache already has {current_tokens} tokens (needs {max_tokens}). Skipping.")
            return

        missing_tokens = max_tokens - current_tokens
        mode_str = "Appending" if current_tokens > 0 else "Tokenizing and caching"
        print(f"[{split_name}] {mode_str} {missing_tokens} missing tokens (out of {max_tokens} total) to {target_path}...")
        
        total_tokens = current_tokens
        buffer_size = 1_000_000
        buffer = np.zeros(buffer_size, dtype=np.uint16)
        buffer_idx = 0
        
        tmp_path = target_path + ".tmp"
        if current_tokens > 0:
            import shutil
            shutil.copy2(target_path, tmp_path)
            
        with open(tmp_path, "ab" if current_tokens > 0 else "wb") as f:
            with tqdm(total=max_tokens, initial=current_tokens, unit="tok") as pbar:
                batch_texts = []
                
                def flush_batch():
                    nonlocal total_tokens, buffer_idx
                    if not batch_texts:
                        return False
                    
                    # Multi-threaded batch encoding (the primary speedup)
                    encoded_batch = enc.encode_ordinary_batch(batch_texts, num_threads=max(1, (os.cpu_count() or 4) - 1))
                    batch_texts.clear()
                    
                    for tokens in encoded_batch:
                        if eot_token is not None:
                            tokens.append(eot_token)
                            
                        arr = np.array(tokens, dtype=np.uint16)
                        
                        # Stop exactly at max_tokens
                        if total_tokens + len(arr) > max_tokens:
                            arr = arr[:max_tokens - total_tokens]
                            
                        # Fast array buffer writing (the secondary speedup)
                        space_left = buffer_size - buffer_idx
                        if len(arr) < space_left:
                            buffer[buffer_idx:buffer_idx+len(arr)] = arr
                            buffer_idx += len(arr)
                        else:
                            buffer[buffer_idx:buffer_size] = arr[:space_left]
                            f.write(buffer.tobytes())
                            
                            rem = arr[space_left:]
                            while len(rem) >= buffer_size:
                                f.write(rem[:buffer_size].tobytes())
                                rem = rem[buffer_size:]
                                
                            buffer[:len(rem)] = rem
                            buffer_idx = len(rem)
                            
                        total_tokens += len(arr)
                        pbar.update(len(arr))
                        
                        if total_tokens >= max_tokens:
                            return True
                    return False

                for example in dataset_iter:
                    batch_texts.append(example["text"])
                    if len(batch_texts) >= 5000:
                        if flush_batch():
                            break

                # Flush any remaining text in the final incomplete batch
                if total_tokens < max_tokens:
                    flush_batch()
                        
            if buffer_idx > 0:
                f.write(buffer[:buffer_idx].tobytes())
                
        os.replace(tmp_path, target_path)
        print(f"Successfully cached {total_tokens} tokens to {target_path}.")

    # Always process validation first from the start of the stream
    process_split(val_cache_path, val_max_tokens, "validation")
    # Then immediately process training from where validation left off
    process_split(train_cache_path, train_max_tokens, "training")

    return val_cache_path, train_cache_path

if __name__ == "__main__":
    config = LanguageModelingExperimentConfig()
    download_and_tokenize(config)
