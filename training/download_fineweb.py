"""
FineWeb-Edu Downloader and Tokenizer

This script prepares validation and training data for language modeling.
It conditionally alters its loading strategy based on target token count to prevent
network timeouts and memory OOM kills on massive datasets:

- Target <= 10B tokens: Streams 'sample-10BT' via HuggingFace datasets with local shuffling.
- Target > 10B tokens: Fetches raw parquets from 'sample-100BT' and streams via PyArrow.
  This bypasses HF dataset memory leaks and natively skips documents instantly via metadata.
  (Note: Raw parquets are already pre-shuffled by HuggingFace, so skipping a local buffer
  shuffle at this scale does not impact training diversity).

A single iterator is used sequentially for both validation and training to guarantee
disjoint sets. Output files are safely truncated to the exact token count requested.
"""

import os
import sys
import types
import json
import queue
import threading

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
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm
from training.training_config import LanguageModelingExperimentConfig

import itertools
from collections import deque


    

def download_and_tokenize(config: LanguageModelingExperimentConfig):
    dataset_name = "HuggingFaceFW/fineweb-edu"
    model_size = config.model_name.split('_')[0]

    base_dir = os.path.expanduser("data/datasets")
    os.makedirs(base_dir, exist_ok=True)
    
    val_cache_path = f"data/datasets/data_{model_size}_val_cache.bin"
    train_cache_path = f"data/datasets/data_{model_size}_cache.bin"
    offset_path = f"data/datasets/offset_state_{model_size}.json"

    # Evaluation needs seq_len + 1 tokens per sequence to form input and target pairs
    val_max_tokens = int(config.val_num_batches) * int(config.batch_size) * (int(config.seq_len) + 1)
    train_max_tokens = config.max_tokens
    total_requested_tokens = val_max_tokens + train_max_tokens
    
    # Load offset state
    if os.path.exists(offset_path):
        with open(offset_path, "r") as f:
            offset_state = json.load(f)
            
        for split_path, split_key in [(val_cache_path, "validation"), (train_cache_path, "training")]:
            if offset_state.get(f"{split_key}_tokens", 0) > 0 and not os.path.exists(split_path):
                raise RuntimeError(
                    f"Data inconsistency: {offset_path} claims {split_key} has "
                    f"{offset_state.get(f'{split_key}_tokens')} tokens, but {split_path} is missing!\n"
                    f"Did you manually delete the .bin file? Please delete {offset_path} and any other cache files for this model_size to start fresh."
                )
    else:
        for split_path in [val_cache_path, train_cache_path]:
            if os.path.exists(split_path):
                raise RuntimeError(
                    f"{split_path} exists but {offset_path} is missing — cannot safely resume. "
                    f"Delete {split_path} (and any other cache files for this model_size) and rerun from scratch."
                )
        offset_state = {
            "validation_docs": 0, "validation_tokens": 0,
            "training_docs": 0, "training_tokens": 0
        }

    # Safety truncate files to match offset state
    for split_path, split_key in [(val_cache_path, "validation"), (train_cache_path, "training")]:
        if os.path.exists(split_path):
            expected_bytes = offset_state[f"{split_key}_tokens"] * 2
            actual_bytes = os.path.getsize(split_path)
            if actual_bytes > expected_bytes:
                print(f"Warning: {split_path} is larger than expected ({actual_bytes} > {expected_bytes}). Truncating to safe boundary.")
                os.truncate(split_path, expected_bytes)

    val_current = os.path.getsize(val_cache_path) // 2 if os.path.exists(val_cache_path) else 0
    train_current = os.path.getsize(train_cache_path) // 2 if os.path.exists(train_cache_path) else 0

    if val_current >= val_max_tokens and train_current >= train_max_tokens:
        print("Both validation and training caches already exist with enough tokens. Skipping download.")
        return val_cache_path, train_cache_path
    
    # Conditional loading logic based on required size
    if total_requested_tokens > 10_000_000_000:
        num_parquets = int(total_requested_tokens / 750_000_000) + 2
        print(f"Target size ({total_requested_tokens} tokens) > 10B. Fetching file list from HF...")
        
        all_files = list_repo_files(dataset_name, repo_type="dataset")
        valid_files = sorted([f for f in all_files if f.startswith("sample/100BT/") and f.endswith(".parquet")])
        data_files = valid_files[:num_parquets]
        
        print(f"Downloading {len(data_files)} raw parquets...")
        local_files = []
        for hf_file in data_files:
            local_path = hf_hub_download(
                repo_id=dataset_name, 
                filename=hf_file, 
                repo_type="dataset"
            )
            local_files.append(local_path)
            
        import pyarrow.parquet as pq

        def raw_parquet_iterator(files, skip_count):
            skipped = 0
            for file_path in files:
                pf = pq.ParquetFile(file_path)
                file_rows = pf.metadata.num_rows
                
                # Instantly skip entire files via metadata without reading a single byte
                if skipped + file_rows <= skip_count:
                    skipped += file_rows
                    continue
                    
                for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
                    for text in batch["text"].to_pylist():
                        if skipped < skip_count:
                            skipped += 1
                            continue
                        yield {"text": text}
                        
        total_skip = offset_state["validation_docs"] + offset_state["training_docs"]
        print(f"Instantly skipping {total_skip} documents and streaming via raw PyArrow...")
        dataset_iter = raw_parquet_iterator(local_files, total_skip)
        
    else:
        print(f"Target size ({total_requested_tokens} tokens) <= 10B. Streaming sample-10BT...")
        dataset = load_dataset(
            dataset_name, 
            name="sample-10BT", 
            split="train", 
            streaming=True,
            cache_dir=hf_cache_dir
        )
        dataset = dataset.shuffle(seed=config.seed, buffer_size=10000)
        
        total_skip = offset_state["validation_docs"] + offset_state["training_docs"]
        dataset_iter = iter(dataset)
        
        if total_skip > 0:
            print(f"Resuming stream: fast-forwarding {total_skip} documents...")
            import itertools
            from collections import deque
            chunk_size = 50_000
            with tqdm(total=total_skip, unit="doc", desc="Skipping") as pbar:
                for _ in range(total_skip // chunk_size):
                    deque(itertools.islice(dataset_iter, chunk_size), maxlen=0)
                    pbar.update(chunk_size)
                remainder = total_skip % chunk_size
                if remainder > 0:
                    deque(itertools.islice(dataset_iter, remainder), maxlen=0)
                    pbar.update(remainder)

            print("Fast-forward complete. Starting tokenization pipelines...")

    # Global background worker for prefetching to overlap network I/O with tokenization
    batch_queue = queue.Queue(maxsize=3)
    stop_event = threading.Event()
    
    def prefetch_worker():
        try:
            batch = []
            for example in dataset_iter:
                if stop_event.is_set():
                    break
                batch.append(example["text"])
                if len(batch) >= 500:
                    batch_queue.put(batch)
                    batch = []
            if batch and not stop_event.is_set():
                batch_queue.put(batch)
            batch_queue.put(None)  # Sentinel
        except Exception as e:
            batch_queue.put(e)
            
    prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
    prefetch_thread.start()

    enc = tiktoken.get_encoding(config.model_config.tokenizer_name)
    eot_token = enc.eot_token
    
    assert enc.n_vocab <= 65536, "Vocab size > 65536, uint16 array will overflow!"

    FSYNC_EVERY_N_BATCHES = 20

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
        
        batches_since_fsync = 0
        docs_since_fsync = 0
        
        with open(target_path, "ab" if current_tokens > 0 else "wb") as f:
            with tqdm(total=max_tokens, initial=current_tokens, unit="tok") as pbar:
                
                def flush_batch(batch_texts):
                    nonlocal total_tokens, buffer_idx, batches_since_fsync, docs_since_fsync
                    if not batch_texts:
                        return False
                    
                    docs_in_batch = len(batch_texts)
                    
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
                            
                        # Fast array buffer writing
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
                            break
                            
                    docs_since_fsync += docs_in_batch
                    batches_since_fsync += 1
                    
                    should_fsync = (batches_since_fsync >= FSYNC_EVERY_N_BATCHES) or (total_tokens >= max_tokens)
                    
                    if should_fsync:
                        # Flush the remaining buffer to disk atomically for this batch
                        if buffer_idx > 0:
                            f.write(buffer[:buffer_idx].tobytes())
                            buffer_idx = 0
                            
                        f.flush()
                        os.fsync(f.fileno())
                        
                        # Update state
                        offset_state[f"{split_name}_docs"] += docs_since_fsync
                        offset_state[f"{split_name}_tokens"] = total_tokens
                        
                        # Atomically save JSON
                        tmp_offset = offset_path + ".tmp"
                        with open(tmp_offset, "w") as jf:
                            json.dump(offset_state, jf)
                        os.replace(tmp_offset, offset_path)
                        
                        docs_since_fsync = 0
                        batches_since_fsync = 0
                    
                    if total_tokens >= max_tokens:
                        return True
                    return False

                while True:
                    batch_texts = batch_queue.get()
                    if batch_texts is None:
                        # Final flush of buffer and state if stream ends early
                        if buffer_idx > 0:
                            f.write(buffer[:buffer_idx].tobytes())
                            
                        f.flush()
                        os.fsync(f.fileno())
                        
                        offset_state[f"{split_name}_docs"] += docs_since_fsync
                        offset_state[f"{split_name}_tokens"] = total_tokens
                        tmp_offset = offset_path + ".tmp"
                        with open(tmp_offset, "w") as jf:
                            json.dump(offset_state, jf)
                        os.replace(tmp_offset, offset_path)
                        break # End of dataset
                    if isinstance(batch_texts, Exception):
                        raise batch_texts
                        
                    if flush_batch(batch_texts):
                        break
                        
        print(f"Successfully cached {total_tokens} tokens to {target_path}.")
        if total_tokens < max_tokens:
            print(f"Warning: Stream ended early! Only collected {total_tokens} tokens instead of {max_tokens}!")

    try:
        # Always process validation first from the start of the stream
        process_split(val_cache_path, val_max_tokens, "validation")
        # Then immediately process training from where validation left off
        process_split(train_cache_path, train_max_tokens, "training")
    finally:
        # Ensure thread exits cleanly
        stop_event.set()

    return val_cache_path, train_cache_path

if __name__ == "__main__":
    config = LanguageModelingExperimentConfig()
    download_and_tokenize(config)