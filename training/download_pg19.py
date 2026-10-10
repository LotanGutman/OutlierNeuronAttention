"""
PG19 dataset preparation for long-context continual pretraining and evaluation.
Tokenizes books with tiktoken (gpt2) into uint16 flat binaries compatible with FastTokenLoader.
Windows are packed strictly within individual books so sequences never span book boundaries.
"""

import os
import json
import time
import hashlib
import itertools
from typing import Dict, Iterable, Iterator, List, Optional
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from training.training_config import LanguageModelingExperimentConfig, DatasetType

HF_DATASET = "emozilla/pg19"
LENGTH_THRESHOLDS = (1024, 4096, 8192, 16384, 32768, 65536, 131072)


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_write_json(obj: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def batched(iterable: Iterable, n: int) -> Iterator[list]:
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def length_stats(lengths: List[int]) -> dict:
    arr = np.asarray(lengths, dtype=np.int64)
    if arr.size == 0:
        return {"num_books": 0}
    return {
        "num_books": int(arr.size),
        "total_tokens": int(arr.sum()),
        "mean_tokens": float(arr.mean()),
        "median_tokens": float(np.median(arr)),
        "min_tokens": int(arr.min()),
        "max_tokens": int(arr.max()),
        "num_books_with_at_least_tokens": {str(t): int((arr >= t).sum()) for t in LENGTH_THRESHOLDS},
    }


def iter_books_hf(split: str, shuffle_seed: Optional[int] = None, buffer_size: int = 64) -> Iterator[dict]:
    ds = load_dataset(HF_DATASET, split=split, streaming=True)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=buffer_size)
    for ex in ds:
        yield {
            "title": ex.get("short_book_title", ""),
            "url": ex.get("url", ""),
            "date": ex.get("publication_date"),
            "text": ex["text"],
        }


def build_window_split(
    books: Iterable[dict],
    enc: tiktoken.Encoding,
    split: str,
    out_path: str,
    window: int,
    max_windows: Optional[int],
    batch_books: int = 8,
    num_threads: int = 8,
) -> dict:
    """Packs non-overlapping (seq_len + 1) windows strictly inside individual books."""
    tmp = out_path + ".tmp"
    sha = hashlib.sha256()
    entries: List[dict] = []
    n_windows = 0
    books_seen = 0
    short_books = 0
    remainder_dropped = 0
    done = False
    t0 = time.time()

    pbar_total = max_windows if max_windows is not None else None
    pbar = tqdm(total=pbar_total, desc=f"{split} windows", unit="win")

    with open(tmp, "wb") as f:
        for chunk in batched(books, batch_books):
            texts = [b["text"] for b in chunk]
            toks_list = [np.asarray(t, dtype=np.uint16) for t in enc.encode_ordinary_batch(texts, num_threads=num_threads)]

            for b, toks in zip(chunk, toks_list):
                book_idx = books_seen
                books_seen += 1
                n_full = len(toks) // window
                if n_full == 0:
                    short_books += 1
                    continue

                n_take = n_full
                if max_windows is not None:
                    n_take = min(n_take, max_windows - n_windows)

                data = toks[: n_take * window].tobytes()
                f.write(data)
                sha.update(data)
                remainder_dropped += len(toks) - n_take * window

                entries.append(
                    {
                        "book_idx": book_idx,
                        "title": b["title"],
                        "url": b["url"],
                        "date": b["date"],
                        "book_tokens": int(len(toks)),
                        "first_window": n_windows,
                        "num_windows": int(n_take),
                    }
                )
                n_windows += n_take
                pbar.update(n_take)

                if max_windows is not None and n_windows >= max_windows:
                    done = True
                    break

            if done:
                break
        f.flush()
        os.fsync(f.fileno())
    pbar.close()
    os.replace(tmp, out_path)

    elapsed = time.time() - t0
    tokens = n_windows * window
    print(f"[{split}] {n_windows:,} windows ({tokens:,} tokens) from {len(entries)} books ({elapsed:.1f}s)")

    return {
        "path": out_path,
        "num_windows": n_windows,
        "window_tokens": window,
        "tokens": tokens,
        "bytes": tokens * 2,
        "sha256": sha.hexdigest(),
        "books_seen": books_seen,
        "books_used": len(entries),
        "books_shorter_than_window_skipped": short_books,
        "remainder_tokens_dropped": int(remainder_dropped),
        "entries": entries,
        "book_length_stats": length_stats([e["book_tokens"] for e in entries]),
    }


def build_book_level_split(
    books: Iterable[dict],
    enc: tiktoken.Encoding,
    split: str,
    bin_path: str,
    index_path: str,
    batch_books: int = 8,
    num_threads: int = 8,
) -> dict:
    """Concatenates full unbroken books and generates an index with token offsets."""
    tmp = bin_path + ".tmp"
    sha = hashlib.sha256()
    entries: List[dict] = []
    pos = 0
    t0 = time.time()

    pbar = tqdm(desc=f"{split} full books", unit="book")
    with open(tmp, "wb") as f:
        for chunk in batched(books, batch_books):
            texts = [b["text"] for b in chunk]
            toks_list = [np.asarray(t, dtype=np.uint16) for t in enc.encode_ordinary_batch(texts, num_threads=num_threads)]

            for b, toks in zip(chunk, toks_list):
                data = toks.tobytes()
                f.write(data)
                sha.update(data)
                entries.append(
                    {
                        "book_idx": len(entries),
                        "title": b["title"],
                        "url": b["url"],
                        "date": b["date"],
                        "start": pos,
                        "length": int(len(toks)),
                    }
                )
                pos += len(toks)
                pbar.update(1)
        f.flush()
        os.fsync(f.fileno())
    pbar.close()
    os.replace(tmp, bin_path)

    stats = length_stats([e["length"] for e in entries])
    index = {
        "split": split,
        "total_tokens": pos,
        "books": entries,
        "length_stats": stats,
        "dtype": "uint16",
    }
    atomic_write_json(index, index_path)

    elapsed = time.time() - t0
    print(f"[{split}] {len(entries)} full books ({pos:,} tokens) written ({elapsed:.1f}s)")

    return {
        "path": bin_path,
        "index_path": index_path,
        "tokens": pos,
        "bytes": pos * 2,
        "sha256": sha.hexdigest(),
        "num_books": len(entries),
        "entries": entries,
        "length_stats": stats,
    }


def download_and_tokenize_pg19(config: LanguageModelingExperimentConfig, base_dir: str = "data/datasets"):
    os.makedirs(base_dir, exist_ok=True)

    tokenizer_name = config.model_config.tokenizer_name
    enc = tiktoken.get_encoding(tokenizer_name)
    assert enc.n_vocab <= 65536, (
        f"Tokenizer '{tokenizer_name}' vocab size {enc.n_vocab} > 65536. "
        f"FastTokenLoader requires uint16 format."
    )

    seq_len = config.seq_len
    window = seq_len + 1  # input + shifted target
    tag = f"{seq_len // 1024}k" if seq_len % 1024 == 0 else str(seq_len)

    train_tokens = config.max_tokens
    val_tokens = int(config.val_num_batches) * int(config.batch_size) * window
    if val_tokens == 0:
        val_tokens = 5_000_000

    max_train_windows = train_tokens // window
    max_val_windows = val_tokens // window

    paths = {
        "train_bin": os.path.join(base_dir, f"pg19_{tag}_train.bin"),
        "train_index": os.path.join(base_dir, f"pg19_{tag}_train_index.json"),
        "val_bin": os.path.join(base_dir, f"pg19_{tag}_val.bin"),
        "val_index": os.path.join(base_dir, f"pg19_{tag}_val_index.json"),
        "val_books_bin": os.path.join(base_dir, "pg19_val_books.bin"),
        "val_books_index": os.path.join(base_dir, "pg19_val_books_index.json"),
        "test_books_bin": os.path.join(base_dir, "pg19_test_books.bin"),
        "test_books_index": os.path.join(base_dir, "pg19_test_books_index.json"),
        "manifest": os.path.join(base_dir, f"pg19_{tag}_manifest.json"),
    }

    if os.path.exists(paths["manifest"]) and all(os.path.exists(p) for p in paths.values()):
        print(f"PG19 dataset cache already exists at '{base_dir}'. Verified via {paths['manifest']}.")
        return

    print(f"Preparing PG19: seq_len={seq_len:,}, train_tokens={train_tokens:,}, target_dir={base_dir}")

    results = {}
    num_threads = max(1, (os.cpu_count() or 4) - 1)

    print("Processing train split...")
    r_train = build_window_split(
        books=iter_books_hf("train", shuffle_seed=config.seed),
        enc=enc,
        split="train",
        out_path=paths["train_bin"],
        window=window,
        max_windows=max_train_windows,
        batch_books=8,
        num_threads=num_threads,
    )
    atomic_write_json({"split": "train", "window_tokens": window, "books": r_train["entries"]}, paths["train_index"])
    results["train_windows"] = {k: v for k, v in r_train.items() if k != "entries"}

    print("Processing validation windows...")
    r_val = build_window_split(
        books=iter_books_hf("validation", shuffle_seed=None),
        enc=enc,
        split="validation",
        out_path=paths["val_bin"],
        window=window,
        max_windows=max_val_windows,
        batch_books=8,
        num_threads=num_threads,
    )
    atomic_write_json({"split": "validation", "window_tokens": window, "books": r_val["entries"]}, paths["val_index"])
    results["val_windows"] = {k: v for k, v in r_val.items() if k != "entries"}

    print("Processing validation full books...")
    r_val_books = build_book_level_split(
        books=iter_books_hf("validation", shuffle_seed=None),
        enc=enc,
        split="validation",
        bin_path=paths["val_books_bin"],
        index_path=paths["val_books_index"],
        batch_books=8,
        num_threads=num_threads,
    )
    results["val_books"] = {k: v for k, v in r_val_books.items() if k != "entries"}

    print("Processing test full books...")
    r_test_books = build_book_level_split(
        books=iter_books_hf("test", shuffle_seed=None),
        enc=enc,
        split="test",
        bin_path=paths["test_books_bin"],
        index_path=paths["test_books_index"],
        batch_books=8,
        num_threads=num_threads,
    )
    results["test_books"] = {k: v for k, v in r_test_books.items() if k != "entries"}

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_source": HF_DATASET,
        "tokenizer": tokenizer_name,
        "seq_len": seq_len,
        "window_tokens": window,
        "train_tokens_target": train_tokens,
        "val_tokens_target": val_tokens,
        "paths": paths,
        "results": results,
    }
    atomic_write_json(manifest, paths["manifest"])
    print(f"PG19 preparation complete. Manifest saved to {paths['manifest']}")
