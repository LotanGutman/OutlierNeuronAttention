import os
import torch
import numpy as np

class FastTokenLoader:
    def __init__(self, cache_path: str, global_batch_size: int, seq_len: int, start_idx: int = 0):
        self.cache_path = cache_path
        self.global_batch_size = global_batch_size
        self.seq_len = seq_len
        self.tokens_per_batch = global_batch_size * (seq_len + 1)
        
        file_size_bytes = os.path.getsize(cache_path)
        self.total_tokens = file_size_bytes // 2
        
        # Memory map the dataset directly for OS-level I/O optimization
        self.mmap = np.memmap(cache_path, dtype=np.uint16, mode='r')
        self.current_idx = start_idx
        
    def get_batch(self):
        """Returns x, y (pinned CPU tensors), and the next start_idx"""
        if self.current_idx + self.tokens_per_batch > self.total_tokens:
            self.current_idx = 0  # Wrap around
            
        # Slice from the memmap (lazy loaded by OS)
        chunk = self.mmap[self.current_idx : self.current_idx + self.tokens_per_batch]
        
        # Cast to int64, convert to tensor, and pin memory for fast async H2D transfer
        data = torch.from_numpy(chunk.astype(np.int64)).pin_memory()
        data = data.view(self.global_batch_size, self.seq_len + 1)
        
        self.current_idx += self.tokens_per_batch
        
        x = data[:, :-1].contiguous()
        y = data[:, 1:].contiguous()
        
        return x, y, self.current_idx
