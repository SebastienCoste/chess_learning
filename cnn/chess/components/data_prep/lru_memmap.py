from collections import OrderedDict
import threading
import numpy as np
import torch
from torch.utils.data import Dataset
import psutil


class LRUCachedMemmapDataset(Dataset):
    def __init__(self, base_path, cache_size_mb=8192):  # 8GB cache by default
        self.base_path = base_path
        meta = np.load(f"{base_path}_meta.npz", allow_pickle=True)
        self.length = int(meta['length'])
        self.metadata = meta

        # Calculate cache capacity based on memory
        sample_size = (19 * 8 * 8 + 4096) * 4  # bytes per sample (float32)
        max_samples = (cache_size_mb * 1024 * 1024) // sample_size
        self.cache_capacity = min(max_samples, self.length // 4)  # Cache 25% max

        # LRU cache with thread safety
        self.cache = OrderedDict()
        self.cache_lock = threading.RLock()
        self.cache_hits = 0
        self.cache_misses = 0

        # Memory-mapped files (lazy loaded)
        self._inputs = None
        self._outputs = None

    def _ensure_loaded(self):
        """Lazy loading of memory-mapped files"""
        if self._inputs is None:
            self._inputs = np.memmap(f"{self.base_path}_inputs.dat",
                                     dtype=np.float32, mode='r',
                                     shape=(self.length, 19, 8, 8))
            self._outputs = np.memmap(f"{self.base_path}_outputs.dat",
                                      dtype=np.float32, mode='r',
                                      shape=(self.length, 4096))

    def _get_from_cache(self, idx):
        """Thread-safe cache access with LRU update"""
        with self.cache_lock:
            if idx in self.cache:
                # Move to end (most recently used)
                data = self.cache.pop(idx)
                self.cache[idx] = data
                self.cache_hits += 1
                return data
            return None

    def _put_in_cache(self, idx, data):
        """Thread-safe cache insertion with LRU eviction"""
        with self.cache_lock:
            # Remove oldest items if cache is full
            while len(self.cache) >= self.cache_capacity:
                oldest_idx = next(iter(self.cache))
                del self.cache[oldest_idx]

            self.cache[idx] = data
            self.cache_misses += 1

    def __getitem__(self, idx):
        # Try cache first
        cached_data = self._get_from_cache(idx)
        if cached_data is not None:
            return cached_data

        # Load from memory-mapped file
        self._ensure_loaded()
        input_data = self._inputs[idx].copy()
        output_data = self._outputs[idx].copy()

        # Convert to tensors
        input_tensor = torch.from_numpy(input_data)
        output_tensor = torch.from_numpy(output_data)

        # Cache the tensors
        data = (input_tensor, output_tensor)
        self._put_in_cache(idx, data)

        return data

    def get_cache_stats(self):
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'capacity': self.cache_capacity,
            'hits': self.cache_hits,
            'misses': self.cache_misses
        }

    def __len__(self):
        return self.length
