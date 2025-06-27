from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import hashlib


class SharedMemoryCachedDataset(Dataset):
    def __init__(self, base_path, shared_cache_size_mb=4096):
        self.base_path = base_path
        meta = np.load(f"{base_path}_meta.npz", allow_pickle=True)
        self.length = int(meta['length'])
        self.metadata = meta

        # Calculate shared memory requirements
        self.sample_size = (19 * 8 * 8 + 4096) * 4  # bytes per sample
        self.cache_capacity = (shared_cache_size_mb * 1024 * 1024) // self.sample_size

        # Shared memory setup
        self.cache_memory = None
        self.cache_index = {}  # Maps sample_idx -> (offset, size) in shared memory
        self.next_offset = 0

        # Memory-mapped files
        self._inputs = None
        self._outputs = None

    def setup_shared_cache(self):
        """Setup shared memory cache (call from main process)"""
        cache_size_bytes = self.cache_capacity * self.sample_size
        self.cache_memory = shared_memory.SharedMemory(
            create=True,
            size=cache_size_bytes,
            name=f"chess_cache_{os.getpid()}"
        )
        return self.cache_memory.name

    def attach_to_shared_cache(self, cache_name):
        """Attach to existing shared cache (call from worker processes)"""
        self.cache_memory = shared_memory.SharedMemory(name=cache_name)

    def _ensure_loaded(self):
        """Lazy loading of memory-mapped files"""
        if self._inputs is None:
            self._inputs = np.memmap(f"{self.base_path}_inputs.dat",
                                     dtype=np.float32, mode='r',
                                     shape=(self.length, 19, 8, 8))
            self._outputs = np.memmap(f"{self.base_path}_outputs.dat",
                                      dtype=np.float32, mode='r',
                                      shape=(self.length, 4096))

    def _is_cached(self, idx):
        """Check if sample is in shared cache"""
        return idx in self.cache_index

    def _get_from_shared_cache(self, idx):
        """Retrieve sample from shared memory cache"""
        if idx not in self.cache_index:
            return None

        offset, size = self.cache_index[idx]

        # Read from shared memory
        cached_bytes = bytes(self.cache_memory.buf[offset:offset + size])
        input_tensor, output_tensor = pickle.loads(cached_bytes)

        return input_tensor, output_tensor

    def _put_in_shared_cache(self, idx, input_tensor, output_tensor):
        """Store sample in shared memory cache"""
        if len(self.cache_index) >= self.cache_capacity:
            return  # Cache full

        # Serialize tensors
        data_bytes = pickle.dumps((input_tensor, output_tensor))
        data_size = len(data_bytes)

        if self.next_offset + data_size > len(self.cache_memory.buf):
            return  # Not enough space

        # Write to shared memory
        self.cache_memory.buf[self.next_offset:self.next_offset + data_size] = data_bytes
        self.cache_index[idx] = (self.next_offset, data_size)
        self.next_offset += data_size

    def __getitem__(self, idx):
        # Try shared cache first
        if self.cache_memory is not None:
            cached_data = self._get_from_shared_cache(idx)
            if cached_data is not None:
                return cached_data

        # Load from memory-mapped file
        self._ensure_loaded()
        input_tensor = torch.from_numpy(self._inputs[idx].copy())
        output_tensor = torch.from_numpy(self._outputs[idx].copy())

        # Try to cache in shared memory
        if self.cache_memory is not None:
            self._put_in_shared_cache(idx, input_tensor, output_tensor)

        return input_tensor, output_tensor

    def __len__(self):
        return self.length

    def cleanup_shared_cache(self):
        """Cleanup shared memory (call from main process)"""
        if self.cache_memory is not None:
            self.cache_memory.close()
            self.cache_memory.unlink()
