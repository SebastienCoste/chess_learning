import ctypes
import ctypes.util
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class PinnedMemoryCache:
    def __init__(self, size_mb=1024):  # 1GB pinned cache
        self.size_bytes = size_mb * 1024 * 1024

        # Allocate pinned memory
        libc = ctypes.CDLL(ctypes.util.find_library("c"))

        # Allocate memory
        self.memory = ctypes.create_string_buffer(self.size_bytes)

        # Pin memory using mlock
        MCL_CURRENT = 1
        MCL_FUTURE = 2

        result = libc.mlock(
            ctypes.cast(self.memory, ctypes.c_void_p),
            ctypes.c_size_t(self.size_bytes)
        )

        if result != 0:
            print(f"Warning: mlock failed with code {result}")
        else:
            print(f"Successfully pinned {size_mb}MB in memory")

        # Cache management
        self.cache = {}
        self.next_offset = 0

    def store(self, key, data):
        """Store data in pinned memory"""
        data_bytes = pickle.dumps(data)
        data_size = len(data_bytes)

        if self.next_offset + data_size > self.size_bytes:
            return False  # Not enough space

        # Copy to pinned memory
        ctypes.memmove(
            ctypes.byref(self.memory, self.next_offset),
            data_bytes,
            data_size
        )

        self.cache[key] = (self.next_offset, data_size)
        self.next_offset += data_size
        return True

    def retrieve(self, key):
        """Retrieve data from pinned memory"""
        if key not in self.cache:
            return None

        offset, size = self.cache[key]
        data_bytes = ctypes.string_at(
            ctypes.byref(self.memory, offset),
            size
        )

        return pickle.loads(data_bytes)

    def __del__(self):
        """Cleanup pinned memory"""
        if hasattr(self, 'memory'):
            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            libc.munlock(
                ctypes.cast(self.memory, ctypes.c_void_p),
                ctypes.c_size_t(self.size_bytes)
            )
