import mmap
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import ctypes
from ctypes import c_void_p, c_size_t, c_int

# Load the C library for madvise calls
libc = ctypes.CDLL("libc.so.6")

# madvise constants
MADV_RANDOM = 1  # Expect random page references
MADV_SEQUENTIAL = 2  # Expect sequential page references
MADV_WILLNEED = 3  # Will need these pages soon
MADV_DONTNEED = 4  # Don't need these pages


class OptimizedMemmapChessDataset(Dataset):
    def __init__(self, base_path):
        self.base_path = base_path
        # Store metadata but don't open files yet
        meta = np.load(f"{base_path}_meta.npz", allow_pickle=True)
        self.length = int(meta['length'])
        self.metadata = meta

        # Don't store file handles - these will be opened per-worker
        self._inputs = None
        self._outputs = None
        self._input_fd = None
        self._output_fd = None

    def _ensure_loaded(self):
        """Lazy loading with Linux optimization hints"""
        if self._inputs is None:
            # Open files and get file descriptors
            self._input_fd = os.open(f"{self.base_path}_inputs.dat", os.O_RDONLY)
            self._output_fd = os.open(f"{self.base_path}_outputs.dat", os.O_RDONLY)

            self._inputs = np.memmap(f"{self.base_path}_inputs.dat",
                                     dtype=np.float32, mode='r',
                                     shape=(self.length, 19, 8, 8))
            self._outputs = np.memmap(f"{self.base_path}_outputs.dat",
                                      dtype=np.float32, mode='r',
                                      shape=(self.length, 4096))

            # Provide memory access hints to the kernel
            self._optimize_memory_access()

    def _optimize_memory_access(self):
        """Use madvise to optimize memory access patterns"""
        input_size = self._inputs.nbytes
        output_size = self._outputs.nbytes

        # Get memory addresses
        input_addr = self._inputs.ctypes.data
        output_addr = self._outputs.ctypes.data

        # Tell kernel we'll access data randomly (not sequentially)
        libc.madvise(c_void_p(input_addr), c_size_t(input_size), c_int(MADV_RANDOM))
        libc.madvise(c_void_p(output_addr), c_size_t(output_size), c_int(MADV_RANDOM))

        # Tell kernel we'll need this data soon (preload into page cache)
        libc.madvise(c_void_p(input_addr), c_size_t(input_size), c_int(MADV_WILLNEED))
        libc.madvise(c_void_p(output_addr), c_size_t(output_size), c_int(MADV_WILLNEED))

    def __getitem__(self, idx):
        self._ensure_loaded()
        input_tensor = torch.from_numpy(self._inputs[idx].copy())
        output_tensor = torch.from_numpy(self._outputs[idx].copy())
        return input_tensor, output_tensor

    def __len__(self):
        return self.length

    def __del__(self):
        """Clean up file descriptors"""
        if self._input_fd is not None:
            os.close(self._input_fd)
        if self._output_fd is not None:
            os.close(self._output_fd)
