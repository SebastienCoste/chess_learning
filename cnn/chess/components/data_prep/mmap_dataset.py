import numpy as np
import torch
from torch.utils.data import Dataset
import threading
import os

def convert_pickle_to_memmap(pickle_file, output_base):
    """Convert existing pickle data to memory-mapped format"""
    import pickle

    # Load existing data
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    # Create memory-mapped arrays
    inputs = np.memmap(f"{output_base}_inputs.dat", dtype=np.float32,
                       mode='w+', shape=(len(data), 19, 8, 8))
    outputs = np.memmap(f"{output_base}_outputs.dat", dtype=np.float32,
                        mode='w+', shape=(len(data), 4096))

    # Populate arrays
    for i, item in enumerate(data):
        inputs[i] = item['input']
        outputs[i] = item['output']

    # Save metadata separately (only non-tensor data)
    metadata = {
        'move_uci': [x['move_uci'] for x in data],
        'fen': [x['fen'] for x in data],
        'move_number': [x['move_number'] for x in data],
        'length': len(data)
    }
    np.savez(f"{output_base}_meta.npz", **metadata)

    return len(data)


class MemmapChessDataset(Dataset):
    def __init__(self, base_path):
        self.base_path = base_path
        # Store metadata but don't open files yet
        meta = np.load(f"{base_path}_meta.npz", allow_pickle=True)
        self.length = int(meta['length'])
        self.metadata = meta

        # Don't store file handles - these will be opened per-worker
        self._inputs = None
        self._outputs = None

    def _ensure_loaded(self):
        """Lazy loading of memory-mapped files in each worker process"""
        if self._inputs is None:
            self._inputs = np.memmap(f"{self.base_path}_inputs.dat",
                                     dtype=np.float32, mode='r',
                                     shape=(self.length, 19, 8, 8))
            self._outputs = np.memmap(f"{self.base_path}_outputs.dat",
                                      dtype=np.float32, mode='r',
                                      shape=(self.length, 4096))

    def __getitem__(self, idx):
        self._ensure_loaded()  # Open files if not already open
        input_tensor = torch.from_numpy(self._inputs[idx].copy())
        output_tensor = torch.from_numpy(self._outputs[idx].copy())
        return input_tensor, output_tensor

    def __len__(self):
        return self.length


# Global dictionary for process-specific arrays
process_arrays = {}


class MemmapChessDatasetWindows(Dataset):
    def __init__(self, base_path):
        self.base_path = base_path
        meta_path = f"{base_path}_meta.npz"
        self.length = int(np.load(meta_path, allow_pickle=True)['length'])

    def __getitem__(self, idx):
        pid = os.getpid()

        # Initialize arrays for this process if not exist
        if pid not in process_arrays:
            inputs = np.memmap(f"{self.base_path}_inputs.dat",
                               dtype=np.float32, mode='r',
                               shape=(self.length, 19, 8, 8))
            outputs = np.memmap(f"{self.base_path}_outputs.dat",
                                dtype=np.float32, mode='r',
                                shape=(self.length, 4096))
            process_arrays[pid] = (inputs, outputs)

        inputs, outputs = process_arrays[pid]
        return torch.from_numpy(inputs[idx].copy()), torch.from_numpy(outputs[idx].copy())

    def __len__(self):
        return self.length


class MemmapChessDatasetWindowsNoThread(Dataset):
    def __init__(self, base_path):
        self.base_path = base_path
        # Load metadata without keeping file handles
        meta_path = f"{base_path}_meta.npz"
        self.length = int(np.load(meta_path, allow_pickle=True)['length'])
        # No file handles stored at class level!

    def __getitem__(self, idx):
        # Open memmap files fresh for each access
        input_data = np.memmap(f"{self.base_path}_inputs.dat",
                               dtype=np.float32, mode='r',
                               shape=(self.length, 19, 8, 8))[idx].copy()

        output_data = np.memmap(f"{self.base_path}_outputs.dat",
                                dtype=np.float32, mode='r',
                                shape=(self.length, 4096))[idx].copy()

        return torch.from_numpy(input_data), torch.from_numpy(output_data)

    def __len__(self):
        return self.length


if __name__ == "__main__":
    convert_pickle_to_memmap("../data/all_train_data.pkl", "../data/all_train_data")
