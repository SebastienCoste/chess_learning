import atexit
import os
import platform
import random
import sys
from typing import List, Dict

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from cnn.chess.components.config import TRAINING_CONFIG
from cnn.chess.components.data_prep.lru_memmap import LRUCachedMemmapDataset
if not platform.system() == 'Windows':
    from cnn.chess.components.data_prep.optimized_memap import OptimizedMemmapChessDataset
from cnn.chess.components.data_prep.shared_cache_memmap import SharedMemoryCachedDataset
from cnn.chess.components.utils.chess_board_utils import board_to_tensor


class ChessTrainingDataGenerator:
    """
    Converts PGN chess games into training data for CNN.
    """

    def __init__(self,
                 #min_elo: int = 2000,
                 skip_opening_moves: int = 6,
                 #skip_endgame_moves: int = 10
                 ):
        #self.min_elo = min_elo
        self.skip_opening_moves = skip_opening_moves
        # self.skip_endgame_moves = skip_endgame_moves


    def move_to_index(self, move: chess.Move) -> int:
        """Convert move to index in 4096-dimensional output vector."""
        return move.from_square * 64 + move.to_square

    def process_pgn_string(self, game) -> List[Dict]:
        """Process PGN string and extract training positions."""
        # pgn_io = io.StringIO(pgn_string)
        # game = chess.pgn.read_game(pgn_io)

        if not game:
            return []

        # Check ELO requirements
        try:
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
        #     if white_elo < self.min_elo or black_elo < self.min_elo:
        #         print(f"Skipping game: ELO too low (W:{white_elo}, B:{black_elo})")
        #         return []
        except (ValueError, TypeError):
            # print("Could not parse ELO ratings")
            white_elo = 0
            black_elo = 0

        training_data = []
        board = game.board()
        moves = list(game.mainline_moves())
        is_puzzle = board.fullmove_number > 1
        # skip_one = is_puzzle #On puzzles skip the 1st move it's the mistake
        for move_idx, move in enumerate(moves):
            # Skip opening and endgame moves
            if not is_puzzle and move_idx < self.skip_opening_moves:
                board.push(move)
                continue


            # Create input tensor from current position
            for shape in ["normal", "flipped"]:
                input_tensor = board_to_tensor(board, shape == "flipped")

                # Create one-hot output vector
                output_vector = np.zeros(4096, dtype=np.float32)
                move_index = self.move_to_index(move)
                output_vector[move_index] = 1.0

                # Store training example
                training_data.append({
                    'input': input_tensor,
                    'output': output_vector,
                    'move_uci': move.uci(),
                    'fen': board.fen(),
                    'move_number': board.fullmove_number,
                    'game_info': {
                        'white': game.headers.get("White", "Unknown") if shape == "flipped" else game.headers.get("Black", "Unknown_flipped"),
                        'black': game.headers.get("Black", "Unknown") if shape == "flipped" else game.headers.get("White", "Unknown_flipped"),
                        'white_elo': white_elo if shape == "flipped" else black_elo,
                        'black_elo': black_elo if shape == "flipped" else white_elo,
                        'event': game.headers.get("Event", "Unknown")
                    }
                })

            board.push(move)

        return training_data

# Global dictionary for process-specific arrays
process_arrays = {}

@atexit.register
def cleanup():
    for pid, (inputs, outputs) in process_arrays.items():
        del inputs  # Close memmap files
        del outputs
    torch.cuda.empty_cache()

def create_optimized_dataloaders(datasets, base_path, batch_size=TRAINING_CONFIG["batch_size"], cache_type = TRAINING_CONFIG["cache_type"]):
    """Create CUDA-optimized data loaders"""
    worker_init_fn = None
    cached_datasets = []
    print(f"Received {len(datasets)} training and val sets")
    for ds in datasets:
        if cache_type == "lru":
            # Use LRU cache
            print("Using LRU cache")
            cached_dataset = LRUCachedMemmapDataset(ds.base_path)
        elif cache_type == "shared":
            # Use shared memory cache
            print("Using shared cache")
            cached_dataset = SharedMemoryCachedDataset(ds.base_path)
            cache_name = cached_dataset.setup_shared_cache()

            # Configure workers to attach to shared cache
            def worker_init_fn(worker_id):
                worker_dataset = cached_dataset
                worker_dataset.attach_to_shared_cache(cache_name)
        elif not cache_type == "none" and not platform.system() == 'Windows':
            # Use optimized memmap
            print("Using optimized cache")
            cached_dataset = OptimizedMemmapChessDataset(ds.base_path)
        else:
            cached_dataset = ds
        cached_datasets.append(cached_dataset)

    if len(cached_datasets) == 1:
        solo_ds = cached_datasets[0]
        # Split dataset
        train_size = int(0.8 * len(solo_ds))
        val_size = len(solo_ds) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            solo_ds, [train_size, val_size]
        )
        cached_datasets = [val_dataset, train_dataset]

    print(f"Using {len(cached_datasets)} training and val sets")

    is_windows = platform.system() == 'Windows'

    def worker_init_fn_win(worker_id):
        # Initialize arrays for this worker
        pid = os.getpid()
        base_path = worker_init_fn_win.base_path  # Set before creating DataLoader
        if pid not in process_arrays:
            inputs = np.memmap(f"{base_path}_inputs.dat",
                               dtype=np.float32, mode='r',
                               shape=(worker_init_fn_win.length, 19, 8, 8))
            outputs = np.memmap(f"{base_path}_outputs.dat",
                                dtype=np.float32, mode='r',
                                shape=(worker_init_fn_win.length, 4096))
            process_arrays[pid] = (inputs, outputs)

    if is_windows:
        # Before creating DataLoader
        worker_init_fn_win.base_path = base_path
        meta = np.load(f"{base_path}_meta.npz", allow_pickle=True)
        worker_init_fn_win.length = int(meta['length'])

    train_loaders = []
    # Optimized loader configuration
    for cd in cached_datasets[1::]:
        train_loaders.append(
            DataLoader(
                cd,
                batch_size=batch_size,
                shuffle=False,              # Disable shuffling for better cache locality
                num_workers=TRAINING_CONFIG["num_workers"],              # Reduced workers to prevent memory pressure
                pin_memory=True,           # Disable to reduce memory pressure
                prefetch_factor=2,          # Minimal prefetch
                persistent_workers= False, # because each epoch runs a different DS #not is_windows,    # Reuse workers
                worker_init_fn=worker_init_fn if cache_type == "shared" else worker_init_fn_win if is_windows else None
            )
        )

    val_loader = DataLoader(
        cached_datasets[0],
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers= False, # because each epoch runs a different DS #not is_windows,
        worker_init_fn=worker_init_fn if cache_type == "shared" else worker_init_fn_win if is_windows else None
    )

    return train_loaders, val_loader, cached_datasets


def load_pgn_games(pgn_file, max_games:int = 999999999) -> List[chess.pgn.Game]:
    """Load all games from the PGN file"""
    games = []
    with open(pgn_file, 'r', encoding='utf-8') as f:
        while len(games) < max_games:
            try:
                if len(games) % 10_000 == 0:
                    print(f"Loading game {len(games)}, remains {max_games - len(games)}")
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                games.append(game)
            except Exception as e:
                print(f"Error reading game: {e}")
                continue

    return games

def aggregate_pgn_games(directory, max_games):
    all_games = []
    for filename in os.listdir(directory):
        if filename.endswith('.pgn'):
            file_path = os.path.join(directory, filename)
            print(f"Loading games from {file_path}")
            games = load_pgn_games(file_path, max_games - len(all_games))
            print(f"Loaded {len(games)} games from {file_path}, remains {max_games - len(games)}")
            all_games.extend(games)
    random.shuffle(all_games)
    return all_games


def build_ios(file, pos, shape):
    buffered_shape = shape + 10_000
    inputs_mmap = np.memmap(f'{file}_{pos}_inputs.dat', dtype=np.float32, mode='w+',
                            shape=(buffered_shape, 19, 8, 8))
    outputs_mmap = np.memmap(f'{file}_{pos}_outputs.dat', dtype=np.float32, mode='w+',
                             shape=(buffered_shape, 4096))
    metadata = {
        'move_uci': [],
        'fen': [],
        'move_number': [],
        # Add other metadata fields as needed
    }
    return inputs_mmap, outputs_mmap, metadata

# Example usage
if __name__ == "__main__":
    # Sample master game
    mmap_filename = "data/all_train_data_with_puzzles_v3"
    max_games = 1_700_000
    first_chunk_size = int(1024 * 1024 * 3.18) #first chunk is used as validation
    next_chunk_size = int(1024 * 1024 * 3.18) #Overshoot a bit to avoid small batches
    current_split = 0

    fail_if_more_positions_available = True
    print("Loading PGN games...")
    games = []
    games.extend(aggregate_pgn_games("data/pgn/puzzles/", max_games-len(games)))
    games.extend(aggregate_pgn_games("data/pgn/", max_games-len(games)))
    random.shuffle(games)
    print("shuffled games")
    # Process the game
    generator = ChessTrainingDataGenerator()
    inputs_mmap, outputs_mmap, metadata = build_ios(mmap_filename, current_split, first_chunk_size)
    chunk_size = first_chunk_size
    idx = 0
    chunk_count = 0
    for i, game in enumerate(games):
        if i % 100_000 == 0:
            print(f"Processing game {i}/{len(games)} position {chunk_count} ({idx}) split {current_split}. Max per split: {chunk_size}")
        try:
            training_data = generator.process_pgn_string(game)
            for item in training_data:
                inputs_mmap[chunk_count] = item['input']
                outputs_mmap[chunk_count] = item['output']
                metadata['move_uci'].append(item['move_uci'])
                metadata['fen'].append(item['fen'])
                metadata['move_number'].append(item['move_number'])
                idx += 1
                chunk_count +=1
                if chunk_count % 100_000 == 0:
                    print(f"Flushing to mmap split {current_split} game {i} position {chunk_count}, global: {idx}")
                    inputs_mmap.flush()
                    outputs_mmap.flush()
                if idx >= current_split * next_chunk_size + first_chunk_size:
                    inputs_mmap.flush()
                    outputs_mmap.flush()
                    np.savez(f'{mmap_filename}_{current_split}_meta.npz', **metadata, length=chunk_count)
                    current_split += 1
                    chunk_count = 0
                    chunk_size = next_chunk_size
                    print(f"Generated split {current_split-1} of training in {mmap_filename}. Next position should be 0: {chunk_count} ==? {idx - first_chunk_size - (current_split - 1) * next_chunk_size}")
                    inputs_mmap, outputs_mmap, metadata = build_ios(mmap_filename, current_split, next_chunk_size)
        except Exception as e:
            print(f"Error processing game {i}: {e}")
            continue


    # Optionally, trim arrays if you overestimated num_examples
    inputs_mmap.flush()
    outputs_mmap.flush()

    # Save metadata and actual length
    np.savez(f'{mmap_filename}_{current_split}_meta.npz', **metadata, length=chunk_count)
    print(f"Generated {idx} training examples in {mmap_filename}")

