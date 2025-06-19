import random

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from typing import List, Dict, Tuple

from cnn.chess_v3.components.chess_board_utils import board_to_tensor
from cnn.chess_v3.components.mmap_dataset import convert_pickle_to_memmap


class ChessTrainingDataGenerator:
    """
    Converts PGN chess games into training data for CNN.
    """

    def __init__(self, min_elo: int = 2000, skip_opening_moves: int = 6,
                 skip_endgame_moves: int = 10):
        self.min_elo = min_elo
        self.skip_opening_moves = skip_opening_moves
        self.skip_endgame_moves = skip_endgame_moves


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
            print("Could not parse ELO ratings")
            white_elo = 0
            black_elo = 0

        training_data = []
        board = game.board()
        moves = list(game.mainline_moves())

        for move_idx, move in enumerate(moves):
            # Skip opening and endgame moves
            if (move_idx < self.skip_opening_moves or
                    len(moves) - move_idx <= self.skip_endgame_moves):
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
                        'white': game.headers.get("White", "Unknown"),
                        'black': game.headers.get("Black", "Unknown"),
                        'white_elo': white_elo,
                        'black_elo': black_elo,
                        'event': game.headers.get("Event", "Unknown")
                    }
                })

            board.push(move)

        return training_data


class ChessDataset(Dataset):
    """PyTorch Dataset for chess training data."""

    def __init__(self, training_data: List[Dict]):
        self.data = training_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        #if sent to CUDA, DataLoader.pin_memory needs to be set to false, because it's used to optimize the migration from CPU to CUDA
        input_tensor = torch.FloatTensor(item['input'])#.to('cuda' if TRAINING_CONFIG["device"] == "cuda" else 'cpu')
        output_tensor = torch.FloatTensor(item['output'])#.to('cuda' if TRAINING_CONFIG["device"] == "cuda" else 'cpu')
        return input_tensor, output_tensor


def create_optimized_dataloaders(dataset, batch_size=512):
    """Create CUDA-optimized data loaders"""

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Optimized loader configuration
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Increase based on CPU cores
        pin_memory=True,  # Enable fast CPU->GPU transfer
       # persistent_workers=True,  # Keep workers alive between epochs
       # prefetch_factor=4  # Prefetch multiple batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=0,
        pin_memory=True,
       # persistent_workers=True
    )

    return train_loader, val_loader


def create_chess_data_loaders(
        training_data: List[Dict],
        batch_size,
        num_workers,
        train_split: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""

    # Split data
    split_idx = int(len(training_data) * train_split)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]

    print(f"Training set: {len(train_data)} examples")
    print(f"Validation set: {len(val_data)} examples")

    # Create datasets
    train_dataset = ChessDataset(train_data)
    val_dataset = ChessDataset(val_data)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True, # Keep workers alive between epochs
        prefetch_factor=4       # Prefetch multiple batches
        # pin_memory_device=TRAINING_CONFIG["device"], #RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True, # Keep workers alive between epochs
        # pin_memory_device=TRAINING_CONFIG["device"], #RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
    )

    return train_loader, val_loader

def load_pgn_games(pgn_file, max_games:int = 999999999) -> List[chess.pgn.Game]:
    """Load all games from the PGN file"""
    games = []
    with open(pgn_file, 'r', encoding='utf-8') as f:
        while len(games) < max_games:
            try:
                if len(games) % 100 == 0:
                    print(f"Loading game {len(games)}")
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                games.append(game)
            except Exception as e:
                print(f"Error reading game: {e}")
                continue

    return games

# Example usage
if __name__ == "__main__":
    # Sample master game
    pkl_filename = "data/mvl_train_data.pkl"
    max_games = 999999999
    print("Loading PGN games...")
    games = load_pgn_games("data/pgn/VachierLagrave.pgn", max_games)
    print(f"Loaded {len(games)} games")
    random.shuffle(games)
    print("shuffled games")
    # Process the game
    generator = ChessTrainingDataGenerator(min_elo=2000)

    all_training_data = []

    for i, game in enumerate(games):
        if i % 100 == 0:
            print(f"Processing game {i + 1}/{len(games)}")

        try:
            training_data = generator.process_pgn_string(game)
            all_training_data.extend(training_data)
        except Exception as e:
            print(f"Error processing game {i}: {e}")
            continue

    print(f"Generating {len(all_training_data)} training examples in {pkl_filename}")
    random.shuffle(all_training_data)
    print("shuffled all_training_data")
    # Save training data
    with open(pkl_filename, 'wb') as f:
        pickle.dump(all_training_data, f)
    print(f"Generated {len(all_training_data)} training examples in {pkl_filename}")
    convert_pickle_to_memmap(pkl_filename, pkl_filename.replace(".pkl", ""))
    print(f"moved {pkl_filename} to memmap files")

