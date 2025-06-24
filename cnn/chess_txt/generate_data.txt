import random
import os
import gc
import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple

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
        num_workers=8,  # Increase based on CPU cores
        pin_memory=True,  # Enable fast CPU->GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4  # Prefetch multiple batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
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
                if len(games) % 1000 == 0:
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

# Example usage
if __name__ == "__main__":
    # Sample master game
    mmap_filename = "data/all_train_data_with_puzzles_v2"
    max_games = 10_000_000
    estimated_positions_count = 16_667_704 + 10 # used to allocate space
    fail_if_more_positions_available = True
    print("Loading PGN games...")
    games = []
    games.extend(aggregate_pgn_games("data/pgn/puzzles/", max_games-len(games)))
    games.extend(aggregate_pgn_games("data/pgn/", max_games-len(games)))
    random.shuffle(games)
    print("shuffled games")
    # Process the game
    generator = ChessTrainingDataGenerator()

    inputs_mmap = np.memmap(f'{mmap_filename}_inputs.dat', dtype=np.float32, mode='w+',
                            shape=(estimated_positions_count, 19, 8, 8))
    outputs_mmap = np.memmap(f'{mmap_filename}_outputs.dat', dtype=np.float32, mode='w+', shape=(estimated_positions_count, 4096))

    metadata = {
        'move_uci': [],
        'fen': [],
        'move_number': [],
        # Add other metadata fields as needed
    }
    idx = 0
    for i, game in enumerate(games):
        if i % 100 == 0:
            print(f"Processing game {i + 1}/{len(games)}")
        try:
            training_data = generator.process_pgn_string(game)
            for item in training_data:
                if idx >= estimated_positions_count:
                    if fail_if_more_positions_available:
                        raise Exception(f"estimated {estimated_positions_count} but we actually have more examples")
                    else:
                        break
                inputs_mmap[idx] = item['input']
                outputs_mmap[idx] = item['output']
                metadata['move_uci'].append(item['move_uci'])
                metadata['fen'].append(item['fen'])
                metadata['move_number'].append(item['move_number'])
                idx += 1
                if idx % 100_000 == 0:
                    print(f"Flushing to mmap game {idx}")
                    inputs_mmap.flush()
                    outputs_mmap.flush()
        except Exception as e:
            print(f"Error processing game {i}: {e}")
            continue

    # Optionally, trim arrays if you overestimated num_examples
    inputs_mmap.flush()
    outputs_mmap.flush()

    # Save metadata and actual length
    np.savez(f'{mmap_filename}_meta.npz', **metadata, length=idx)
    print(f"Generated {idx} training examples in {mmap_filename}")
    del inputs_mmap
    del outputs_mmap
    gc.collect()

