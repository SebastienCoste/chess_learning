import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import io
from typing import List, Dict, Tuple


class ChessTrainingDataGenerator:
    """
    Converts PGN chess games into training data for CNN.
    """

    def __init__(self, min_elo: int = 2000, skip_opening_moves: int = 6,
                 skip_endgame_moves: int = 10):
        self.min_elo = min_elo
        self.skip_opening_moves = skip_opening_moves
        self.skip_endgame_moves = skip_endgame_moves

    def board_to_tensor(self, board: chess.Board) -> np.ndarray:
        """
        Convert chess board to 19-channel tensor representation.

        Channels 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
        Channels 6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
        Channels 12-15: Castling rights
        Channel 16: En passant target
        Channel 17: Move count (normalized)
        Channel 18: Turn to move (1=white, 0=black)
        """
        tensor = np.zeros((19, 8, 8), dtype=np.float32)

        # Piece mapping for channels 0-11
        piece_map = {
            (chess.PAWN, chess.WHITE): 0,
            (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK, chess.WHITE): 3,
            (chess.QUEEN, chess.WHITE): 4,
            (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6,
            (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK, chess.BLACK): 9,
            (chess.QUEEN, chess.BLACK): 10,
            (chess.KING, chess.BLACK): 11,
        }

        # Fill piece channels
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = piece_map[(piece.piece_type, piece.color)]
                row = 7 - (square // 8)  # Convert to array indexing
                col = square % 8
                tensor[channel, row, col] = 1.0

        # Game state channels (12-18)
        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[12, 7, 0] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[13, 7, 7] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[14, 0, 0] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[15, 0, 7] = 1.0

        # En passant target
        if board.ep_square is not None:
            row = 7 - (board.ep_square // 8)
            col = board.ep_square % 8
            tensor[16, row, col] = 1.0

        # Move count (normalized by 100)
        tensor[17, :, :] = min(board.fullmove_number / 100.0, 1.0)

        # Turn to move
        if board.turn == chess.WHITE:
            tensor[18, :, :] = 1.0

        return tensor

    def move_to_index(self, move: chess.Move) -> int:
        """Convert move to index in 4096-dimensional output vector."""
        return move.from_square * 64 + move.to_square

    def process_pgn_string(self, pgn_string: str) -> List[Dict]:
        """Process PGN string and extract training positions."""
        pgn_io = io.StringIO(pgn_string)
        game = chess.pgn.read_game(pgn_io)

        if not game:
            return []

        # Check ELO requirements
        try:
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
            if white_elo < self.min_elo or black_elo < self.min_elo:
                print(f"Skipping game: ELO too low (W:{white_elo}, B:{black_elo})")
                return []
        except (ValueError, TypeError):
            print("Could not parse ELO ratings")
            return []

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
            input_tensor = self.board_to_tensor(board)

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
        input_tensor = torch.FloatTensor(item['input'])
        output_tensor = torch.FloatTensor(item['output'])
        return input_tensor, output_tensor


def create_chess_data_loaders(
        training_data: List[Dict],
        train_split: float = 0.8,
        batch_size: int = 64,
        num_workers: int = 4
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
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    # Sample master game
    sample_pgn = """[Event "World Championship"]
[Site "London"]
[Date "2018.11.24"]
[Round "12"]
[White "Carlsen, Magnus"]
[Black "Caruana, Fabiano"]
[Result "1/2-1/2"]
[WhiteElo "2835"]
[BlackElo "2832"]

1. e4 c5 2. Nf3 Nc6 3. Bb5 g6 4. Bxc6 dxc6 5. d3 Bg7 6. h3 Nf6 7. Nc3 O-O 8. Be3 b6 9. Qd2 e5 10. Bh6 Qe7 11. Bxg7 Qxg7 12. O-O-O Rd8 13. Kb1 Rd7 14. Rhe1 Re8 15. h4 h6 16. h5 g5 17. Nh2 Nh7 18. f4 exf4 19. Qxf4 Qxf4 20. Rxf4 Re2 21. Rf2 Re1+ 22. Rxe1 1/2-1/2"""

    # Process the game
    generator = ChessTrainingDataGenerator(min_elo=2000)
    training_data = generator.process_pgn_string(sample_pgn)

    # Save training data
    with open('chess_training_data.pkl', 'wb') as f:
        pickle.dump(training_data, f)

    print(f"Generated {len(training_data)} training examples")
    print(f"Saved to chess_training_data.pkl")
