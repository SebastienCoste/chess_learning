import torch
import chess
import numpy as np
from typing import Optional

from cnn.chess_v3.components.chess_board_utils import board_to_tensor
from cnn.chess_v3.components.chess_cnn import EnhancedChessCNN
from cnn.chess_v3.components.config import TRAINING_CONFIG


class SimpleChessEngine:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path:
            print(f"Loading model from: {model_path}")
            try:
                # Load the model architecture (you'll need to import your actual model class)

                # Initialize model with same config as training
                self.model = EnhancedChessCNN(
                    **TRAINING_CONFIG["config"]
                )

                # Load trained weights
                checkpoint = torch.load(model_path, map_location=self.device)

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    self.model.load_state_dict(checkpoint)

                self.model.to(self.device)
                self.model.eval()
                print(f"Model successfully loaded on {self.device}")

            except Exception as e:
                print(f"Error loading model: {e}")
                print("Falling back to random move generator")
                self.model = None
        else:
            print("No model file provided. Using random move generator for demo.")
            self.model = None

    def idx_to_move(self, move_idx: int, board: chess.Board) -> Optional[chess.Move]:
        """
        Convert model output index to chess move.
        Move index = from_square * 64 + to_square
        """
        from_square = move_idx // 64
        to_square = move_idx % 64

        # Create the move
        move = chess.Move(from_square, to_square)

        # Handle promotion moves (assume queen promotion for simplicity)
        if board.piece_at(from_square) and board.piece_at(from_square).piece_type == chess.PAWN:
            # Check if it's a promotion move
            if (board.turn == chess.WHITE and to_square >= 56) or (board.turn == chess.BLACK and to_square <= 7):
                move = chess.Move(from_square, to_square, promotion=chess.QUEEN)

        return move if move in board.legal_moves else None

    def get_top_k_moves(self, board: chess.Board, k: int = 5) -> list:
        """
        Get top-k move predictions from the model.
        """
        if self.model is None:
            return []

        try:
            # Get model predictions
            x = board_to_tensor(board)
            with torch.no_grad():
                logits = self.model(x)
                probabilities = torch.softmax(logits, dim=1)

            # Get top-k predictions
            top_k_values, top_k_indices = torch.topk(probabilities, k, dim=1)

            moves_with_probs = []
            for i in range(k):
                move_idx = top_k_indices[0][i].item()
                prob = top_k_values[0][i].item()
                move = self.idx_to_move(move_idx, board)

                if move and move in board.legal_moves:
                    moves_with_probs.append((move, prob))

            return moves_with_probs

        except Exception as e:
            print(f"Error in get_top_k_moves: {e}")
            return []

    def get_model_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Get the best move from the model, with fallback strategies.
        """
        if self.model is None:
            return self._get_random_move(board)

        try:
            # Get model predictions
            x = board_to_tensor(board)
            # Convert to PyTorch tensor
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            x = x.to(self.device)
            # Add batch dimension if needed
            if len(x.shape) == 2:  # If it's just [height, width]
                x = x.unsqueeze(0)  # Make it [1, height, width]
            elif len(x.shape) == 3:  # If it's [channels, height, width]
                x = x.unsqueeze(0)  # Make it [1, channels, height, width]
            with torch.no_grad():
                logits = self.model(x)
                probabilities = torch.softmax(logits, dim=1)

            # Try top moves until we find a legal one
            top_k_values, top_k_indices = torch.topk(probabilities, 10, dim=1)

            for i in range(10):
                move_idx = top_k_indices[0][i].item()
                prob = top_k_values[0][i].item()
                move = self.idx_to_move(move_idx, board)

                if move and move in board.legal_moves:
                    print(f"AI (model) selects: {board.san(move)} (confidence: {prob:.3f})")
                    return move

            # If no top-10 moves are legal, fall back to random legal move
            print("AI model's top predictions were illegal, selecting random legal move.")
            return self._get_random_move(board)

        except Exception as e:
            print(f"Error in model inference: {e}")
            print("Falling back to random move.")
            return self._get_random_move(board)

    def _get_random_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Fallback random move generator."""
        legal_moves = list(board.legal_moves)
        if legal_moves:
            move = np.random.choice(legal_moves)
            print(f"AI (random) selects: {board.san(move)}")
            return move
        return None

    def get_move_analysis(self, board: chess.Board) -> dict:
        """
        Provide detailed analysis of the current position.
        """
        if self.model is None:
            return {"error": "No model loaded"}

        try:
            top_moves = self.get_top_k_moves(board, k=5)

            analysis = {
                "position_fen": board.fen(),
                "to_move": "White" if board.turn else "Black",
                "legal_moves_count": len(list(board.legal_moves)),
                "top_moves": []
            }

            for move, prob in top_moves:
                analysis["top_moves"].append({
                    "move": board.san(move),
                    "uci": move.uci(),
                    "probability": f"{prob:.3f}",
                    "percentage": f"{prob * 100:.1f}%"
                })

            return analysis

        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
