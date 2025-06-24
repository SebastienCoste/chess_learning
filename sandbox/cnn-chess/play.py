# play.py
import chess
from chess import svg
from IPython.display import display


class ChessVisualizer:
    def __init__(self, model):
        self.board = chess.Board()
        self.model = model

    def display_board(self):
        print("\n" + "=" * 40)
        print(f"Move {len(self.board.move_stack)} - {'White' if self.board.turn else 'Black'} to play")
        print("=" * 40)
        print(self.board.unicode(borders=True))

    def get_model_move(self):
        input_tensor = convert_board_to_input(self.board)
        with torch.no_grad():
            policy, value = self.model(input_tensor)
        moves = decode_policy(policy)
        return select_best_move(moves)

    def play_loop(self):
        while not self.board.is_game_over():
            self.display_board()
            if self.board.turn == chess.WHITE:
                move = self.get_model_move()
            else:
                move = input("Enter your move (UCI format): ")
            self.board.push_uci(move)
