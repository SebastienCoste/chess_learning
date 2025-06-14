#!/usr/bin/env python3
"""
Chess Console Game Interface

Play chess against your trained CNN model with a rich console interface.

Usage:
    python play_chess.py --model checkpoints/best_model.pth --color white --difficulty medium
"""

import argparse
import sys
import os
import chess
import chess.engine
import torch
import numpy as np
from pathlib import Path
import time

class SimpleChessDisplay:
    """
    Simple console chess board display.
    This is a simplified version for demonstration.
    """
    
    def __init__(self, use_unicode=True):
        self.use_unicode = False #use_unicode
        
        # Simple ASCII pieces
        self.pieces = {
            'K': '♔' if self.use_unicode else ' K', 'Q': '♕' if self.use_unicode else ' Q',
            'R': '♖' if self.use_unicode else ' R', 'B': '♗' if self.use_unicode else ' B',
            'N': '♘' if self.use_unicode else ' N', 'P': '♙' if self.use_unicode else ' P',
            'k': '♚' if self.use_unicode else ' k', 'q': '♛' if self.use_unicode else ' q',
            'r': '♜' if self.use_unicode else ' r', 'b': '♝' if self.use_unicode else ' b',
            'n': '♞' if self.use_unicode else ' n', 'p': '♟' if self.use_unicode else ' p'
        }
    
    def display_board(self, board):
        """Display the chess board."""
        print("\n" + "="*40)
        print("   CHESS BOARD")
        print("="*40)
        print("    a   b   c   d   e   f   g   h")
        print("  ┌" + "─" * 32 + "┐")
        
        for row in range(7, -1, -1):
            row_str = f"{row + 1} │"
            for col in range(8):
                square = chess.square(col, row)
                piece = board.piece_at(square)
                if piece:
                    piece_char = self.pieces[piece.symbol()]
                else:
                    piece_char = '  '
                row_str += f" {piece_char} "
            row_str += f"│ {row + 1}"
            print(row_str)
        
        print("  └" + "─" * 32 + "┘")
        print("    a   b   c   d   e   f   g   h")
    
    def display_info(self, board, player_color):
        """Display game information."""
        turn = "White" if board.turn else "Black"
        player = "White" if player_color else "Black"
        
        print(f"\nTurn: {turn}")
        print(f"You are: {player}")
        
        if board.is_checkmate():
            winner = "Black" if board.turn else "White"
            print(f"\n🏆 CHECKMATE! {winner} wins!")
        elif board.is_stalemate():
            print("\n🤝 STALEMATE! Draw!")
        elif board.is_check():
            print("\n⚠️  CHECK!")
    
    def get_human_move(self, board):
        """Get move from human player."""
        while True:
            try:
                legal_moves = list(board.legal_moves)
                print(f"\nLegal moves: {len(legal_moves)}")
                sample_moves = [board.san(move) for move in legal_moves[:8]]
                print(f"Examples: {', '.join(sample_moves)}")
                
                user_input = input("\nEnter your move (e.g., 'e4', 'Nf3'): ").strip()
                
                if user_input.lower() == 'quit':
                    print("Thanks for playing!")
                    sys.exit(0)
                
                try:
                    move = board.parse_san(user_input)
                except ValueError:
                    try:
                        move = chess.Move.from_uci(user_input)
                    except ValueError:
                        print("Invalid move format. Try again.")
                        continue
                
                if move in board.legal_moves:
                    return move
                else:
                    print("Illegal move! Try again.")
                    
            except KeyboardInterrupt:
                print("\nThanks for playing!")
                sys.exit(0)


class SimpleChessEngine:
    """
    Simple chess engine interface.
    This is a placeholder for the full CNN implementation.
    """
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        print(f"Loading model from: {model_path}")
        print("Note: This is a simplified demo. The full implementation")
        print("would load your trained CNN model here.")
    
    def get_model_move(self, board):
        """Get best move from the model."""
        # In the full implementation, this would:
        # 1. Convert board to tensor
        # 2. Run inference with CNN
        # 3. Select best legal move
        
        # For demo: return a random legal move
        legal_moves = list(board.legal_moves)
        if legal_moves:
            move = np.random.choice(legal_moves)
            print(f"AI thinking... Selected move: {board.san(move)}")
            return move
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Play chess against CNN")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--color", choices=["white", "black"], default="white",
                       help="Your color")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], 
                       default="medium", help="AI difficulty")
    
    args = parser.parse_args()
    
    print("="*50)
    print("CHESS CNN CONSOLE GAME")
    print("="*50)
    
    if not args.model:
        print("No model specified. Using random move generator for demo.")
        print("To use a trained model, specify --model path/to/model.pth")
    
    # Initialize components
    board = chess.Board()
    display = SimpleChessDisplay()
    engine = SimpleChessEngine(args.model)
    
    player_color = chess.WHITE if args.color == "white" else chess.BLACK
    
    print(f"\nYou are playing {args.color}")
    print(f"AI difficulty: {args.difficulty}")
    print("Commands: Type moves like 'e4', 'Nf3', or 'quit' to exit")
    
    # Game loop
    while not board.is_game_over():
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
        display.display_board(board)
        display.display_info(board, player_color)
        
        if board.turn == player_color:
            # Human turn
            print("\n👤 Your turn!")
            move = display.get_human_move(board)
            board.push(move)
        else:
            # AI turn
            print("\n🤖 AI is thinking...")
            #time.sleep(1)  # Simulate thinking time
            move = engine.get_model_move(board)
            if move:
                board.push(move)
            else:
                print("AI cannot move!")
                break
            input("Press Enter to continue...")
    
    # Game over
    os.system('cls' if os.name == 'nt' else 'clear')
    display.display_board(board)
    display.display_info(board, player_color)
    
    print("\n🏁 Game Over!")
    result = board.result()
    if result == "1-0":
        print("White wins!")
    elif result == "0-1":
        print("Black wins!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    main()