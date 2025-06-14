#!/usr/bin/env python3
"""
Chess PGN to LLM Training Data Preparation Script
Converts PGN chess games into a format suitable for training Mistral LLM.
"""

import chess
import chess.pgn
import json
import random
from typing import List, Dict, Tuple
import re
from pathlib import Path
from transformers import AutoTokenizer

class ChessDataProcessor:
    def __init__(self, pgn_file: str = "chess_games_10k.pgn"):
        self.pgn_file = pgn_file
        self.training_data = []
        self.validation_data = []
        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    def load_pgn_games(self) -> List[chess.pgn.Game]:
        """Load all games from the PGN file"""
        games = []

        with open(self.pgn_file, 'r', encoding='utf-8') as f:
            while True:
                try:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    games.append(game)
                except Exception as e:
                    print(f"Error reading game: {e}")
                    continue

        return games

    def game_to_training_examples(self, game: chess.pgn.Game) -> List[Dict]:
        """Convert a single chess game into multiple training examples"""
        examples = []
        board = game.board()

        # Extract game metadata
        headers = dict(game.headers)
        white_player = headers.get('White', 'Unknown')
        black_player = headers.get('Black', 'Unknown')
        result = headers.get('Result', '*')
        event = headers.get('Event', 'Unknown')

        # Get the game moves
        moves = []
        node = game
        while node.variations:
            next_node = node.variation(0)
            if next_node.move:
                moves.append(next_node.move)
            node = next_node

        if len(moves) < 10:  # Skip very short games
            return examples

        # Create training examples at different game positions
        for i in range(5, min(len(moves), 50), 5):  # Every 5 moves up to move 50
            # Play moves up to position i
            temp_board = chess.Board()
            game_history = []

            for move_idx in range(i):
                if move_idx < len(moves):
                    move = moves[move_idx]
                    if move in temp_board.legal_moves:
                        san_move = temp_board.san(move)
                        game_history.append(san_move)
                        temp_board.push(move)
                    else:
                        break  # Invalid move, skip this game

            if len(game_history) < 5:
                continue

            # Create the next move prediction example
            if i < len(moves) and moves[i] in temp_board.legal_moves:
                next_move = temp_board.san(moves[i])

                # Format the conversation for training
                conversation = self.create_move_prediction_conversation(
                    game_history, next_move, white_player, black_player, temp_board
                )
                examples.append(conversation)

                # Also create position analysis examples
                analysis_conversation = self.create_position_analysis_conversation(
                    game_history, temp_board, next_move
                )
                examples.append(analysis_conversation)

        return examples

    def create_move_prediction_conversation(self, game_history: List[str],
                                            next_move: str, white_player: str,
                                            black_player: str, board: chess.Board) -> Dict:
        """Create a conversation for move prediction"""

        # Create game state description
        move_number = (len(game_history) + 1) // 2 + 1
        is_white_turn = board.turn == chess.WHITE

        pgn_sequence = self.format_moves_as_pgn(game_history)

        user_prompt = f"""I'm analyzing a chess game between {white_player} (White) and {black_player} (Black).

Current position after moves: {pgn_sequence}

It's {"White" if is_white_turn else "Black"} to move. What is the next move?"""

        assistant_response = f"The next move is: {next_move}"
        # Prepare messages list
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]

        # Format using TinyLlama chat template
        conversation = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False  # No extra prompt needed for training targets
        )

        return {"text": conversation, "messages": messages}

    def create_position_analysis_conversation(self, game_history: List[str],
                                              board: chess.Board, actual_move: str) -> Dict:
        """Create a conversation for position analysis"""

        pgn_sequence = self.format_moves_as_pgn(game_history)

        # Analyze the position
        legal_moves = list(board.legal_moves)
        top_moves = [board.san(move) for move in legal_moves[:5]]  # Top 5 legal moves

        is_check = board.is_check()
        is_white_turn = board.turn == chess.WHITE

        user_prompt = f"""Analyze this chess position:

Moves played: {pgn_sequence}

Current turn: {"White" if is_white_turn else "Black"}
{"The king is in check!" if is_check else ""}

What are some good move options and what was actually played?"""

        assistant_response = f"""Looking at this position:

Legal moves available: {', '.join(top_moves[:5])}
{"This is a check position, so the king must be addressed." if is_check else ""}

The move actually played was: {actual_move}

This move {"addresses the check" if is_check else "continues the game's development"}."""
        # Prepare messages list
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]

        # Format using TinyLlama chat template
        conversation = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False  # No extra prompt needed for training targets
        )

        return {"text": conversation, "messages": messages}

    def format_moves_as_pgn(self, moves: List[str]) -> str:
        """Format a list of moves as PGN notation"""
        pgn = ""
        for i, move in enumerate(moves):
            if i % 2 == 0:  # White move
                pgn += f"{i // 2 + 1}. {move} "
            else:  # Black move
                pgn += f"{move} "
        return pgn.strip()

    def process_all_games(self, train_split: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """Process all games and split into training and validation"""
        print("Loading PGN games...")
        games = self.load_pgn_games()
        print(f"Loaded {len(games)} games")

        all_examples = []

        for i, game in enumerate(games):
            if i % 100 == 0:
                print(f"Processing game {i + 1}/{len(games)}")

            try:
                examples = self.game_to_training_examples(game)
                all_examples.extend(examples)
            except Exception as e:
                print(f"Error processing game {i}: {e}")
                continue

        print(f"Generated {len(all_examples)} training examples")

        # Shuffle and split
        random.shuffle(all_examples)
        split_idx = int(len(all_examples) * train_split)

        train_data = all_examples[:split_idx]
        val_data = all_examples[split_idx:]

        return train_data, val_data

    def save_training_data(self, train_data: List[Dict], val_data: List[Dict],
                           train_file: str = "chess_train.jsonl",
                           val_file: str = "chess_val.jsonl"):
        """Save training data in JSONL format for Mistral fine-tuning"""

        # Save training data
        with open(train_file, 'w', encoding='utf-8') as f:
            for example in train_data:
                f.write(json.dumps(example) + '\n')

        # Save validation data
        with open(val_file, 'w', encoding='utf-8') as f:
            for example in val_data:
                f.write(json.dumps(example) + '\n')

        print(f"Saved {len(train_data)} training examples to {train_file}")
        print(f"Saved {len(val_data)} validation examples to {val_file}")

    def create_sample_data(self, num_samples: int = 1000):
        """Create sample training data if PGN file doesn't exist"""
        print(f"Creating {num_samples} sample training examples...")

        sample_games = [
            "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O",
            "1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 dxc4 5. a4 Bf5 6. e3 e6 7. Bxc4 Bb4 8. O-O O-O",
            "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e6 7. f3 b5 8. Qd2 Bb7",
            "1. Nf3 Nf6 2. g3 g6 3. Bg2 Bg7 4. O-O O-O 5. d3 d6 6. e4 e5 7. Nc3 Nc6 8. h3 h6"
        ]

        examples = []

        for i in range(num_samples):
            base_game = sample_games[i % len(sample_games)]
            moves = base_game.split()[1:]  # Remove move numbers

            # Create a conversation about this position
            move_count = random.randint(3, len(moves) - 1)
            game_moves = moves[:move_count]
            next_move = moves[move_count] if move_count < len(moves) else "..."

            pgn_text = ""
            for j, move in enumerate(game_moves):
                if j % 2 == 0:
                    pgn_text += f"{j // 2 + 1}. {move} "
                else:
                    pgn_text += f"{move} "

            user_prompt = f"""Analyze this chess position:

Moves: {pgn_text.strip()}

What should be the next move?"""

            assistant_response = f"The next move should be: {next_move}"

            example = {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ]
            }
            examples.append(example)

        return examples

    def prepare_data(self):
        """Main method to prepare all training data"""
        # Check if PGN file exists
        if not Path(self.pgn_file).exists():
            raise(f"PGN file {self.pgn_file} not found. NOT Creating sample data...")
            #train_data = self.create_sample_data(8000)
            #val_data = self.create_sample_data(2000)
        else:
            # Process real PGN data
            train_data, val_data = self.process_all_games()

        # Save the processed data
        self.save_training_data(train_data, val_data)

        print("Data preparation complete!")
        print(f"Training examples: {len(train_data)}")
        print(f"Validation examples: {len(val_data)}")

        return train_data, val_data


if __name__ == "__main__":
    processor = ChessDataProcessor()
    train_data, val_data = processor.prepare_data()