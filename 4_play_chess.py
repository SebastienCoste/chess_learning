import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import random
from typing import List, Optional
from pathlib import Path
import re

class ChessLLMPlayer:
    def __init__(self,
                 model_path: str = "./improved-chess-llm",
                 model_name: str  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.board = chess.Board()
        self.game_history = []
        self.load_model()

    def load_model(self):
        print("Loading chess LLM...")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}.")

        try:
            config_path = Path(self.model_path) / "training_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                base_model_name = config.get("base_model", self.model_name)
            else:
                base_model_name = self.model_name

            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="right")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()
            print("Chess LLM loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def format_game_state(self) -> str:
        if not self.game_history:
            return "Starting position - no moves played yet."
        pgn_moves = ""
        for i, move in enumerate(self.game_history):
            if i % 2 == 0:
                pgn_moves += f"{i // 2 + 1}. {move} "
            else:
                pgn_moves += f"{move} "
        return pgn_moves.strip()

    def build_chat_messages(self, failed_candidates: List[str] = []) -> List[dict]:
        # Optionally, you can add a system prompt to steer the model
        messages = [
            {"role": "system", "content": "You are a chess expert. Respond only with the best next move in standard algebraic notation, no commentary."}
        ]
        # Reconstruct the conversation as alternating user/assistant turns
        moves = self.game_history
        for idx, move in enumerate(moves):
            role = "user" if idx % 2 == 0 else "assistant"
            messages.append({"role": role, "content": move})
        # Add the current user prompt
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        san_moves = [self.board.san(move) for move in list(self.board.legal_moves)]
        #print(f"AI will pick from these moves: {", ".join(san_moves)}, not {", ".join(failed_candidates)}")
        user_prompt = (
            f"Moves played: {self.format_game_state()}\n"
            f"It's {turn} to move. What is thd4e next move? Your answer is only 1 word and has to be one of the following words: [ {", ".join(san_moves)} ]. Do NOT answer any of these: [ {" , ".join(failed_candidates)} ]"
        )
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def generate_move_candidates(self, num_candidates: int = 10) -> List[str]:
        if self.model is None:
            return self.generate_fallback_moves(num_candidates)


        # Use the chat template to format the prompt for TinyLlama

        candidates = []
        failed_candidates = ["toto"]
        while not candidates and len(failed_candidates) < 25:
            messages = self.build_chat_messages(failed_candidates)
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True  # Tells the model to generate the assistant's reply
                )
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                    padding=True
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=16,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        top_k=40,
                        top_p=0.95,
                        repetition_penalty=1.1
                    )

                # Only decode the newly generated tokens (assistant's reply)
                generated = outputs[0][inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                response = response.strip().split("\n")[0].replace("'", "").replace("The next move is: ", "")
                if ":" in response:
                    response = response.split(":")[1].replace(" ", "")
                move = self.extract_move_from_response(response)
                if not move and response:
                    failed_candidates.append(response)
                if move:
                    candidates.append(move)
            except Exception as e:
                print(f"Error generating candidate: {e}")
                continue

        if not candidates:
            print(f"Nothing good, generate_fallback_moves")
            return self.generate_fallback_moves(num_candidates)
        return candidates

    def extract_move_from_response(self, response: str) -> Optional[str]:
        # Try to extract a valid SAN move from the model's response
        # Remove any commentary or extra text
        # Patterns for SAN moves
        patterns = [
            r'\b([NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[NBRQ])?[+#]?)\b',  # Standard moves
            r'\b(O-O-O|O-O)\b',  # Castling
            r'\b([a-h][1-8])\b'  # Simple pawn moves
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                move_str = match if isinstance(match, str) else match[0]
                try:
                    move = self.board.parse_san(move_str)
                    if move in self.board.legal_moves:
                        return move_str
                except Exception as e:
                    #print(f"Error extract_move_from_response: {e}")
                    continue
        return None

    def generate_fallback_moves(self, num_candidates: int = 10) -> List[str]:
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return []
        san_moves = [self.board.san(move) for move in legal_moves]
        random.shuffle(san_moves)
        return san_moves[:num_candidates]

    def select_best_move(self, candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            try:
                move = self.board.parse_san(candidate)
                if move in self.board.legal_moves:
                    return candidate
            except Exception:
                continue
        legal_moves = list(self.board.legal_moves)
        if legal_moves:
            random_move = random.choice(legal_moves)
            return self.board.san(random_move)
        return None

    def make_ai_move(self) -> Optional[str]:
        print("\nAI is thinking...")
        candidates = self.generate_move_candidates(10)
        print(f"Generated candidates: {candidates}")
        move = self.select_best_move(candidates)
        if move:
            parsed_move = self.board.parse_san(move)
            self.board.push(parsed_move)
            self.game_history.append(move)
            print(f"AI plays: {move}")
            return move
        else:
            print("AI couldn't find a valid move!")
            return None

    def make_human_move(self, move_str: str) -> bool:
        try:
            move = self.board.parse_san(move_str)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.game_history.append(move_str)
                return True
            else:
                print("Invalid move! Please try again.")
                return False
        except Exception:
            print("Could not parse move! Please use standard algebraic notation (e.g., e4, Nf3, O-O)")
            return False

    def display_board(self):
        letters = "   a   b   c   d   e   f   g   h"
        print("\n" + "=" * 50)
        print("Current Position:")
        print("=" * 50)
        board_str = str(self.board)
        lines = board_str.split('\n')
        print(letters)
        for i, line in enumerate(lines):
            rank = 8 - i
            print(f"{rank}  {' '.join(line)}  {rank}")
        print(letters)
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn else "White"
            print(f"\nCheckmate! {winner} wins!")
        elif self.board.is_stalemate():
            print("\nStalemate! The game is a draw.")
        elif self.board.is_check():
            turn = "White" if self.board.turn else "Black"
            print(f"\n{turn} is in check!")
        if self.game_history:
            print(f"\nMoves played: {self.format_game_state()}")

    def play_game(self):
        print("Welcome to Chess vs AI!")
        print("Enter moves in standard algebraic notation (e.g., e4, Nf3, O-O)")
        print("Type 'quit' to exit, 'help' for available commands\n")
        self.display_board()
        opening = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Nc3", "Be7"]
        move_cnt = 0
        while not self.board.is_game_over():
            if move_cnt < len(opening):
                self.make_human_move(opening[move_cnt])
                self.display_board()
                move_cnt += 1
            else:
                if self.board.turn == chess.WHITE:
                    print(f"\nYour turn (White). Legal moves: {len(list(self.board.legal_moves))}")
                    user_input = input("Enter your move: ").strip()
                    if user_input.lower() == 'quit':
                        print("Thanks for playing!")
                        break
                    elif user_input.lower() == 'help':
                        self.show_help()
                        continue
                    elif user_input.lower() == 'moves':
                        legal_moves = [self.board.san(move) for move in self.board.legal_moves]
                        print(f"Legal moves: {', '.join(legal_moves)}.")
                        continue
                    if self.make_human_move(user_input):
                        self.display_board()
                else:
                    move = self.make_ai_move()
                    if move:
                        self.display_board()
                    else:
                        print("AI error - ending game")
                        break
        self.show_game_result()

    def show_help(self):
        print("\nCommands:")
        print("- Enter moves in algebraic notation: e4, Nf3, Bxf7+, O-O, etc.")
        print("- 'moves' - Show legal moves")
        print("- 'quit' - Exit the game")
        print("- 'help' - Show this help")

    def show_game_result(self):
        print("\n" + "=" * 50)
        print("GAME OVER")
        print("=" * 50)
        if self.board.is_checkmate():
            winner = "White" if not self.board.turn else "Black"
            print(f"Checkmate! {winner} wins!")
        elif self.board.is_stalemate():
            print("Stalemate! The game is a draw.")
        elif self.board.is_insufficient_material():
            print("Draw by insufficient material.")
        elif self.board.is_fivefold_repetition():
            print("Draw by fivefold repetition.")
        elif self.board.is_seventyfive_moves():
            print("Draw by 75-move rule.")
        print(f"\nFinal position: {self.board.fen()}")
        print(f"Game moves: {self.format_game_state()}")

def main():
    try:
        player = ChessLLMPlayer(model_path="./improved-chess-llm-v6/iter-16/checkpoint-60026") #./improved-chess-llm-v6/iter-16\checkpoint-60026 is the best declared CP
        player.play_game()
    except KeyboardInterrupt:
        print("\nGame interrupted. Thanks for playing!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
