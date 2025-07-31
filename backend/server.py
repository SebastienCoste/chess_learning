from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import chess.pgn
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import uuid

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chess Move Encoder - Converts moves to/from model outputs
class ChessMoveEncoder:
    """
    Encodes chess moves for transformer model.
    Uses a comprehensive move representation covering all possible moves.
    """
    
    def __init__(self):
        self.move_to_index = {}
        self.index_to_move = {}
        self._build_move_mapping()
    
    def _build_move_mapping(self):
        """Build complete mapping of all possible chess moves to indices."""
        index = 0
        
        # Regular moves (from_square, to_square) - 64 * 64 = 4096
        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq != to_sq:
                    move_key = f"{from_sq}-{to_sq}"
                    self.move_to_index[move_key] = index
                    self.index_to_move[index] = move_key
                    index += 1
        
        # Promotion moves (from_square, to_square, promotion_piece) 
        # Only for pawns moving to 1st/8th rank
        promotion_pieces = ['q', 'r', 'b', 'n']  # queen, rook, bishop, knight
        
        for from_sq in range(64):
            for to_sq in range(64):
                # White pawn promotions (rank 7 to rank 8)
                if (48 <= from_sq <= 55) and (56 <= to_sq <= 63):
                    for piece in promotion_pieces:
                        move_key = f"{from_sq}-{to_sq}-{piece}"
                        self.move_to_index[move_key] = index
                        self.index_to_move[index] = move_key
                        index += 1
                
                # Black pawn promotions (rank 2 to rank 1)
                if (8 <= from_sq <= 15) and (0 <= to_sq <= 7):
                    for piece in promotion_pieces:
                        move_key = f"{from_sq}-{to_sq}-{piece}"
                        self.move_to_index[move_key] = index
                        self.index_to_move[index] = move_key
                        index += 1
        
        self.total_moves = index
        print(f"Total possible moves encoded: {self.total_moves}")
    
    def encode_move(self, move: chess.Move) -> int:
        """Convert chess.Move to model output index."""
        from_sq = move.from_square
        to_sq = move.to_square
        
        if move.promotion:
            piece_map = {chess.QUEEN: 'q', chess.ROOK: 'r', 
                        chess.BISHOP: 'b', chess.KNIGHT: 'n'}
            piece = piece_map[move.promotion]
            move_key = f"{from_sq}-{to_sq}-{piece}"
        else:
            move_key = f"{from_sq}-{to_sq}"
        
        return self.move_to_index.get(move_key, -1)
    
    def decode_move(self, index: int) -> Optional[chess.Move]:
        """Convert model output index to chess.Move."""
        if index not in self.index_to_move:
            return None
        
        move_key = self.index_to_move[index]
        parts = move_key.split('-')
        from_sq = int(parts[0])
        to_sq = int(parts[1])
        
        if len(parts) == 3:  # Promotion move
            piece_map = {'q': chess.QUEEN, 'r': chess.ROOK, 
                        'b': chess.BISHOP, 'n': chess.KNIGHT}
            promotion = piece_map[parts[2]]
            return chess.Move(from_sq, to_sq, promotion=promotion)
        else:
            return chess.Move(from_sq, to_sq)

# Chess Position Encoder - Converts board positions to model inputs
class ChessPositionEncoder:
    """
    Encodes chess positions as tensors for transformer input.
    Uses 18-channel representation: pieces + metadata.
    """
    
    def __init__(self):
        self.piece_to_channel = {
            chess.PAWN: 0, chess.ROOK: 1, chess.KNIGHT: 2,
            chess.BISHOP: 3, chess.QUEEN: 4, chess.KING: 5
        }
    
    def encode_position(self, board: chess.Board) -> torch.Tensor:
        """
        Convert chess board to 8x8x18 tensor.
        Channels 0-5: White pieces (pawn, rook, knight, bishop, queen, king)
        Channels 6-11: Black pieces (pawn, rook, knight, bishop, queen, king)
        Channels 12-15: Castling rights (white kingside, queenside, black kingside, queenside)
        Channel 16: En passant target square
        Channel 17: Turn to play (1 for white, 0 for black)
        """
        position = torch.zeros(8, 8, 18, dtype=torch.float32)
        
        # Encode pieces (channels 0-11)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = square // 8
                col = square % 8
                channel_offset = 0 if piece.color == chess.WHITE else 6
                channel = channel_offset + self.piece_to_channel[piece.piece_type]
                position[row, col, channel] = 1.0
        
        # Encode castling rights (channels 12-15)
        if board.has_kingside_castling_rights(chess.WHITE):
            position[:, :, 12] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            position[:, :, 13] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            position[:, :, 14] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            position[:, :, 15] = 1.0
        
        # Encode en passant (channel 16)
        if board.ep_square is not None:
            ep_row = board.ep_square // 8
            ep_col = board.ep_square % 8
            position[ep_row, ep_col, 16] = 1.0
        
        # Encode turn to play (channel 17)
        if board.turn == chess.WHITE:
            position[:, :, 17] = 1.0
        
        return position

class ChessTransformerConfig:
    """Configuration class for the chess transformer model."""
    
    def __init__(self,
                 input_channels: int = 18,
                 board_size: int = 8,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 total_moves: int = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.input_channels = input_channels
        self.board_size = board_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.total_moves = total_moves
        self.device = device

class ChessTransformer(nn.Module):
    """
    Transformer model for chess move prediction.
    Optimized for CUDA with efficient attention mechanisms.
    """
    
    def __init__(self, config: ChessTransformerConfig):
        super().__init__()
        self.config = config
        
        # Input projection: flatten board and project to d_model
        self.input_projection = nn.Linear(
            config.input_channels * config.board_size * config.board_size,
            config.d_model
        )
        
        # Positional encoding for board squares
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.board_size * config.board_size, config.d_model)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        # Output heads
        self.policy_head = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.total_moves)
        )
        
        # Optional value head for position evaluation
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, 1),
            nn.Tanh()
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for better training stability."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass.
        x: (batch_size, 8, 8, 18) - board positions
        Returns: policy logits (batch_size, total_moves), value (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # Flatten and project input: (batch_size, 8*8*18) -> (batch_size, d_model)
        x = x.view(batch_size, -1)
        x = self.input_projection(x)
        
        # Reshape for transformer: (batch_size, seq_len, d_model)
        # Treat each square as a sequence element
        x = x.view(batch_size, 1, self.config.d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :1, :]
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Output heads
        policy = self.policy_head(x)  # (batch_size, total_moves)
        value = self.value_head(x)    # (batch_size, 1)
        
        return policy, value

# Training utilities
class ChessTrainer:
    """Training pipeline for the chess transformer."""
    
    def __init__(self, model, config, move_encoder, position_encoder):
        self.model = model
        self.config = config
        self.move_encoder = move_encoder
        self.position_encoder = position_encoder
        self.device = config.device
        
        # Optimizers
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10000,
            eta_min=1e-6
        )
        
        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # Move model to device
        self.model.to(self.device)
    
    def compute_loss(self, positions, target_moves, target_values, legal_moves_mask):
        """
        Compute training loss with penalties for illegal moves.
        
        Args:
            positions: (batch_size, 8, 8, 18) - board positions
            target_moves: (batch_size,) - target move indices
            target_values: (batch_size,) - position evaluation targets
            legal_moves_mask: (batch_size, total_moves) - 1 for legal moves, 0 for illegal
        """
        positions = positions.to(self.device)
        target_moves = target_moves.to(self.device)
        target_values = target_values.to(self.device)
        legal_moves_mask = legal_moves_mask.to(self.device)
        
        # Forward pass
        policy_logits, value_pred = self.model(positions)
        
        # Policy loss with illegal move penalty
        policy_loss = self.policy_loss_fn(policy_logits, target_moves)
        
        # Add penalty for illegal moves (negative log likelihood on illegal moves)
        illegal_penalty = -torch.log(torch.sigmoid(-policy_logits) + 1e-8) * (1 - legal_moves_mask)
        illegal_penalty = illegal_penalty.sum(dim=1).mean()
        
        # Value loss
        value_loss = self.value_loss_fn(value_pred.squeeze(), target_values)
        
        # Combined loss
        total_loss = policy_loss + 0.1 * illegal_penalty + 0.5 * value_loss
        
        return total_loss, policy_loss, value_loss, illegal_penalty
    
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        positions, target_moves, target_values, legal_moves_mask = batch
        
        loss, policy_loss, value_loss, illegal_penalty = self.compute_loss(
            positions, target_moves, target_values, legal_moves_mask
        )
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'total_loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'illegal_penalty': illegal_penalty.item(),
            'lr': self.scheduler.get_last_lr()[0]
        }

# Global instances
move_encoder = ChessMoveEncoder()
position_encoder = ChessPositionEncoder()

# Model configuration
config = ChessTransformerConfig(
    input_channels=18,
    board_size=8,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    total_moves=move_encoder.total_moves,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Initialize model and trainer
model = ChessTransformer(config)
trainer = ChessTrainer(model, config, move_encoder, position_encoder)

print(f"Chess Transformer initialized!")
print(f"Device: {config.device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Total possible moves: {move_encoder.total_moves}")

# Pydantic models for API
class PositionRequest(BaseModel):
    fen: str

class TrainingRequest(BaseModel):
    pgn_games: List[str]
    epochs: int = 10
    batch_size: int = 32

class PredictionResponse(BaseModel):
    best_moves: List[Dict]
    position_value: float
    legal_moves_count: int

@app.get("/")
async def root():
    return {
        "message": "Chess Transformer API",
        "model_info": {
            "device": config.device,
            "parameters": sum(p.numel() for p in model.parameters()),
            "total_moves": move_encoder.total_moves
        }
    }

@app.post("/api/predict")
async def predict_move(request: PositionRequest):
    """Predict best moves for a given chess position."""
    try:
        # Parse FEN
        board = chess.Board(request.fen)
        
        # Encode position
        position_tensor = position_encoder.encode_position(board).unsqueeze(0)
        
        # Get model prediction
        model.eval()
        with torch.no_grad():
            policy_logits, value = model(position_tensor.to(config.device))
            policy_probs = F.softmax(policy_logits, dim=1)[0]
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        legal_move_probs = []
        
        for move in legal_moves:
            move_idx = move_encoder.encode_move(move)
            if move_idx != -1:
                prob = policy_probs[move_idx].item()
                legal_move_probs.append({
                    'move': move.uci(),
                    'probability': prob,
                    'san': board.san(move)
                })
        
        # Sort by probability
        legal_move_probs.sort(key=lambda x: x['probability'], reverse=True)
        
        return PredictionResponse(
            best_moves=legal_move_probs[:10],  # Top 10 moves
            position_value=value.item(),
            legal_moves_count=len(legal_moves)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/train")
async def train_model(request: TrainingRequest):
    """Train the model on provided PGN games."""
    try:
        # This is a placeholder for training implementation
        # In a real implementation, you'd parse PGN games and create training data
        
        return {
            "message": f"Training started with {len(request.pgn_games)} games",
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "status": "Training pipeline ready - implement data processing"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/model/info")
async def model_info():
    """Get model architecture and configuration info."""
    return {
        "architecture": "Chess Transformer",
        "config": {
            "input_channels": config.input_channels,
            "board_size": config.board_size,
            "d_model": config.d_model,
            "num_heads": config.nhead,
            "num_layers": config.num_encoder_layers,
            "total_moves": config.total_moves,
            "device": config.device
        },
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)