import chess
import numpy as np
# Piece mapping for channels 0-11
PIECES_MAP = {
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


def board_to_tensor(board: chess.Board, flipped: bool = False) -> np.ndarray:
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

    # Fill piece channels
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = PIECES_MAP[(piece.piece_type, piece.color if not flipped else not piece.color)]
            row = 7 - (square // 8)  # Convert to array indexing
            col = square % 8
            tensor[channel, row, col] = 1.0

    # Game state channels (12-18)
    if board.has_kingside_castling_rights(chess.WHITE if not flipped else chess.BLACK):
        tensor[12, 7, 0] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE if not flipped else chess.BLACK):
        tensor[13, 7, 7] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK if not flipped else chess.BLACK):
        tensor[14, 0, 0] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK if not flipped else chess.BLACK):
        tensor[15, 0, 7] = 1.0

    # En passant target
    if board.ep_square is not None:
        row = 7 - (board.ep_square // 8)
        col = board.ep_square % 8
        tensor[16, row, col] = 1.0

    # Move count (normalized by 100)
    tensor[17, :, :] = min(board.fullmove_number / 100.0, 1.0)

    # Turn to move
    if board.turn == chess.WHITE if not flipped else chess.BLACK:
        tensor[18, :, :] = 1.0

    return tensor