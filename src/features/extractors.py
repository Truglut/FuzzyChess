import chess
from typing import Iterable

# CENTRAL_MASK = chess.BB_D4 | chess.BB_E4 | chess.BB_D5 | chess.BB_E5
CENTRAL_MASK = chess.BB_CENTER

EXTENDED_CENTRAL_MASK = (
    chess.BB_RANK_3 | chess.BB_RANK_4 | chess.BB_RANK_5 | chess.BB_RANK_6
) & (chess.BB_FILE_C | chess.BB_FILE_D | chess.BB_FILE_E | chess.BB_FILE_F)

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def count_pieces_in_mask_by_type(
    board: chess.Board,
    piece_types: Iterable[chess.PieceType],
    color: chess.Color,
    mask: chess.Bitboard,
) -> int:
    """
    Counts the total number of pieces of the given color and piece type in the given mask
    """
    total = 0
    piece_mask = board.pieces_mask
    for piece in piece_types:
        total += (piece_mask(piece, color) & mask).bit_count()
    return total


def count_pieces_in_mask(
    board: chess.Board,
    color: chess.Color,
    mask: chess.Bitboard,
) -> int:
    """
    Counts the total number of pieces of the given color in the given mask
    """
    return (mask & board.occupied_co[color]).bit_count()


def count_attackers_in_mask(
    board: chess.Board, color: chess.Color, mask: chess.Bitboard
) -> int:
    """
    Counts the number of pieces of the given color that are controlling squares
    in the given mask.
    A piece that is controlling more than one square in the mask will count more than once.
    Pinned pieces still count as attackers.
    """
    total = 0
    attackers_mask = board.attackers_mask

    bb = mask
    while bb:
        sq = chess.lsb(bb)
        bb &= bb - 1
        total += attackers_mask(color, sq).bit_count()

    return total
