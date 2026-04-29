import chess
from typing import Iterable

CENTRAL_MASK = chess.BB_CENTER

EXTENDED_CENTRAL_MASK = (
    chess.BB_RANK_3 | chess.BB_RANK_4 | chess.BB_RANK_5 | chess.BB_RANK_6
) & (chess.BB_FILE_C | chess.BB_FILE_D | chess.BB_FILE_E | chess.BB_FILE_F)

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3.15,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

# Constants for adjacent file masking to prevent shift wrapping
NOT_A_FILE = 0xFEFEFEFEFEFEFEFE
NOT_H_FILE = 0x7F7F7F7F7F7F7F7F

# 64-bit mask to prevent integer expansion in Python during left shifts
MASK_64 = 0xFFFFFFFFFFFFFFFF


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


def get_pawn_attacks_bitboard(
    pawns_bitboard: chess.Bitboard, color: chess.Color
) -> chess.Bitboard:
    if color:
        return ((pawns_bitboard << 7) & ~chess.BB_FILE_H) | (
            (pawns_bitboard << 9) & ~chess.BB_FILE_A
        )

    return ((pawns_bitboard >> 7) & ~chess.BB_FILE_A) | (
        (pawns_bitboard >> 9) & ~chess.BB_FILE_H
    )


def collapse_bitboard_by_file(bb: chess.Bitboard) -> chess.Bitboard:
    bb |= bb >> 8
    bb |= bb >> 16
    bb |= bb >> 32
    return bb & 0xFF


def get_adjacent_files_bitboard(file: int) -> chess.Bitboard:
    if file == 0:
        return chess.BB_FILE_A | chess.BB_FILE_B | chess.BB_FILE_C
    elif file == 7:
        return chess.BB_FILE_H | chess.BB_FILE_G | chess.BB_FILE_F

    return chess.BB_FILES[file - 1] | chess.BB_FILES[file] | chess.BB_FILES[file + 1]