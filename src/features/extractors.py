import chess
from typing import Tuple
from .utils import *


def get_center_piece_diff(board: chess.Board):
    """
    Calculates and returns the difference of number of pieces occupying the center.
    """

    occ_white = count_pieces_in_mask(board, chess.WHITE, CENTRAL_MASK)
    occ_black = count_pieces_in_mask(board, chess.BLACK, CENTRAL_MASK)

    return occ_white - occ_black


def get_center_att_diff(board: chess.Board):
    """
    Calculates and returns the difference of number of attackers on central squares.
    """

    att_white = count_attackers_in_mask(board, chess.WHITE, CENTRAL_MASK)
    att_black = count_attackers_in_mask(board, chess.BLACK, CENTRAL_MASK)

    return att_white - att_black


def get_center_params(board: chess.Board):
    """
    Calculates and returns the two basic features on which central control is based:
    difference of number of pieces occupying the center and
    difference of number of pieces attacking central squares.
    """

    return get_center_piece_diff(board), get_center_att_diff(board)


def get_king_attackers_diff(board: chess.Board, color: chess.Color):
    """
    Calculates the number of pieces of each color attacking the squares adjacent
    to the king (including the square the king is on).
    Returns the difference between the number of friendly pieces and enemy pieces.
    Pieces attacking more than one square adjacent to the king count more than once.
    Pinned pieces count as attackers.
    """
    king_square = board.king(color)
    if king_square is None:
        return 0

    king_ring = chess.BB_KING_ATTACKS[king_square] | chess.BB_SQUARES[king_square]

    # Calculate attackers and defenders on adjacent squares
    attackers = count_attackers_in_mask(board, not color, king_ring)
    defenders = (
        count_attackers_in_mask(board, color, king_ring) - king_ring.bit_count() + 1
    )

    return defenders - attackers


def get_pawn_shield_metric(board: chess.Board, color: chess.Color):
    """
    Calculates a measure of how solid the king's pawn shield is.
    For every adjacent file, the following is computed:
    - two points are added if there is a pawn one rank ahead of the king
    - otherwise, check if there are any pawns on the same rank as the king or two
      ranks ahead. If that's the case, add one point.

    The result is an integer between 0 and 6 (both inclusive).
    """
    king_square = board.king(color)

    king_rank = chess.square_rank(king_square)
    enemy_back_rank = 7 if color else 0

    # If the king is in the enemy backrank or adjacent rank, there is no shield
    if abs(king_rank - enemy_back_rank) <= 1:
        return 0

    friend_pawns = board.pawns & board.occupied_co[color]

    king_file = chess.square_file(king_square)
    direction = 1 if color else -1

    # Determine adjacent files
    if king_file == 0:
        adj_files = (0, 1, 2)
    elif king_file == 7:
        adj_files = (5, 6, 7)
    else:
        adj_files = (king_file - 1, king_file, king_file + 1)

    rank_1_idx = king_rank + direction
    rank_2_idx = king_rank + 2 * direction

    bb_rank_1 = chess.BB_RANKS[rank_1_idx]
    bb_rank_2 = chess.BB_RANKS[rank_2_idx]
    bb_king_rank = chess.BB_RANKS[king_rank]
    bb_rank_near = bb_rank_1 | bb_rank_2 | bb_king_rank

    shield = 0
    bb_king_rank = chess.BB_RANKS[king_rank]
    for file in adj_files:
        bb_file = chess.BB_FILES[file]

        # Pawns one rank ahead will count twice
        shield += min(1, (bb_file & friend_pawns & bb_rank_1).bit_count())
        shield += min(1, (bb_file & bb_rank_near & friend_pawns).bit_count())

    return shield


def get_king_safety_params(board: chess.Board, color: chess.Color):
    """
    Returns the two basic features on which king safety is based:
    - difference between enemy and friendly pieces attacking squares adjacent to the king,
    - measure of how solid the king's pawn shield is.
    """
    return get_king_attackers_diff(board, color), get_pawn_shield_metric(board, color)


def get_material_count(board: chess.Board):
    material_count = 0
    for piece_type, value in PIECE_VALUES.items():
        material_count += value * board.pieces_mask(piece_type, chess.WHITE).bit_count()
        material_count -= value * board.pieces_mask(piece_type, chess.BLACK).bit_count()

    return material_count

## Pawn structure features
# Number of pawns undefended by other pawns
def count_undefended_pawns(pawn_bitboard: int, color: chess.Color) -> int:
    return (pawn_bitboard & (~get_pawn_attacks_bitboard(pawn_bitboard, color))).bit_count()


# Number of pawn islands
def count_pawn_islands(pawn_bitboard:int) -> int:
    # Collapse pawn bitboard onto first rank
    collapsed = collapse_bitboard_by_file(pawn_bitboard)

    # Count number of pawn islands (pawns that don't have a pawn to their right)
    return (collapsed & (~ (collapsed << 1))).bit_count()


# Number of squares advanced by friendly pawns
def count_advanced_squares(pawn_bitboard: int, color: chess.Color) -> int:
    total_advanced = 0
    if color:
        total_advanced += (pawn_bitboard & chess.BB_RANK_3).bit_count() * 1
        total_advanced += (pawn_bitboard & chess.BB_RANK_4).bit_count() * 2
        total_advanced += (pawn_bitboard & chess.BB_RANK_5).bit_count() * 3
        total_advanced += (pawn_bitboard & chess.BB_RANK_6).bit_count() * 4
        total_advanced += (pawn_bitboard & chess.BB_RANK_7).bit_count() * 5
    else:
        total_advanced += (pawn_bitboard & chess.BB_RANK_2).bit_count() * 5
        total_advanced += (pawn_bitboard & chess.BB_RANK_3).bit_count() * 4
        total_advanced += (pawn_bitboard & chess.BB_RANK_4).bit_count() * 3
        total_advanced += (pawn_bitboard & chess.BB_RANK_5).bit_count() * 2
        total_advanced += (pawn_bitboard & chess.BB_RANK_6).bit_count() * 1

    return total_advanced
    

def get_pawn_structure_params(board: chess.Board, color: chess.Color) -> Tuple[int, int, int]:
    pawn_bitboard = board.pawns & board.occupied_co[color]

    return (
        count_advanced_squares(pawn_bitboard, color),
        count_undefended_pawns(pawn_bitboard, color),
        count_pawn_islands(pawn_bitboard)
    )
