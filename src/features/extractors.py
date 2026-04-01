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
    chess.BISHOP: 3.15,
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


def get_center_params(board: chess.Board):
    """
    Calculates and returns the two basic features on which central control is based:
    difference of number of pieces occupying the center and
    difference of number of pieces attacking central squares.
    """
    occupied_white = count_pieces_in_mask(board, chess.WHITE, CENTRAL_MASK)
    occupied_black = count_pieces_in_mask(board, chess.BLACK, CENTRAL_MASK)

    attackers_white = count_attackers_in_mask(board, chess.WHITE, CENTRAL_MASK)
    attackers_black = count_attackers_in_mask(board, chess.BLACK, CENTRAL_MASK)

    return occupied_white - occupied_black, attackers_white - attackers_black


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
    defenders = count_attackers_in_mask(board, color, king_ring)

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

    # If the king is in the enemy back rank, there is no shield
    if king_rank == enemy_back_rank:
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
    rank_2_idx = king_rank + direction

    bb_rank_1 = chess.BB_RANKS[rank_1_idx] if 0 <= rank_1_idx <= 7 else 0
    bb_rank_2 = chess.BB_RANKS[rank_2_idx] if 0 <= rank_2_idx <= 7 else 0
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
