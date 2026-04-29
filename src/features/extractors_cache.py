import chess
from dataclasses import dataclass
from .utils import (
    CENTRAL_MASK,
    PIECE_VALUES,
    MASK_64,
    NOT_A_FILE,
    NOT_H_FILE,
    count_pieces_in_mask,
    count_attackers_in_mask,
    get_pawn_attacks_bitboard,
    collapse_bitboard_by_file,
    get_adjacent_files_bitboard,
)


# ---------------------------------------------------------------------------
# BoardCache
#
# Computes every reusable primitive ONCE per board position. Pass this object
# to all feature extractors instead of passing `board` directly so that no
# bitboard operation is repeated across calls.
#
# Construction is O(1) per field — no loops, no board traversal.
# The only loops happen inside the feature functions themselves, and only
# where unavoidable (e.g. iterating over piece squares for mobility).
# ---------------------------------------------------------------------------


@dataclass
class BoardCache:
    # Raw pawn bitboards
    white_pawns: int
    black_pawns: int
    all_pawns: int

    # Pawn attack masks (used by mobility AND pawn-structure features)
    white_pawn_attacks: int
    black_pawn_attacks: int

    # King squares and derived geometry
    white_king_sq: int
    black_king_sq: int
    white_king_rank: int
    black_king_rank: int
    white_king_file: int
    black_king_file: int
    white_king_ring: int  # BB_KING_ATTACKS[sq] | BB_SQUARES[sq]
    black_king_ring: int

    # King distance to centre (Manhattan, pre-computed as float)
    white_king_dist_to_center: float
    black_king_dist_to_center: float

    # Material balance (white positive)
    material_count: int

    # Keep a reference to the board for methods that have no pure-bitboard
    # equivalent (board.attacks_mask, board.pieces, count_attackers_in_mask…)
    board: chess.Board

    @staticmethod
    def from_board(board: chess.Board) -> "BoardCache":
        white_pawns = board.pawns & board.occupied_co[chess.WHITE]
        black_pawns = board.pawns & board.occupied_co[chess.BLACK]

        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        wk_rank = chess.square_rank(wk)
        bk_rank = chess.square_rank(bk)
        wk_file = chess.square_file(wk)
        bk_file = chess.square_file(bk)

        material = 0
        for piece_type, value in PIECE_VALUES.items():
            material += value * board.pieces_mask(piece_type, chess.WHITE).bit_count()
            material -= value * board.pieces_mask(piece_type, chess.BLACK).bit_count()

        return BoardCache(
            white_pawns=white_pawns,
            black_pawns=black_pawns,
            all_pawns=white_pawns | black_pawns,
            white_pawn_attacks=get_pawn_attacks_bitboard(white_pawns, chess.WHITE),
            black_pawn_attacks=get_pawn_attacks_bitboard(black_pawns, chess.BLACK),
            white_king_sq=wk,
            black_king_sq=bk,
            white_king_rank=wk_rank,
            black_king_rank=bk_rank,
            white_king_file=wk_file,
            black_king_file=bk_file,
            white_king_ring=chess.BB_KING_ATTACKS[wk] | chess.BB_SQUARES[wk],
            black_king_ring=chess.BB_KING_ATTACKS[bk] | chess.BB_SQUARES[bk],
            # Rescale king distances to center to [-10, 10]
            white_king_dist_to_center=3.3
            * (abs(wk_rank - 3.5) + abs(wk_file - 3.5) - 4),
            black_king_dist_to_center=3.3
            * (abs(bk_rank - 3.5) + abs(bk_file - 3.5) - 4),
            material_count=material,
            board=board,
        )

    # Convenience helpers so callers can use cache.pawns(color) etc.
    def pawns(self, color: chess.Color) -> int:
        return self.white_pawns if color else self.black_pawns

    def enemy_pawn_attacks(self, color: chess.Color) -> int:
        """Pawn attacks of the ENEMY of `color` — used by mobility."""
        return self.black_pawn_attacks if color else self.white_pawn_attacks

    def king_sq(self, color: chess.Color) -> int:
        return self.white_king_sq if color else self.black_king_sq

    def king_rank(self, color: chess.Color) -> int:
        return self.white_king_rank if color else self.black_king_rank

    def king_file(self, color: chess.Color) -> int:
        return self.white_king_file if color else self.black_king_file

    def king_ring(self, color: chess.Color) -> int:
        return self.white_king_ring if color else self.black_king_ring


# ---------------------------------------------------------------------------
# Feature extractors
# All functions accept a BoardCache instead of a raw chess.Board.
# ---------------------------------------------------------------------------

# ── Centre control ──────────────────────────────────────────────────────────


def get_center_params(cache: BoardCache):
    """
    Returns (center_piece_diff, center_att_diff) — white minus black.
    """
    board = cache.board
    piece_diff = count_pieces_in_mask(
        board, chess.WHITE, CENTRAL_MASK
    ) - count_pieces_in_mask(board, chess.BLACK, CENTRAL_MASK)
    att_diff = count_attackers_in_mask(
        board, chess.WHITE, CENTRAL_MASK
    ) - count_attackers_in_mask(board, chess.BLACK, CENTRAL_MASK)
    return piece_diff, att_diff


# ── King safety ─────────────────────────────────────────────────────────────


def get_king_attackers_diff(cache: BoardCache, color: chess.Color) -> int:
    """
    Defenders minus attackers around the king of `color`.
    Pieces attacking more than one adjacent square count more than once.
    """
    board = cache.board
    ring = cache.king_ring(color)
    attackers = count_attackers_in_mask(board, not color, ring)
    defenders = count_attackers_in_mask(board, color, ring) - ring.bit_count() + 1
    return defenders - attackers


def get_pawn_shield_metric(cache: BoardCache, color: chess.Color) -> int:
    """
    Pawn shield score [0, 6].
    Two points per file if a pawn is one rank ahead of the king;
    one point if a pawn is on the king rank or two ranks ahead.
    """
    king_rank = cache.king_rank(color)
    enemy_back_rank = 7 if color else 0

    if abs(king_rank - enemy_back_rank) <= 1:
        return 0

    friend_pawns = cache.pawns(color)
    king_file = cache.king_file(color)
    direction = 1 if color else -1

    if king_file == 0:
        adj_files = (0, 1, 2)
    elif king_file == 7:
        adj_files = (5, 6, 7)
    else:
        adj_files = (king_file - 1, king_file, king_file + 1)

    bb_rank_1 = chess.BB_RANKS[king_rank + direction]
    bb_rank_2 = chess.BB_RANKS[king_rank + 2 * direction]
    bb_king_rank = chess.BB_RANKS[king_rank]
    bb_rank_near = bb_rank_1 | bb_rank_2 | bb_king_rank

    shield = 0
    for file in adj_files:
        bb_file = chess.BB_FILES[file]
        shield += min(1, (bb_file & friend_pawns & bb_rank_1).bit_count())
        shield += min(1, (bb_file & bb_rank_near & friend_pawns).bit_count())

    return shield


def get_king_safety_params(cache: BoardCache, color: chess.Color):
    """
    Returns (king_attackers_diff, pawn_shield_metric).
    """
    return get_king_attackers_diff(cache, color), get_pawn_shield_metric(cache, color)


# ── Pawn structure ───────────────────────────────────────────────────────────


def count_undefended_pawns(pawn_bitboard: int, pawn_attacks: int) -> int:
    """Pawns not defended by any friendly pawn."""
    return (pawn_bitboard & ~pawn_attacks).bit_count()


def count_pawn_islands(pawn_bitboard: int) -> int:
    collapsed = collapse_bitboard_by_file(pawn_bitboard)
    return (collapsed & ~(collapsed << 1)).bit_count()


def count_advanced_squares(pawn_bitboard: int, color: chess.Color) -> int:
    if color:
        return (
            (pawn_bitboard & chess.BB_RANK_3).bit_count() * 1
            + (pawn_bitboard & chess.BB_RANK_4).bit_count() * 2
            + (pawn_bitboard & chess.BB_RANK_5).bit_count() * 3
            + (pawn_bitboard & chess.BB_RANK_6).bit_count() * 4
            + (pawn_bitboard & chess.BB_RANK_7).bit_count() * 5
        )
    else:
        return (
            (pawn_bitboard & chess.BB_RANK_6).bit_count() * 1
            + (pawn_bitboard & chess.BB_RANK_5).bit_count() * 2
            + (pawn_bitboard & chess.BB_RANK_4).bit_count() * 3
            + (pawn_bitboard & chess.BB_RANK_3).bit_count() * 4
            + (pawn_bitboard & chess.BB_RANK_2).bit_count() * 5
        )


def get_pawn_structure_params(cache: BoardCache, color: chess.Color):
    """
    Returns (advanced_squares, undefended_pawns, pawn_islands).
    Reuses pawn attack masks already computed in BoardCache.
    """
    pawn_bb = cache.pawns(color)
    pawn_attacks = cache.white_pawn_attacks if color else cache.black_pawn_attacks
    return (
        count_advanced_squares(pawn_bb, color),
        count_undefended_pawns(pawn_bb, pawn_attacks),
        count_pawn_islands(pawn_bb),
    )


# ── Pawn shield v2 ───────────────────────────────────────────────────────────


def calculate_pawn_shield(cache: BoardCache, color: chess.Color) -> int:
    """
    Alternative pawn shield metric (v2).
    Uses the pawn bitboard already stored in cache.
    """
    king_rank = cache.king_rank(color)
    if (color and king_rank >= 6) or (not color and king_rank <= 1):
        return 0

    king_file = cache.king_file(color)
    king_adj_sq = get_adjacent_files_bitboard(king_file) & chess.BB_RANKS[king_rank]
    pawn_bb = cache.pawns(color)

    if color:
        shifted_1 = pawn_bb >> 8
        pawn_metric = (shifted_1 & king_adj_sq).bit_count()
        pawn_metric += (
            (shifted_1 | (pawn_bb >> 16) | pawn_bb) & king_adj_sq
        ).bit_count()
    else:
        shifted_1 = pawn_bb << 8
        pawn_metric = (shifted_1 & king_adj_sq).bit_count()
        pawn_metric += (
            (shifted_1 | (pawn_bb << 16) | pawn_bb) & king_adj_sq
        ).bit_count()

    return pawn_metric


# ── Open files near king ─────────────────────────────────────────────────────


def count_open_adjacent_files(cache: BoardCache, file: int) -> int:
    adj_files_bb = get_adjacent_files_bitboard(file)
    local_pawns = cache.all_pawns & adj_files_bb
    return 3 - collapse_bitboard_by_file(local_pawns).bit_count()


# ── Mobility ─────────────────────────────────────────────────────────────────

PIECE_MOBILITY_TYPES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def get_mobility_features(cache: BoardCache, color: chess.Color):
    """
    Returns (piece_count, total_safe_moves, total_forward_moves).
    Uses the enemy pawn attack mask already computed in BoardCache.
    """
    board = cache.board
    enemy_attacks = cache.enemy_pawn_attacks(color)
    forward_ranks = 0xFFFFFFFF00000000 if color else 0x00000000FFFFFFFF

    piece_count = 0
    total_safe_moves = 0
    total_forward_moves = 0

    for piece_type in PIECE_MOBILITY_TYPES:
        for sq in board.pieces(piece_type, color):
            pseudo_legal = board.attacks_mask(sq)
            safe = pseudo_legal & ~enemy_attacks
            total_safe_moves += safe.bit_count()
            total_forward_moves += (safe & forward_ranks).bit_count()
            piece_count += 1

    return piece_count, total_safe_moves, total_forward_moves


# ── Promotion chances ────────────────────────────────────────────────────────


def count_passed_pawns(cache: BoardCache, color: chess.Color) -> int:
    """Uses pre-computed pawn bitboards from cache."""
    friend_pawns = cache.pawns(color)
    enemy_pawns = cache.pawns(not color)

    if color:
        span = enemy_pawns >> 8
        span |= span >> 8
        span |= span >> 16
        span |= span >> 32
    else:
        span = (enemy_pawns << 8) & MASK_64
        span |= (span << 8) & MASK_64
        span |= (span << 16) & MASK_64
        span |= (span << 32) & MASK_64

    span |= ((span << 1) & NOT_A_FILE) | ((span >> 1) & NOT_H_FILE)
    return collapse_bitboard_by_file(friend_pawns & ~span).bit_count()


def calculate_min_distance_to_promotion(cache: BoardCache, color: chess.Color) -> int:
    pawns = cache.pawns(color)
    if not pawns:
        return 8
    if color:
        return 7 - ((pawns.bit_length() - 1) >> 3)
    else:
        return ((pawns & -pawns).bit_length() - 1) >> 3


def get_promotion_chances_params(cache: BoardCache, color: chess.Color):
    return (
        calculate_min_distance_to_promotion(cache, color),
        count_passed_pawns(cache, color),
    )


# ── King distance to centre ──────────────────────────────────────────────────


def get_king_distance_to_center(cache: BoardCache, color: chess.Color) -> float:
    return cache.white_king_dist_to_center if color else cache.black_king_dist_to_center


# ---------------------------------------------------------------------------
# Top-level entry point
#
# Call this ONCE per node in the minimax tree. It builds the cache and
# returns every feature group in a single dictionary.
# ---------------------------------------------------------------------------


def extract_all_features(board: chess.Board) -> dict:
    cache = BoardCache.from_board(board)

    center_piece_diff, center_att_diff = get_center_params(cache)

    return {
        # Material
        "material": cache.material_count,
        # Centre
        "center_piece_diff": center_piece_diff,
        "center_att_diff": center_att_diff,
        # King safety — white
        "w_king_att_diff": get_king_attackers_diff(cache, chess.WHITE),
        "w_pawn_shield": get_pawn_shield_metric(cache, chess.WHITE),
        "w_pawn_shield_v2": calculate_pawn_shield(cache, chess.WHITE),
        "w_king_dist_center": cache.white_king_dist_to_center,
        "w_open_adj_files": count_open_adjacent_files(cache, cache.white_king_file),
        # King safety — black
        "b_king_att_diff": get_king_attackers_diff(cache, chess.BLACK),
        "b_pawn_shield": get_pawn_shield_metric(cache, chess.BLACK),
        "b_pawn_shield_v2": calculate_pawn_shield(cache, chess.BLACK),
        "b_king_dist_center": cache.black_king_dist_to_center,
        "b_open_adj_files": count_open_adjacent_files(cache, cache.black_king_file),
        # Pawn structure — white
        "w_advanced_squares": count_advanced_squares(cache.white_pawns, chess.WHITE),
        "w_undefended_pawns": count_undefended_pawns(
            cache.white_pawns, cache.white_pawn_attacks
        ),
        "w_pawn_islands": count_pawn_islands(cache.white_pawns),
        # Pawn structure — black
        "b_advanced_squares": count_advanced_squares(cache.black_pawns, chess.BLACK),
        "b_undefended_pawns": count_undefended_pawns(
            cache.black_pawns, cache.black_pawn_attacks
        ),
        "b_pawn_islands": count_pawn_islands(cache.black_pawns),
        # Mobility — white
        "w_mobility": get_mobility_features(cache, chess.WHITE),
        # Mobility — black
        "b_mobility": get_mobility_features(cache, chess.BLACK),
        # Promotion — white
        "w_promotion": get_promotion_chances_params(cache, chess.WHITE),
        # Promotion — black
        "b_promotion": get_promotion_chances_params(cache, chess.BLACK),
    }
