import pytest
import chess
from src.features.extractors import *

INITIAL = chess.STARTING_FEN
ADVANCEMENT_COUNTS = "rnbqkbnr/p2p1ppp/8/2p1p3/1p2P3/2P2P1P/PP1P2P1/RNBQKBNR w KQkq - 0 5"
TWO_ISLANDS = "8/2p2ppp/8/8/8/8/PPP1PP2/8 w - - 0 1"
FOUR_ISLANDS = "8/8/8/8/8/P1P1P1P1/8/8 w - - 0 1"
EMPTY_BOARD = "8/8/8/8/8/8/8/8 w - - 0 1"

@pytest.mark.parametrize(
    "position, color, expected", [
        # Initial position: all pawns on starting squares are undefended by other pawns
        (INITIAL, chess.WHITE, 8),
        (INITIAL, chess.BLACK, 8),
    ]
)
def test_count_undefended_pawns(make_board, position, color, expected):
    board = make_board(position)
    pawn_bitboard = board.pawns & board.occupied_co[color]
    assert count_undefended_pawns(pawn_bitboard, color) == expected


@pytest.mark.parametrize(
    "position, color, expected", [
        # Initial position: 1 continuous island for both sides
        (INITIAL, chess.WHITE, 1),
        (INITIAL, chess.BLACK, 1),
        # Two pawn islands each
        (TWO_ISLANDS, chess.WHITE, 2),
        (TWO_ISLANDS, chess.BLACK, 2),
        # Pawns on a, c, e, g files (none adjacent) -> 4 islands
        (FOUR_ISLANDS, chess.WHITE, 4),
        # No pawns on the board -> 0 islands
        (EMPTY_BOARD, chess.WHITE, 0),
    ]
)
def test_count_pawn_islands(make_board, position, color, expected):
    board = make_board(position)
    # The island function only checks files and ignores color, but we extract one color's mask
    pawn_bitboard = int(board.pieces(chess.PAWN, color))
    assert count_pawn_islands(pawn_bitboard) == expected


@pytest.mark.parametrize(
    "position, color, expected", [
        # Initial position: pawns are on starting squares, advanced 0 squares
        (INITIAL, chess.WHITE, 0),
        (INITIAL, chess.BLACK, 0),
        # Some position where white has advanced 5 squares and black has advanced 7
        (ADVANCEMENT_COUNTS, chess.WHITE, 5),
        (ADVANCEMENT_COUNTS, chess.BLACK, 7)
    ]
)
def test_count_advanced_squares(make_board, position, color, expected):
    board = make_board(position)
    pawn_bitboard = int(board.pieces(chess.PAWN, color))
    assert count_advanced_squares(pawn_bitboard, color) == expected