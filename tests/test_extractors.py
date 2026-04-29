import pytest
import chess
from fuzzychess.evaluation.features.extractors_board import *
from positions import *


# Test for central occupation
@pytest.mark.parametrize(
    "position, color, expected",
    [
        # Initial position: no pieces in the center
        (INITIAL, chess.WHITE, 0),
        (INITIAL, chess.BLACK, 0),
        # e4 e5 - one white pawn and one black pawn in the center
        (E4_E5, chess.WHITE, 1),
        (E4_E5, chess.BLACK, 1),
        # A position where black has somewhat neglected control of the center
        # white has pawns on d4, e5; black has a pawn on d5. The e4 square is empty
        (BLACK_DELAYS, chess.WHITE, 2),
        (BLACK_DELAYS, chess.BLACK, 1),
    ],
)
def test_count_pieces_in_mask_center(make_board, position, color, expected):
    board = make_board(position)
    assert count_pieces_in_mask(board, color, CENTRAL_MASK) == expected


# Test for central attackers
@pytest.mark.parametrize(
    "position, color, expected",
    [
        # Initial position: no attacks on central squares
        (INITIAL, chess.WHITE, 0),
        # 1. e4 e5 2. Nf3 Nc6
        # White: knight on f3 attacks d4 and e5; e4 pawn attacks d5 -> 3 attackers
        (NORMAL_VARIATION, chess.WHITE, 3),
        # Black: similar - knight controls 2 squares, e5 pawn controls d4 -> 3 attackers
        (NORMAL_VARIATION, chess.BLACK, 3),
        # Position with slight white advantage:
        # White pawn on d4 controls e5; knights on c3 and f3 control 2 central
        # squares each; queen controls d4 -> 6 attackers.
        # Black pawn on d5 controls e4; queen and bishop control e4 -> 3 attackers
        (BLACK_DELAYS, chess.WHITE, 6),
        (BLACK_DELAYS, chess.BLACK, 3),
    ],
)
def test_count_attackers_in_mask_center(make_board, position, color, expected):
    board = make_board(position)
    assert count_attackers_in_mask(board, color, CENTRAL_MASK) == expected


@pytest.mark.parametrize(
    "position, color, expected",
    [
        (INITIAL, chess.WHITE, 7),
        (INITIAL, chess.BLACK, 7),
        (BLACK_ATTACK, chess.WHITE, 5 - 10),
        (BLACK_ATTACK, chess.BLACK, 3 - 1),
    ],
)
def test_get_king_attackers_diff(make_board, position, color, expected):
    board = make_board(position)
    assert get_king_attackers_diff(board, color) == expected


@pytest.mark.parametrize(
    "position, color, expected", [
        # Initial position
        (INITIAL, chess.WHITE, 6),
        (INITIAL, chess.BLACK, 6),
        # Test doubled pawns and pawns on same rank as the king
        (WHITE_DOUBLED_PAWNS, chess.WHITE, 4),
        (KINGS_WITH_PAWNS, chess.WHITE, 4),
        (KINGS_WITH_PAWNS, chess.BLACK, 5),
        # Test king on one of the edge files
        ("8/8/8/8/8/P1P5/KPP5/8 w - - 0 1", chess.WHITE, 5),
        ("8/8/8/8/5P2/7P/6PK/8 w - - 0 1", chess.WHITE, 4),
        # Test king on enemy backrank or second-to-last rank
        (KINGS_CLOSETO_BACKRANKS, chess.WHITE, 0),
        (KINGS_CLOSETO_BACKRANKS, chess.BLACK, 0)
    ]
)
def test_get_pawn_shield_metric(make_board, position, color, expected):
    board = make_board(position)
    assert get_pawn_shield_metric(board, color) == expected


@pytest.mark.parametrize(
    "position, color, expected", [
        # Initial position
        (INITIAL, chess.WHITE, 6),
        (INITIAL, chess.BLACK, 6),
        # Test doubled pawns and pawns on same rank as the king
        (WHITE_DOUBLED_PAWNS, chess.WHITE, 4),
        (KINGS_WITH_PAWNS, chess.WHITE, 4),
        (KINGS_WITH_PAWNS, chess.BLACK, 5),
        # Test king on one of the edge files
        ("8/8/8/8/8/P1P5/KPP5/8 w - - 0 1", chess.WHITE, 5),
        ("8/8/8/8/5P2/7P/6PK/8 w - - 0 1", chess.WHITE, 4),
        # Test king on enemy backrank or second-to-last rank
        (KINGS_CLOSETO_BACKRANKS, chess.WHITE, 0),
        (KINGS_CLOSETO_BACKRANKS, chess.BLACK, 0)
    ]
)
def test_calculate_pawn_shield(make_board, position, color, expected):
    board = make_board(position)
    pawn_bitboard = board.pawns & board.occupied_co[color]
    assert calculate_pawn_shield(board, color, pawn_bitboard) == expected
