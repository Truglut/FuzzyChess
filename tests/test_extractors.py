import pytest
import chess
from src.features.extractors import (
    count_pieces_in_mask,
    count_attackers_in_mask,
    CENTRAL_MASK,
)
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
