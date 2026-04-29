import pytest
import chess
from fuzzychess.search.search import alpha_beta_search
from positions import INITIAL

# FEN Constants for Testing
DRAW_POSITION = "k7/8/8/8/8/8/8/K7 w - - 0 1"  # King vs King (Insufficient material)
MATE_IN_1_WHITE = "4k3/Q6R/8/8/8/8/8/4K3 w - - 0 1"  # White to move, Qa8#, Qe7# or Rh8#
MATE_IN_1_BLACK = "4k3/8/8/8/8/6n1/q7/4K3 b - - 0 1"  # Black to move, Qe2#


# A simple dummy evaluation function for testing logic
def dummy_eval(board: chess.Board) -> float:
    return 1.5


def test_alpha_beta_checkmate_depth_preference(make_board):
    # Create a board where Black is already checkmated
    board = make_board("4k3/3RQ3/8/8/8/8/8/4K3 b - - 0 1")

    _, score_depth_2 = alpha_beta_search(board, dummy_eval, depth=2)
    _, score_depth_0 = alpha_beta_search(board, dummy_eval, depth=0)

    # Base checkmate is -10000. Depth is subtracted.
    assert score_depth_2 == -10002
    assert score_depth_0 == -10000


@pytest.mark.parametrize(
    "fen, turn, expected_score",
    [
        # dummy_eval returns 1.5.
        # For White to move, relative eval is 1.5.
        (INITIAL, chess.WHITE, 1.5),
        # For Black to move, relative eval is -1.5.
        (INITIAL, chess.BLACK, -1.5),
    ],
)
def test_alpha_beta_depth_zero(make_board, fen, turn, expected_score):
    board = make_board(fen)
    board.turn = turn

    best_move, score = alpha_beta_search(board, dummy_eval, depth=0)

    assert best_move is None
    assert score == expected_score


@pytest.mark.parametrize(
    "fen, depth, expected_winning_moves",
    [
        # White mates in 1 (depth 1 is enough to see the mate)
        (MATE_IN_1_WHITE, 1, ["a7a8", "a7e7", "h7h8"]),
        # Black mates in 1
        (MATE_IN_1_BLACK, 1, ["a2e2"]),
    ],
)
def test_alpha_beta_finds_mate_in_one(make_board, fen, depth, expected_winning_moves):
    board = make_board(fen)

    # Provide a flat eval function so the engine strictly relies on the mate score
    flat_eval = lambda b: 0.0
    best_move, score = alpha_beta_search(board, flat_eval, depth=depth)

    assert best_move is not None
    assert best_move.uci() in expected_winning_moves

    # Mate found at depth 1 -> the terminal node is hit at depth 0.
    # Terminal node returns -10000. Parent negates it to 10000.
    assert score == 10000
