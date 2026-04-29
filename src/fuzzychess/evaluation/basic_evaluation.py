import chess

from fuzzychess.evaluation.features.extractors_board import get_material_count


# Basic evaluation functions for debugging and testing
def null_eval(board: chess.Board) -> float:
    return 0.0


def material_eval(board: chess.Board) -> float:
    return get_material_count(board)