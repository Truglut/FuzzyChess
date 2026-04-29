import chess

from ..features.extractors_board import get_material_count


def null_eval(board: chess.Board) -> float:
    return 0.0


def material_eval(board: chess.Board) -> float:
    return get_material_count(board)