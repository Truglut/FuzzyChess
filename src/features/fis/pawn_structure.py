import chess
from pathlib import Path

from .base import FIS
from src.features.extractors_board import get_pawn_structure_params

CURRENT_DIR = Path(__file__).resolve().parent

STRUCTURE_LUT_PATH = (
    CURRENT_DIR.parent.parent.parent / "data" / "luts" / "structure_lut.npy"
)
MAX_SQUARES_ADVANCED = 10


class StructureFIS(FIS):
    def __init__(self, lut_path: str):
        super().__init__(lut_path)

    def extract_params(self, board: chess.Board, color: chess.Color) -> tuple:
        return get_pawn_structure_params(board, color)

    def lookup(self, params: tuple) -> float:
        squares_advanced, undef_pawns, pawn_islands = params
        squares_advanced = min(squares_advanced, MAX_SQUARES_ADVANCED)
        return self.lut[squares_advanced, undef_pawns, pawn_islands]

    def compute(self, board: chess.Board, color: chess.Color):
        return self.lookup(self.extract_params(board, color))

    def __call__(self, board: chess.Board, color: chess.Color):
        return self.compute(board, color)
