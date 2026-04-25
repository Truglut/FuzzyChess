import chess
from pathlib import Path

from .base import FIS
from src.features.extractors import get_king_safety_params

CURRENT_DIR = Path(__file__).resolve().parent

KING_SAFETY_LUT_PATH = CURRENT_DIR.parent.parent.parent / "data" /"luts" / "safety_lut.npy"
MAX_KING_ATTACKERS_DIFF = 4
MIN_KING_ATTACKERS_DIFF = -4


class KingSafetyFIS(FIS):
    def __init__(self, lut_path: str):
        super().__init__(lut_path)

    def extract_params(self, board: chess.Board, color: chess.Color) -> tuple:
        return get_king_safety_params(board, color)

    def lookup(self, params: tuple) -> float:
        att_diff, shield = params

        att_diff = max(min(att_diff, MAX_KING_ATTACKERS_DIFF), MIN_KING_ATTACKERS_DIFF)

        return self.lut[att_diff + 4, shield]

    def compute(self, board: chess.Board, color: chess.Color):
        return self.lookup(self.extract_params(board, color))

    def __call__(self, board: chess.Board, color: chess.Color):
        return self.compute(board, color)
