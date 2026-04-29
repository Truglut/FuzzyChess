import chess
from pathlib import Path

from fuzzychess.config import ROOT
from fuzzychess.evaluation.features.fis.base import FIS
from fuzzychess.evaluation.features.extractors_board import get_promotion_chances_params

CURRENT_DIR = Path(__file__).resolve().parent

PROMOTION_LUT_PATH = ROOT / "data" / "luts" / "promotion_chances_lut.npy"


class PromotionFIS(FIS):
    def __init__(self, lut_path: str):
        super().__init__(lut_path)

    def extract_params(self, board: chess.Board, color: chess.Color) -> tuple:
        return get_promotion_chances_params(board, color)

    def lookup(self, params: tuple) -> float:
        return self.lut[*params]

    def compute(self, board: chess.Board, color: chess.Color):
        return self.lookup(self.extract_params(board, color))

    def __call__(self, board: chess.Board, color: chess.Color):
        return self.compute(board, color)
