import chess
from pathlib import Path

from fuzzychess.config import ROOT
from fuzzychess.evaluation.features.fis.base import FIS
from fuzzychess.evaluation.features.extractors_board import get_mobility_features

CURRENT_DIR = Path(__file__).resolve().parent

MOBILITY_LUT_PATH = ROOT / "data" / "luts" / "mobility_lut.npy"
MAX_PIECE_COUNT = 6
MAX_TOTAL_SAFE_MOVES = 80
MAX_FORWARD_SAFE_MOVES = 49


class MobilityFIS(FIS):
    def __init__(self, lut_path: str):
        super().__init__(lut_path)

    def extract_params(self, board: chess.Board, color: chess.Color) -> tuple:
        return get_mobility_features(board, color)

    def lookup(self, params: tuple) -> float:
        piece_count, total_safe_moves, forward_moves = params

        total_safe_moves = min(total_safe_moves, MAX_TOTAL_SAFE_MOVES)
        forward_moves = min(forward_moves, MAX_FORWARD_SAFE_MOVES)
        piece_count = min(piece_count, MAX_PIECE_COUNT)

        return self.lut[piece_count, total_safe_moves, forward_moves]

    def compute(self, board: chess.Board, color: chess.Color):
        return self.lookup(self.extract_params(board, color))

    def __call__(self, board: chess.Board, color: chess.Color):
        return self.compute(board, color)
