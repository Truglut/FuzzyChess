import chess
from .base import FIS
from src.features.extractors import get_king_safety_params

KING_SAFETY_LUT_PATH = "data/luts/safety_lut.npy"
MAX_KING_ATTACKERS_DIFF = 4
MIN_KING_ATTACKERS_DIFF = -4

class KingSafetyFIS(FIS):
    def __init__(self, lut_path: str, color: chess.Color = chess.WHITE):
        super().__init__(lut_path)
        self.color = color


    def extract_params(self, board: chess.Board) -> tuple:
        return get_king_safety_params(board, self.color)
    

    def lookup(self, params: tuple) -> float:
        att_diff, shield = params

        att_diff = max(min(att_diff, MAX_KING_ATTACKERS_DIFF), MIN_KING_ATTACKERS_DIFF)

        return self.lut[att_diff + 4, shield]
    
