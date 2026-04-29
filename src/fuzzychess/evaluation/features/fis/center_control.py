import chess

from fuzzychess.config import ROOT
from fuzzychess.evaluation.features.fis.base import FIS
from fuzzychess.evaluation.features.extractors_board import get_center_params

CENTER_LUT_PATH = ROOT / "data" / "luts" / "center_lut.npy"
MIN_CENTER_ATT_DIFF = -10
MAX_CENTER_ATT_DIFF = 10
MIN_CENTER_OCC_DIFF = -4
MAX_CENTER_OCC_DIFF = 4


class CenterControlFIS(FIS):
    def __init__(self, lut_path):
        super().__init__(lut_path)

    def extract_params(self, board: chess.Board) -> tuple:
        return get_center_params(board)

    def lookup(self, params) -> float:
        occ_diff, att_diff = params

        occ_diff = min(max(occ_diff, MIN_CENTER_OCC_DIFF), MAX_CENTER_OCC_DIFF)
        att_diff = min(max(att_diff, MIN_CENTER_ATT_DIFF), MAX_CENTER_ATT_DIFF)

        return self.lut[occ_diff + MIN_CENTER_OCC_DIFF, att_diff + MAX_CENTER_ATT_DIFF]
