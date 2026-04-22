import numpy as np
import chess


class FIS(object):
    """
    Base class for fuzzy inference systems.
    Subclasses define which board parameters they need and how to map them to a feature
    """
    name: str

    def __init__(self, lut_path: str):
        self.lut = np.load(lut_path)
    
    def extract_params(self, board: chess.Board) -> dict:
        """
        Extracts input parameters from the board.
        """
        raise NotImplementedError("Subclasses must implement the extract_params method")
    
    def lookup(self, params: dict) -> float:
        """
        Maps params to FIS output using the lookup table
        """
        raise NotImplementedError("Subclasses must implement the extract_params method")
    
    def compute(self, board: chess.Board) -> float:
        return self.lookup(self.extract_params(board))
    
    def __call__(self, board: chess.Board) -> float:
        return self.compute(board)
