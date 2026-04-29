import chess
import torch
from dataclasses import dataclass
from typing import Literal

from fuzzychess.evaluation.tapered import TaperedEvaluator
from fuzzychess.evaluation.features.fis.base import FIS
from fuzzychess.evaluation.features.extractors_cache import BoardCache


# FIS registration descriptor
FISKind = Literal["differential", "per_color", "compute_difference"]

@dataclass
class FISEntry:
    """
    Describes one registered FIS.

    Attributes
    ----------
    name : str
        Human-readable identifier used in diagnostics and feature dicts.
    fis : FIS
        The FIS instance (holds its LUT).
    kind : FISKind
        - "differential": the FIS receives a single tuple of params computed
          as (white - black) or similar symmetric diff. Produces one scalar.
          Maps to one non-paired variable in SymmetricEvaluator.
        - "per_color": the FIS is called twice - once per color. Produces two
          scalars (white, black). Maps to one *paired* variable in
          SymmetricEvaluator (parameter sharing enforced by the evaluator).
        - "compute_difference": the FIS is called twice - once per color.
          Then produces one scalar: difference between white and black scores.
    extract : callable[[BoardCache], tuple | tuple[tuple, tuple]]
        A function that receives a BoardCache and returns:
          - For "differential": a single params tuple  ->  fis.lookup(params)
          - For "per_color" or "compute_difference":    (white_params, black_params)
                                ->  fis.lookup(white_params), fis.lookup(black_params)
    """

    name: str
    fis: FIS
    kind: FISKind
    extract: callable


# ---------------- EvaluationEngine ------------------------

class EvaluationEngine:
    """
    Orchestrates the full evaluation pipeline for one board position

    Parameters
    ----------
    fis_entries : list[FISEntry]
        Ordered list of FIS descriptors. The order determines the column
        layout of the input tensor fed to SymmetricEvaluator:

            [material, fis_0_scalar, fis_1_white, fis_1_black, ...]

        "differential"       FIS contribute one column each.
        "per_color"          FIS contribute two columns (white then black),
                             which are marked as paired in var_configs.
        "compute_differece"  FIS contribute one column (white - black), unpaired.

    evaluator : SymmetricEvaluator
        The trainable PyTorch module.  Must have been constructed with
        var_configs consistent with `fis_entries` (see `make_var_configs`).
    """

    def __init__(
        self,
        mg_fis_entries: list[FISEntry],
        eg_fis_entries: list[FISEntry],
        evaluator: TaperedEvaluator,
    ):
        self.mg_fis_entries = mg_fis_entries
        self.eg_fis_entries = eg_fis_entries
        self.evaluator = evaluator
        self.evaluator.eval()

        # Pre-compute the column count so we can allocate tensors cheaply
        # material (1) + one col per differential + two cols per per_color
        self._n_features_mg = 1 + sum(
            2 if e.kind == "per_color" else 1 for e in self.mg_fis_entries
        )
        self._n_features_eg = 1 + sum(
            2 if e.kind == "per_color" else 1 for e in self.eg_fis_entries
        )

    # Class method: build var_configs from a list of FISEntry descriptors
    @staticmethod
    def make_var_configs(fis_entries: list[FISEntry]) -> list[dict]:
        """
        Derives the var_configs list expected by SymmetricEvaluator from
        a list of FISEntry objects.

        Usage
        -----
        var_configs = EvaluationEngine.make_var_configs(entries)
        evaluator   = SymmetricEvaluator(var_configs=var_configs, ...)
        engine      = EvaluationEngine(entries, evaluator)

        Layout
        ------
        - Material always comes first: non-paired.
        - "differential" FIS: non-paired, no learnable center (symmetric
          by construction — 0 is already the neutral point).
        - "per_color" FIS: paired, no learnable center.
        """
        configs = []

        # Material:
        configs.append({"learnable_center": False, "paired": False})

        for entry in fis_entries:
            if entry.kind == "per_color":
                configs.append({"learnable_center": False, "paired": True})
            else:  # differential or compute_difference
                configs.append({"learnable_center": False, "paired": False})

        return configs

    # Game phase calculation
    def _calculate_phase(self, board: chess.Board) -> float:
        """
        Calculates the game phase based on non-pawn material
        """

        # Count total pieces on the board for both colors
        knights = (
            board.pieces_mask(chess.KNIGHT, chess.WHITE).bit_count()
            + board.pieces_mask(chess.KNIGHT, chess.BLACK).bit_count()
        )
        bishops = (
            board.pieces_mask(chess.BISHOP, chess.WHITE).bit_count()
            + board.pieces_mask(chess.BISHOP, chess.BLACK).bit_count()
        )
        rooks = (
            board.pieces_mask(chess.ROOK, chess.WHITE).bit_count()
            + board.pieces_mask(chess.ROOK, chess.BLACK).bit_count()
        )
        queens = (
            board.pieces_mask(chess.QUEEN, chess.WHITE).bit_count()
            + board.pieces_mask(chess.QUEEN, chess.BLACK).bit_count()
        )

        # Calculate phase material
        phase_material = knights + bishops + (2 * rooks) + (4 * queens)

        # Normalize between 0.0 and 1.0 (capped at 24 just in case of promotions)
        phase = min(1.0, max(0.0, phase_material / 24.0))
        return phase

    # Core evaluation
    def evaluate(self, board: chess.Board) -> float:
        """
        Full pipeline for a single board position.
        Returns a scalar score (positive = white advantage).
        """
        cache = BoardCache.from_board(board)

        x_mg = self._extract_features(cache, self.mg_fis_entries, self._n_features_mg)
        x_eg = self._extract_features(cache, self.eg_fis_entries, self._n_features_eg)

        phase_val = self._calculate_phase(board)
        phase_tensor = torch.tensor([phase_val], dtype=torch.float32)

        with torch.no_grad():
            score = self.evaluator(x_mg, x_eg, phase_tensor).item()

        return score

    def extract_features_dict(self, board: chess.Board) -> dict[str, float]:
        """
        Returns a labeled dict of all features for debugging and visualization.
        """
        cache = BoardCache.from_board(board)
        features = {
            "material": cache.material_count,
            "game_phase": self._calculate_phase(board),
        }

        # Helper to populate dict cleanly
        def populate_dict(entries, prefix):
            for entry in entries:
                key = f"{prefix}_{entry.name}"
                if entry.kind == "differential":
                    params = entry.extract(cache)
                    features[key] = entry.fis.lookup(params)
                else:
                    white_params, black_params = entry.extract(cache)
                    features[f"{key}_white"] = entry.fis.lookup(white_params)
                    features[f"{key}_black"] = entry.fis.lookup(black_params)

        populate_dict(self.mg_fis_entries, "MG")
        populate_dict(self.eg_fis_entries, "EG")

        return features

    def _extract_features(
        self, cache: BoardCache, entries: list[FISEntry], n_features: int
    ) -> torch.Tensor:
        """
        Builds a (1, n_features) float32 tensor from a BoardCache.
        """
        values = torch.zeros(n_features, dtype=torch.float32)
        values[0] = cache.material_count

        k = 1
        for entry in entries:
            if entry.kind == "differential":
                params = entry.extract(cache)
                values[k] = entry.fis.lookup(params)
                k += 1
            elif entry.kind == "compute_difference":
                white_params, black_params = entry.extract(cache)
                diff = entry.fis.lookup(white_params) - entry.fis.lookup(black_params)
                values[k] = 0.5 * diff
                k += 1
            else:
                white_params, black_params = entry.extract(cache)
                values[k] = entry.fis.lookup(white_params)
                values[k + 1] = entry.fis.lookup(black_params)
                k += 2

        return values.unsqueeze(0)
