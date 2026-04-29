import torch
import chess
from pathlib import Path

from fuzzychess.config import ROOT
from fuzzychess.evaluation.engine import FISEntry, EvaluationEngine
from fuzzychess.evaluation.tapered import TaperedEvaluator
from fuzzychess.evaluation.evaluator import SymmetricEvaluator
from fuzzychess.evaluation.features.fis.king_safety import KingSafetyFIS, KING_SAFETY_LUT_PATH
from fuzzychess.evaluation.features.fis.center_control import CenterControlFIS, CENTER_LUT_PATH
from fuzzychess.evaluation.features.fis.pawn_structure import StructureFIS, STRUCTURE_LUT_PATH
from fuzzychess.evaluation.features.fis.mobility import MobilityFIS, MOBILITY_LUT_PATH
from fuzzychess.evaluation.features.fis.promotion import PromotionFIS, PROMOTION_LUT_PATH
from fuzzychess.evaluation.features.extractors_cache import (
    get_center_params,
    get_mobility_features,
    get_pawn_structure_params,
    get_king_safety_params,
    get_promotion_chances_params,
)

ANTECEDENT3_MODEL_CHECKPOINT_PATH = ROOT / "data" / "models" / "tapered_evaluator_antecedent_3.pth"
ANTECEDENT2_MODEL_CHECKPOINT_PATH = ROOT / "data" / "models" / "tapered_evaluator_antecedent_2.pth"

MG_FIS_ENTRIES = [
    FISEntry(
        name="Control Central",
        fis=CenterControlFIS(CENTER_LUT_PATH),
        kind="differential",
        extract=lambda cache: get_center_params(cache),
    ),
    FISEntry(
        name="Movilidad",
        fis=MobilityFIS(MOBILITY_LUT_PATH),
        kind="compute_difference",
        extract=lambda cache: (
            get_mobility_features(cache, chess.WHITE),
            get_mobility_features(cache, chess.BLACK),
        ),
    ),
    FISEntry(
        name="Estructura de peones",
        fis=StructureFIS(STRUCTURE_LUT_PATH),
        kind="compute_difference",
        extract=lambda cache: (
            get_pawn_structure_params(cache, chess.WHITE),
            get_pawn_structure_params(cache, chess.BLACK),
        ),
    ),
    FISEntry(
        name="Seguridad del rey",
        fis=KingSafetyFIS(KING_SAFETY_LUT_PATH),
        kind="per_color",
        extract=lambda cache: (
            get_king_safety_params(cache, chess.WHITE),
            get_king_safety_params(cache, chess.BLACK),
        ),
    ),
]


EG_FIS_ENTRIES = [
    FISEntry(
        name="Posibilidades de coronación",
        fis=PromotionFIS(PROMOTION_LUT_PATH),
        kind="compute_difference",
        extract=lambda cache: (
            get_promotion_chances_params(cache, chess.WHITE),
            get_promotion_chances_params(cache, chess.BLACK),
        ),
    ),
    FISEntry(
        name="Movilidad",
        fis=MobilityFIS(MOBILITY_LUT_PATH),
        kind="compute_difference",
        extract=lambda cache: (
            get_mobility_features(cache, chess.WHITE),
            get_mobility_features(cache, chess.BLACK),
        ),
    ),
    FISEntry(
        name="Estructura de peones",
        fis=StructureFIS(STRUCTURE_LUT_PATH),
        kind="compute_difference",
        extract=lambda cache: (
            get_pawn_structure_params(cache, chess.WHITE),
            get_pawn_structure_params(cache, chess.BLACK),
        ),
    ),
]

def load_bot(antecedent_length: int = 3) -> EvaluationEngine:
    if antecedent_length == 3:
        checkpoint = torch.load(ANTECEDENT3_MODEL_CHECKPOINT_PATH, map_location="cpu")
    else:
        checkpoint = torch.load(ANTECEDENT2_MODEL_CHECKPOINT_PATH, map_location="cpu")

    # Initialize middle game and endgame evaluators
    mg_var_configs = EvaluationEngine.make_var_configs(MG_FIS_ENTRIES)
    eg_var_configs = EvaluationEngine.make_var_configs(EG_FIS_ENTRIES)
    mg_evaluator = SymmetricEvaluator(mg_var_configs, antecedent_length=antecedent_length)  # same antecedent length used in training
    eg_evaluator = SymmetricEvaluator(eg_var_configs, antecedent_length=antecedent_length)  # same antecedent length used in training
    mg_evaluator.set_input_scales(torch.ones(mg_evaluator.n_vars))
    eg_evaluator.set_input_scales(torch.ones(eg_evaluator.n_vars))

    # Initialize global evaluator and load state dict
    evaluator = TaperedEvaluator(mg_evaluator=mg_evaluator, eg_evaluator=eg_evaluator)
    evaluator.load_state_dict(checkpoint["model_state_dict"])
    evaluator.eval()

    # Initialize the engine
    trained_engine = EvaluationEngine(MG_FIS_ENTRIES, EG_FIS_ENTRIES, evaluator)

    return trained_engine


