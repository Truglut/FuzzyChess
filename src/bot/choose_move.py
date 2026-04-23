import torch
import chess
from .search import alpha_beta_search
from ..evaluation import SymmetricEvaluator
from ..features.extractors import get_material_count
from ..features.fis.center_control import CenterControlFIS, CENTER_LUT_PATH
from ..features.fis.king_safety import KingSafetyFIS, KING_SAFETY_LUT_PATH


MODEL_CHECKPOINT_PATH = "../models/model_checkpoint.pth"
checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu")

n_vars = len(checkpoint["feature_cols"])
model = SymmetricEvaluator(
    n_labels=checkpoint["n_labels"],
    var_configs=checkpoint["var_configs"],
    X_mean=torch.zeros(n_vars),
    X_std=torch.ones(n_vars),
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

king_safety_fis = KingSafetyFIS(KING_SAFETY_LUT_PATH)
center_control_fis = CenterControlFIS(CENTER_LUT_PATH)


def extract_features(board: chess.Board) -> tuple:
    return (
        king_safety_fis(board, chess.WHITE),
        king_safety_fis(board, chess.BLACK),
        center_control_fis(board),
        get_material_count(board),
    )


def evaluate_board(board: chess.Board) -> float:
    features = torch.tensor(extract_features(board))

    with torch.inference_mode():
        score = model(features).item()

    return score


def choose_move(board: chess.Board) -> chess.Move:
    return alpha_beta_search(board, evaluate_board)[0]