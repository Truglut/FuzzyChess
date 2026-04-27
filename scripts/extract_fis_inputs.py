import pandas as pd
import chess
import argparse
from src.features.fis.king_safety import KingSafetyFIS, KING_SAFETY_LUT_PATH
from src.features.fis.center_control import CenterControlFIS, CENTER_LUT_PATH
from src.features.fis.pawn_structure import StructureFIS, STRUCTURE_LUT_PATH
from src.features.fis.mobility import MobilityFIS, MOBILITY_LUT_PATH
from src.features.fis.promotion import PromotionFIS, PROMOTION_LUT_PATH
from src.features.extractors_board import (
    get_material_count,
    get_king_distance_to_center,
    calculate_game_phase,
)


safety_fis = KingSafetyFIS(KING_SAFETY_LUT_PATH)
center_fis = CenterControlFIS(CENTER_LUT_PATH)
structure_fis = StructureFIS(STRUCTURE_LUT_PATH)
mobility_fis = MobilityFIS(MOBILITY_LUT_PATH)
promotion_fis = PromotionFIS(PROMOTION_LUT_PATH)

FEATURE_PIPELINE = {
    "king_safety_white": lambda board: safety_fis.compute(board, chess.WHITE),
    "king_safety_black": lambda board: safety_fis.compute(board, chess.BLACK),
    "center_control": CenterControlFIS(CENTER_LUT_PATH),
    "pawn_structure_white": lambda board: structure_fis.compute(board, chess.WHITE),
    "pawn_structure_black": lambda board: structure_fis.compute(board, chess.BLACK),
    "pawn_structure_diff": lambda board: 0.5 * (structure_fis.compute(board, chess.WHITE)
    - structure_fis.compute(board, chess.BLACK)),
    "mobility_white": lambda board: mobility_fis.compute(board, chess.WHITE),
    "mobility_black": lambda board: mobility_fis.compute(board, chess.BLACK),
    "mobility_diff": lambda board: 0.5*(mobility_fis.compute(board, chess.WHITE)
    - mobility_fis.compute(board, chess.BLACK)),
    "king_distance_to_center_white": lambda board: get_king_distance_to_center(
        board, chess.WHITE
    ),
    "king_distance_to_center_black": lambda board: get_king_distance_to_center(
        board, chess.BLACK
    ),
    "promotion_chances_white": lambda board: promotion_fis.compute(board, chess.WHITE),
    "promotion_chances_black": lambda board: promotion_fis.compute(board, chess.BLACK),
    "promotion_chances_diff": lambda board: 0.5 * (promotion_fis.compute(board, chess.WHITE)
    - promotion_fis.compute(board, chess.BLACK)),
    "material_count": get_material_count,
    "game_phase": calculate_game_phase,
}

COLUMNS_KEEP = ["Stockfish_Eval"]


def main():
    parser = argparse.ArgumentParser(
        description="Read positions from a csv file and write csv file with relevant features"
    )
    parser.add_argument("input_path", type=str, help="Path to the input .csv")
    parser.add_argument(
        "--out", required=True, type=str, help="Path to the output .csv file"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    output_features = []

    for _, row in df.iterrows():
        board = chess.Board(row["FEN"])
        row_features = {}

        for feature, extractor in FEATURE_PIPELINE.items():
            row_features[feature] = extractor(board)

        for feature in COLUMNS_KEEP:
            row_features[feature] = row[feature]

        output_features.append(row_features)

    pd.DataFrame(output_features).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
