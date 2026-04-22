import pandas as pd
import chess
import argparse
from src.features.fis.king_safety import KingSafetyFIS, KING_SAFETY_LUT_PATH
from src.features.fis.center_control import CenterControlFIS, CENTER_LUT_PATH
from src.features.extractors import get_material_count


FEATURE_PIPELINE = {
    "king_safety_white": KingSafetyFIS(KING_SAFETY_LUT_PATH, chess.WHITE),
    "king_safety_black": KingSafetyFIS(KING_SAFETY_LUT_PATH, chess.BLACK),
    "center_control": CenterControlFIS(CENTER_LUT_PATH),
    "material_count": get_material_count,
}

COLUMNS_KEEP = ["Turn", "Stockfish_Eval"]


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
