import chess


def get_piece_masks_dict(board: chess.Board) -> dict:
    return {
        color: {
            piece_type: board.pieces_mask(piece_type, color)
            for piece_type in chess.PIECE_TYPES
        }
        for color in chess.COLORS
    }

def extract_base_features(board: chess.Board) -> dict:
    # Piece masks
    features = get_piece_masks_dict(board)

    # Occupied by each color masks
    features["occupied"] = {color: board.occupied_co[color] for color in chess.COLORS}

    return features




