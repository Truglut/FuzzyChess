import chess
from src.features.extractors import get_material_count
from typing import Callable, Tuple


CHECKMATE_SCORE = 10_000


def alpha_beta_search(
    board: chess.Board,
    eval_function: Callable[[chess.Board], float],
    alpha: float = -float("inf"),
    beta: float = float("inf"),
    depth: int = 2,
) -> Tuple[chess.Move | None, float]:
    """
    Performs a minimax alpha-beta search (negamax variant) to find the best move.
    
    Args:
        board: The current python-chess Board.
        eval_function: A function returning the absolute static evaluation of the board 
                       (Positive = White advantage, Negative = Black advantage).
        alpha: The minimum score the maximizing player is assured of.
        beta: The maximum score the minimizing player is assured of.
        depth: The remaining depth to search in the tree.

    Returns:
        A tuple containing the best legal move (or None if the game is over) 
        and its evaluation score from the perspective of the player to move.
    """

    # Check for game over
    if board.is_game_over():
        if board.is_checkmate():
            return None, -CHECKMATE_SCORE - depth
        return None, 0.0

    # If depth is zero, return static evaluation from current players pov
    if depth == 0:
        color_multiplier = 1 if board.turn else -1
        return None, color_multiplier * eval_function(board)

    # Initialize evaluation to extreme value and best move to None
    best_eval = -float("inf")
    best_move = None

    # Move search
    for move in board.legal_moves:
        board.push(move)

        _, eval_opp = alpha_beta_search(
            board, eval_function, alpha=-beta, beta=-alpha, depth=depth - 1
        )
        # Negate evaluation since alpha_beta_search will return eval from opponents pov
        eval_move = -eval_opp

        board.pop()

        # Update best move
        if eval_move > best_eval:
            best_eval = eval_move
            best_move = move

            # Update alpha (maximum **assured** score for current player)
            alpha = max(alpha, eval_move)

        # Beta cutoff
        if eval_move >= beta:
            break

    return best_move, best_eval