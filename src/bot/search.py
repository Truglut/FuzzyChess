import chess
from src.features.extractors_board import get_material_count
from typing import Callable, Tuple, Iterable


CHECKMATE_SCORE = 10_000


def alpha_beta_search(
    board: chess.Board,
    eval_function: Callable[[chess.Board], float],
    alpha: float = -float("inf"),
    beta: float = float("inf"),
    depth: int = 2,
    max_quiescence_depth: int | None = 10,
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
        max_quiescence_depth: Maximum depth to use in quiescence searh1.

    Returns:
        A tuple containing the best legal move (or None if the game is over)
        and its evaluation score from the perspective of the player to move.
    """

    # If depth is zero, start quiescence search
    if depth == 0:
        return quiescence_search(
            board,
            eval_function,
            alpha,
            beta,
            max_depth=max_quiescence_depth,
            cur_depth=depth,
        )

    # Initialize evaluation to extreme value and best move to None
    best_eval = -float("inf")
    best_move = None
    has_moves = False

    # Move search
    ordered_moves = order_moves(board.legal_moves)
    for move in ordered_moves:
        has_moves = True

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

    # Game over detection
    if not has_moves:
        if board.is_check():
            return None, -CHECKMATE_SCORE - depth
        else:
            return None, 0.0

    return best_move, best_eval


def quiescence_search(
    board: chess.Board,
    eval_function: Callable[[chess.Board], float],
    alpha: float = -float("inf"),
    beta: float = float("inf"),
    max_depth: int = 10,
    cur_depth: int = 0,
):
    # print("info string Going into quiescence search")
    # Assume the current player can get at least the current board evaluation
    color_multiplier = 1 if board.turn else -1
    stand_pat = color_multiplier * eval_function(board)

    # Beta cutoff
    if stand_pat >= beta:
        return None, stand_pat
    alpha = max(alpha, stand_pat)

    # Check for game over
    if board.is_check():
        if board.is_checkmate():
            return None, -CHECKMATE_SCORE - cur_depth
        ordered_moves = order_moves(board, board.generate_legal_moves())
    else:
        ordered_moves = order_captures(board, board.generate_legal_captures())

    # If max_depth has been reached, return static evaluation
    if max_depth == 0:
        return None, stand_pat

    # Initialize evaluation to extreme value and best move to None
    best_eval = stand_pat
    best_move = None

    # Move search: consider only captures
    for move in ordered_moves:
        board.push(move)

        _, eval_opp = quiescence_search(
            board,
            eval_function,
            alpha=-beta,
            beta=-alpha,
            max_depth=max_depth - 1,
            cur_depth=cur_depth - 1,
        )
        # Negate evaluation since quiescence search will return eval from opponents pov
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


def order_moves(board: chess.Board, moves: Iterable[chess.Move]) -> list[chess.Move]:
    """
    Orders moves using MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
    and prioritizing promotions
    """

    def move_score(move: chess.Move) -> int:
        score = 0

        if board.is_capture(move):
            if board.is_en_passant(move):
                score += 1
            else:
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                score += 1 + victim.piece_type - attacker.piece_type
        
        if move.promotion:
            score += 5
        
        return score
    
    return sorted(moves, key = move_score, reverse=True)



def order_captures(board: chess.Board, moves: Iterable[chess.Move]) -> list[chess.Move]:
    """
    Orders captures using MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
    and prioritizing promotions.
    """

    def move_score(move: chess.Move) -> int:
        if board.is_en_passant(move):
            score = 1
        else:
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            score = 1 + victim.piece_type - attacker.piece_type
        
        if move.promotion:
            score += 5
        
        return score
    
    return sorted(moves, key = move_score, reverse=True)
