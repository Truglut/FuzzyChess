import chess
from enum import Enum

from fuzzychess.search.search import order_moves, order_captures, CHECKMATE_SCORE
from fuzzychess.search.searcher import Searcher
from typing import Callable


class QSearchMode(Enum):
    NONE = "none"  # do not use quiescence search
    EVAL_ONLY = (
        "eval_only"  # use quiescence search for eval but not in principal variation
    )
    INCLUDE_PV = "include_pv"  # use quiescence search in eval and principal variation


class ExplainingSearcher(Searcher):
    """
    An extension of Searcher dedicated to interpretability and diagnostics.
    """

    def __init__(self, tt_size_mb=64):
        super().__init__(tt_size_mb)
        self.nodes_searched = 0

    def diagnostic_search(
        self,
        board: chess.Board,
        eval_function: Callable[[chess.Board], float],
        alpha=-float("inf"),
        beta=float("inf"),
        depth: int = 2,
        max_quiescence_depth: int = 10,
        q_mode: QSearchMode = QSearchMode.EVAL_ONLY,
    ):
        """
        Modified Alpha-Beta that returns the full Principal Variation (List of Moves)
        and the evaluation score.
        """

        # Repetition draws
        if board.is_repetition(2):
            return [], 0.0

        # Base case: depth 0. Handle the 3 Quiescence options here.
        if depth == 0:
            if q_mode == QSearchMode.NONE:
                return [], eval_function(board)

            elif q_mode == QSearchMode.EVAL_ONLY:
                # Use the base class's fast quiescence search
                _, eval_score = self.quiescence_search(
                    board, eval_function, alpha, beta, max_quiescence_depth, cur_depth=0
                )
                return [], eval_score

            elif q_mode == QSearchMode.INCLUDE_PV:
                # Use custom diagnostic q-search to track the capture line
                return self.diagnostic_quiescence(
                    board, eval_function, alpha, beta, max_quiescence_depth, cur_depth=0
                )

        # Initialization
        best_eval = -float("inf")
        principal_variation = []
        has_moves = False

        # Move ordering
        ordered_moves = order_moves(board, board.legal_moves)

        for move in ordered_moves:
            has_moves = True

            board.push(move)

            child_pv, eval_opp = self.diagnostic_search(
                board=board,
                eval_function=eval_function,
                alpha=-beta,
                beta=-alpha,
                depth=depth - 1,
                max_quiescence_depth=max_quiescence_depth,
                q_mode=q_mode,
            )
            eval_move = -eval_opp

            board.pop()

            # Update best move and principal variation
            if eval_move > best_eval:
                best_eval = eval_move
                principal_variation = [move] + child_pv
                alpha = max(alpha, eval_move)

            # beta cutoff
            if eval_move >= beta:
                break

        return principal_variation, best_eval

    def diagnostic_quiescence(
        self,
        board: chess.Board,
        eval_function: Callable[[chess.Board], float],
        alpha: float = -float("inf"),
        beta: float = float("inf"),
        max_depth: int = 10,
        cur_depth: int = 0,
    ) -> tuple[list[chess.Move], float]:
        """
        A version of quiescence search that tracks the sequence of captures.
        Only used if q_mode == QSearchMode.INCLUDE_PV.
        """
        is_check = board.is_check()
        principal_variation = []

        if is_check:
            if board.is_checkmate():
                return [], -CHECKMATE_SCORE - cur_depth
            best_eval = -float("inf")
            ordered_moves = order_moves(board, board.generate_legal_moves())
        else:
            color_multiplier = 1 if board.turn else -1
            stand_pat = color_multiplier * eval_function(board)

            if stand_pat >= beta:
                return [], stand_pat

            alpha = max(alpha, stand_pat)
            best_eval = stand_pat
            ordered_moves = order_captures(board, board.generate_legal_captures())

        if max_depth == 0:
            if is_check:
                color_multiplier = 1 if board.turn else -1
                stand_pat = color_multiplier * eval_function(board)
            return [], stand_pat

        for move in ordered_moves:
            board.push(move)

            child_pv, eval_opp = self.diagnostic_quiescence(
                board=board,
                eval_function=eval_function,
                alpha=-beta,
                beta=-alpha,
                max_depth=max_depth - 1,
                cur_depth=cur_depth - 1,
            )
            eval_move = -eval_opp

            board.pop()

            if eval_move > best_eval:
                best_eval = eval_move
                principal_variation = [move] + child_pv  # Track capture sequence
                alpha = max(alpha, eval_move)

            if eval_move >= beta:
                break

        return principal_variation, best_eval
