import chess
from typing import Callable, Tuple
from fuzzychess.search.search import order_captures, order_moves

CHECKMATE_SCORE = 10_000


class Searcher:
    def __init__(self, tt_size_mb: int = 64):
        # Estimate transposition table size
        entries = (tt_size_mb * 1024 * 1024) // 150
        self.tt_size = entries

        # Initialize transposition table as fixed-size list
        self.tt = [None] * self.tt_size

        self.nodes_searched = 0

    def clear_tt(self):
        """Clears the transpostion table"""
        self.tt = [None] * self.tt_size

    def alpha_beta_search(
        self,
        board: chess.Board,
        eval_function: Callable[[chess.Board], float],
        alpha: float = -float("inf"),
        beta: float = float("inf"),
        depth: int = 2,
        max_quiescence_depth: int | None = 10,
        use_quiescence: bool = True,
    ) -> Tuple[chess.Move | None, float]:
        self.nodes_searched += 1

        # Check transposition table
        zobrist_key = board._transposition_key()
        tt_index = hash(zobrist_key) % self.tt_size
        tt_entry = self.tt[tt_index]

        if tt_entry is not None:
            cached_key, cached_depth, cached_eval, cached_move = tt_entry
            if cached_key == zobrist_key and cached_depth >= depth:
                return cached_move, cached_eval

        # Check for repetition draws
        if board.is_repetition(2):
            return None, 0.0

        # If depth is zero, start quiescence search
        if depth == 0:
            if use_quiescence:
                return self.quiescence_search(
                    board,
                    eval_function,
                    alpha,
                    beta,
                    max_depth=max_quiescence_depth,
                    cur_depth=depth,
                )
            return eval_function(board)

        # Initialize evaluation to extreme value and best move to None
        best_eval = -float("inf")
        best_move = None
        has_moves = False

        # Move search
        ordered_moves = order_moves(board, board.legal_moves)
        for move in ordered_moves:
            has_moves = True

            board.push(move)

            _, eval_opp = self.alpha_beta_search(
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
                best_move = None
                best_eval = -CHECKMATE_SCORE - depth
            else:
                best_move = None
                best_eval = 0.0

        # Store position in transposition table
        self.tt[tt_index] = (zobrist_key, depth, best_eval, best_move)

        return best_move, best_eval

    def quiescence_search(
        self,
        board: chess.Board,
        eval_function: Callable[[chess.Board], float],
        alpha: float = -float("inf"),
        beta: float = float("inf"),
        max_depth: int = 10,
        cur_depth: int = 0,
    ):
        self.nodes_searched += 1

        is_check = board.is_check()
        if is_check:
            if board.is_checkmate():
                return None, -CHECKMATE_SCORE - cur_depth

            best_eval = -float("inf")
            ordered_moves = order_moves(board, board.generate_legal_moves())
        else:
            # Assume the current player can get at least the current board evaluation
            color_multiplier = 1 if board.turn else -1
            stand_pat = color_multiplier * eval_function(board)

            # Beta cutoff
            if stand_pat >= beta:
                return None, stand_pat

            alpha = max(alpha, stand_pat)
            best_eval = stand_pat
            ordered_moves = order_captures(board, board.generate_legal_captures())

        # If max_depth has been reached, return static evaluation
        if max_depth == 0:
            if is_check:
                color_multiplier = 1 if board.turn else -1
                stand_pat = color_multiplier * eval_function(board)
            return None, stand_pat

        # Initialize best move to None
        best_move = None

        # Move search: consider only captures (or evasions if in check)
        for move in ordered_moves:
            board.push(move)

            _, eval_opp = self.quiescence_search(
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
