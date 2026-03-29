import pytest
import chess
from typing import Union, Iterable


@pytest.fixture
def make_board():
    def _make(
        position: Union[str, Iterable[str], chess.Board, None] = None,
    ) -> chess.Board:
        if position is None:
            return chess.Board()

        if isinstance(position, chess.Board):
            return position.copy()

        if isinstance(position, str):
            if "/" in position:
                return chess.Board(position)

            board = chess.Board()
            board.push_san(position)
            return board

        board = chess.Board()
        for move in position:
            board.push_san(move)
        return board

    return _make
