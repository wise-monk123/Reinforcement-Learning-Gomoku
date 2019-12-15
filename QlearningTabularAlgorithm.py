from typing import Dict, List

from GomokuBoard import GomokuBoard, GameResult, NAUGHT, CROSS
from GamePlayerInterface import GamePlayerInterface

import numpy as np

WIN_VALUE = 1.0  # type: float
DRAW_VALUE = 0.5  # type: float
LOSS_VALUE = 0.0  # type: float


class TabularAlgorithm(GamePlayerInterface):
    """
    Implementing Tabular Q Learning
    """

    def __init__(self, alpha=0.9, gamma=0.95, q_init=0.6):

        self.side = None
        self.q = {}  # type: Dict[int, [float]]
        self.move_history = []  # type: List[(int, int)]
        self.learning_rate = alpha
        self.value_discount = gamma
        self.q_init_val = q_init
        super().__init__()

    def get_q(self, board_hash: int) -> [int]:

        # build the Q table in a lazy manner, only adding a state when it is actually used for the first time
        if board_hash in self.q:
            qvals = self.q[board_hash]
        else:
            qvals = np.full(9, self.q_init_val)
            self.q[board_hash] = qvals

        return qvals

    def get_move(self, board: GomokuBoard) -> int:

        board_hash = board.hash_value()  # type: int
        qvals = self.get_q(board_hash)  # type: [int]
        while True:
            m = np.argmax(qvals)  # type: int
            if board.is_legal(m):
                return m
            else:
                qvals[m] = -1.0

    def move(self, board: GomokuBoard):

        m = self.get_move(board)
        self.move_history.append((board.hash_value(), m))
        _, res, finished = board.move(m, self.side)
        return res, finished

    def final_result(self, result: GameResult):

        if (result == GameResult.NAUGHT_WIN and self.side == NAUGHT) or (
                result == GameResult.CROSS_WIN and self.side == CROSS):
            final_value = WIN_VALUE  # type: float
        elif (result == GameResult.NAUGHT_WIN and self.side == CROSS) or (
                result == GameResult.CROSS_WIN and self.side == NAUGHT):
            final_value = LOSS_VALUE  # type: float
        elif result == GameResult.DRAW:
            final_value = DRAW_VALUE  # type: float
        else:
            raise ValueError("Unexpected game result {}".format(result))

        self.move_history.reverse()
        next_max = -1.0  # type: float

        for h in self.move_history:
            qvals = self.get_q(h[0])
            if next_max < 0:  # First time through the loop
                qvals[h[1]] = final_value
            else:
                qvals[h[1]] = qvals[h[1]] * (
                            1.0 - self.learning_rate) + self.learning_rate * self.value_discount * next_max

            next_max = max(qvals)

    def new_game(self, side):

        self.side = side
        self.move_history = []
