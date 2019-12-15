from GomokuBoard import GomokuBoard, EMPTY, GameResult
from GamePlayerInterface import GamePlayerInterface
import random


class MinMaxRandomAlgorithm(GamePlayerInterface):

    WIN_VALUE = 1
    DRAW_VALUE = 0
    LOSS_VALUE = -1

    def __init__(self):

        self.side = None
        self.cache = {}

        super().__init__()

    def new_game(self, side: int):

        if self.side != side:
            self.side = side
            self.cache = {}

    def final_result(self, result: GameResult):

        pass

    def _min(self, board: GomokuBoard) -> int:

        # check available position
        board_hash = board.hash_value()
        if board_hash in self.cache:
            return random.choice(self.cache[board_hash])

        winner = board.who_won()
        if winner == self.side:
            best_moves = {(self.WIN_VALUE, -1)}
        elif winner == board.other_side(self.side):
            best_moves = {(self.LOSS_VALUE, -1)}
        else:

            min_value = self.DRAW_VALUE
            action = -1
            best_moves = {(min_value, action)}
            for index in [i for i, e in enumerate(board.state) if board.state[i] == EMPTY]:
                b = GomokuBoard(board.state)
                b.move(index, board.other_side(self.side))

                res, _ = self._max(b)
                if res < min_value or action == -1:
                    min_value = res
                    action = index
                    best_moves = {(min_value, action)}
                elif res == min_value:
                    action = index
                    best_moves.add((min_value, action))

        best_moves = tuple(best_moves)
        self.cache[board_hash] = best_moves

        return random.choice(best_moves)

    def _max(self, board: GomokuBoard) -> int:


        # check for available position
        board_hash = board.hash_value()
        if board_hash in self.cache:
            return random.choice(self.cache[board_hash])

        winner = board.who_won()
        if winner == self.side:
            best_moves = {(self.WIN_VALUE, -1)}
        elif winner == board.other_side(self.side):
            best_moves = {(self.LOSS_VALUE, -1)}
        else:
            max_value = self.DRAW_VALUE
            action = -1
            best_moves = {(max_value, action)}
            for index in [i for i, e in enumerate(board.state) if board.state[i] == EMPTY]:
                b = GomokuBoard(board.state)
                b.move(index, self.side)

                res, _ = self._min(b)
                if res > max_value or action == -1:
                    max_value = res
                    action = index
                    best_moves = {(max_value, action)}
                elif res == max_value:
                    action = index
                    best_moves.add((max_value, action))

        best_moves = tuple(best_moves)
        self.cache[board_hash] = best_moves

        return random.choice(best_moves)

    def move(self, board: GomokuBoard) -> (GameResult, bool):

        score, action = self._max(board)
        _, res, finished = board.move(action, self.side)
        return res, finished
