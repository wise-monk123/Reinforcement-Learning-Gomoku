from GameController import print_board
from GomokuBoard import GomokuBoard, GameResult
from GamePlayerInterface import GamePlayerInterface


class RandomPlayer(GamePlayerInterface):
    """
    Play Gomoku randomly by choosing an available spot on the game board
    """

    def __init__(self):
        self.side = None
        super().__init__()

    def move(self, board: GomokuBoard) -> (GameResult, bool):
        # print('move to', board.random_empty_spot())
        #location = input("Your move: ")
        _, res, finished = board.move(board.random_empty_spot(), self.side)
        
       # print_board(board)

        return res, finished

    def final_result(self, result: GameResult):
        pass

    def new_game(self, side: int):
        self.side = side