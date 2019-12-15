from abc import ABC, abstractmethod
from GomokuBoard import GomokuBoard, GameResult


class GamePlayerInterface(ABC):
    """
    Interface for Gomoku player
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def move(self, board: GomokuBoard) -> (GameResult, bool):

        pass

    @abstractmethod
    def final_result(self, result: GameResult):

        pass

    @abstractmethod
    def new_game(self, side: int):

        pass
