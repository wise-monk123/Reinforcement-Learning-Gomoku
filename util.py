from IPython.display import HTML, display
from GamePlayerInterface import GamePlayerInterface
from GomokuBoard import GomokuBoard, GameResult, CROSS, NAUGHT
import tensorflow as tf


def print_board(board):
    display(HTML("""
    <style>
    .rendered_html table, .rendered_html th, .rendered_html tr, .rendered_html td {
      border: 1px  black solid !important;
      color: black !important;
    }
    </style>
    """ + board.html_str()))


def play_game(board: GomokuBoard, player1: GamePlayerInterface, player2: GamePlayerInterface):
    player1.new_game(CROSS)
    player2.new_game(NAUGHT)
    board.reset()

    finished = False
    while not finished:
        result, finished = player1.move(board)
        if finished:
            if result == GameResult.DRAW:
                final_result = GameResult.DRAW
            else:
                final_result = GameResult.CROSS_WIN
        else:
            result, finished = player2.move(board)
            if finished:
                if result == GameResult.DRAW:
                    final_result = GameResult.DRAW
                else:
                    final_result = GameResult.NAUGHT_WIN

    # noinspection PyUnboundLocalVariable
    player1.final_result(final_result)
    # noinspection PyUnboundLocalVariable
    player2.final_result(final_result)
    return final_result


def battle(player1: GamePlayerInterface, player2: GamePlayerInterface, num_games: int = 100000, silent: bool = False):
    board = GomokuBoard()
    draw_count = 0
    cross_count = 0
    naught_count = 0
    for _ in range(num_games):
        result = play_game(board, player1, player2)
        if result == GameResult.CROSS_WIN:
            cross_count += 1
        elif result == GameResult.NAUGHT_WIN:
            naught_count += 1
        else:
            draw_count += 1

    if not silent:
        print("After {} game we have draws: {}, GamePlayerInterface 1 wins: {}, and GamePlayerInterface 2 wins: {}.".format(num_games, draw_count,
                                                                                                  cross_count,
                                                                                                  naught_count))

        print("Which gives percentages of draws: {:.2%}, GamePlayerInterface 1 wins: {:.2%}, and GamePlayerInterface 2 wins:  {:.2%}".format(
            draw_count / num_games, cross_count / num_games, naught_count / num_games))

    return cross_count, naught_count, draw_count


def evaluate_players(p1: GamePlayerInterface, p2: GamePlayerInterface, games_per_battle=100, num_battles=100,
                     silent: bool = False):
    p1_wins = []
    p2_wins = []
    draws = []
    game_number = []
    game_counter = 0

    for i in range(num_battles):
        p1win, p2win, draw = battle(p1, p2, games_per_battle, silent)
        p1_wins.append(p1win)
        p2_wins.append(p2win)
        draws.append(draw)
        game_counter = game_counter + 1
        game_number.append(game_counter)
        if writer is not None:
            summary = tf.Summary(value=[tf.Summary.Value(tag='GamePlayerInterface 1 Win', simple_value=p1win),
                                        tf.Summary.Value(tag='GamePlayerInterface 2 Win', simple_value=p2win),
                                        tf.Summary.Value(tag='Draw', simple_value=draw)])
            writer.add_summary(summary, game_counter)

    return game_number, p1_wins, p2_wins, draws
