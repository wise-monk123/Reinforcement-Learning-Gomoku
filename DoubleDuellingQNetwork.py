import numpy as np
import random
import tensorflow as tf
from TensorFlowController import TFSessionManager as TFSN

from GomokuBoard import GomokuBoard, BOARD_SIZE, EMPTY, CROSS, NAUGHT
from GamePlayerInterface import GamePlayerInterface, GameResult


class QNetwork:
    """
    Contains a TensorFlow graph which is suitable for learning the Gomoku Q function
    """

    def __init__(self, name: str, learning_rate: float):
        """
        Constructor for QNetwork. Takes a name and a learning rate for the GradientDescentOptimizer
        :param name: Name of the network
        :param learning_rate: Learning rate for the GradientDescentOptimizer
        """
        self.learningRate = learning_rate
        self.name = name

        # Placeholders

        self.input_positions = None
        self.target_q = None
        self.actions = None

        # Internal tensors
        self.actions_onehot = None
        self.value = None
        self.advantage = None

        self.td_error = None
        self.q = None
        self.loss = None

        # Externally useful tensors

        self.q_values = None
        self.probabilities = None
        self.train_step = None

        self.build_graph(name)

    def add_dense_layer(self, input_tensor: tf.Tensor, output_size: int, activation_fn=None,
                        name: str = None) -> tf.Tensor:

        return tf.layers.dense(input_tensor, output_size, activation=activation_fn,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                               name=name)

    def build_graph(self, name: str):

        with tf.variable_scope(name):
            self.input_positions = tf.placeholder(tf.float32, shape=(None, BOARD_SIZE * 3), name='inputs')
            self.target_q = tf.placeholder(shape=[None], dtype=tf.float32, name='target')

            net = self.input_positions

            net = self.add_dense_layer(net, BOARD_SIZE * 3 * 9, tf.nn.relu)

            self.value = self.add_dense_layer(net, 1, name='value')
            self.advantage = self.add_dense_layer(net, BOARD_SIZE, name='advantage')

            self.q_values = self.value + tf.subtract(self.advantage,
                                                     tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            self.probabilities = tf.nn.softmax(self.q_values, name='probabilities')

            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
            self.actions_onehot = tf.one_hot(self.actions, BOARD_SIZE, dtype=tf.float32)
            self.q = tf.reduce_sum(tf.multiply(self.q_values, self.actions_onehot), axis=1)

            self.td_error = tf.square(self.target_q - self.q)
            self.loss = tf.reduce_mean(self.td_error)

            self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate).minimize(self.loss,
                                                                                                          name='train')


class ReplayBuffer:
    """
    This class manages the Experience Replay buffer for the Neural Network player
    """

    def __init__(self, buffer_size=3000):
        """
        Creates a new `ReplayBuffer` of size `buffer_size`.
        :param buffer_size:
        """
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience: []):

        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:1] = []
        self.buffer.append(experience)

    def sample(self, size) -> []:
        size = min(len(self.buffer), size)
        return random.sample(self.buffer, size)


class ExpDoubleDuelQPlayer(GamePlayerInterface):

    def board_state_to_nn_input(self, state: np.ndarray) -> np.ndarray:

        res = np.array([(state == self.side).astype(int),
                        (state == GomokuBoard.other_side(self.side)).astype(int),
                        (state == EMPTY).astype(int)])
        return res.reshape(-1)

    def create_graph_copy_op(self, src: str, target: str, tau: float) -> [tf.Tensor]:

        src_vars = tf.trainable_variables(src)
        target_vars = tf.trainable_variables(target)

        op_holder = []

        for s, t in zip(src_vars, target_vars):
            op_holder.append(t.assign((s.value() * tau) + ((1 - tau) * t.value())))
        return op_holder

    def __init__(self, name: str, reward_discount: float = 0.95, win_value: float = 1.0, draw_value: float = 0.0,
                 loss_value: float = -1.0, learning_rate: float = 0.01, training: bool = True,
                 random_move_prob: float = 0.95, random_move_decrease: float = 0.95, batch_size=50,
                 pre_training_games: int = 500, tau: float = 0.001):

        self.tau = tau
        self.batch_size = batch_size
        self.reward_discount = reward_discount
        self.win_value = win_value
        self.draw_value = draw_value
        self.loss_value = loss_value
        self.side = None
        self.board_position_log = []
        self.action_log = []
        self.next_state_log = []

        self.name = name
        self.q_net = QNetwork(name + '_main', learning_rate)
        self.target_net = QNetwork(name + '_target', learning_rate)

        self.graph_copy_op = self.create_graph_copy_op(name + '_main', name + '_target', self.tau)
        self.training = training
        self.random_move_prob = random_move_prob
        self.random_move_decrease = random_move_decrease

        self.replay_buffer_win = ReplayBuffer()
        self.replay_buffer_loss = ReplayBuffer()
        self.replay_buffer_draw = ReplayBuffer()

        self.game_counter = 0
        self.pre_training_games = pre_training_games

        super().__init__()

    def new_game(self, side: int):
        self.side = side
        self.board_position_log = []
        self.action_log = []

    def add_game_to_replay_buffer(self, reward: float):

        game_length = len(self.action_log)

        if reward == self.win_value:
            buffer = self.replay_buffer_win
        elif reward == self.loss_value:
            buffer = self.replay_buffer_loss
        else:
            buffer = self.replay_buffer_draw

        for i in range(game_length - 1):
            buffer.add([self.board_position_log[i], self.action_log[i],
                        self.board_position_log[i + 1], 0])

        buffer.add([self.board_position_log[game_length - 1], self.action_log[game_length - 1], None, reward])

    def get_probs(self, input_pos: [np.ndarray], network: QNetwork) -> ([float], [float]):

        probs, qvalues = TFSN.get_session().run([network.probabilities, network.q_values],
                                                feed_dict={network.input_positions: input_pos})
        return probs, qvalues

    def get_valid_probs(self, input_pos: [np.ndarray], network: QNetwork, boards: [GomokuBoard]) -> ([float], [float]):

        probabilities, qvals = self.get_probs(input_pos, network)
        qvals = np.copy(qvals)
        probabilities = np.copy(probabilities)

        # We filter out all illegal moves by setting the probability to 0. We don't change the q values
        # as we don't want the NN to waste any effort of learning different Q values for moves that are illegal
        # anyway.
        for q, prob, b in zip(qvals, probabilities, boards):
            for index, p in enumerate(q):
                if not b.is_legal(index):
                    prob[index] = -1
                elif prob[index] < 0:
                    prob[index] = 0.0

        return probabilities, qvals

    def move(self, board: GomokuBoard) -> (GameResult, bool):

        # We record all game positions to feed them into the NN for training with the corresponding updated Q
        # values.
        self.board_position_log.append(board.state.copy())

        nn_input = self.board_state_to_nn_input(board.state)
        probs, _ = self.get_valid_probs([nn_input], self.q_net, [board])
        probs = probs[0]

        # Most of the time our next move is the one with the highest probability after removing all illegal ones.
        # Occasionally, however we randomly chose a random move to encourage exploration

        # noinspection PyUnresolvedReferences
        if (self.training is True) and \
                ((self.game_counter < self.pre_training_games) or (np.random.rand(1) < self.random_move_prob)):
            move = board.random_empty_spot()
        else:
            move = np.argmax(probs)

        # We record the action we selected as well as the Q values of the current state for later use when
        # adjusting NN weights.
        self.action_log.append(move)

        # We execute the move and return the result
        _, res, finished = board.move(move, self.side)
        return res, finished

    def final_result(self, result: GameResult):

        self.game_counter += 1

        # Compute the final reward based on the game outcome
        if (result == GameResult.NAUGHT_WIN and self.side == NAUGHT) or (
                result == GameResult.CROSS_WIN and self.side == CROSS):
            reward = self.win_value  # type: float
        elif (result == GameResult.NAUGHT_WIN and self.side == CROSS) or (
                result == GameResult.CROSS_WIN and self.side == NAUGHT):
            reward = self.loss_value  # type: float
        elif result == GameResult.DRAW:
            reward = self.draw_value  # type: float
        else:
            raise ValueError("Unexpected game result {}".format(result))

        self.add_game_to_replay_buffer(reward)

        # If we are in training mode we run the optimizer.
        if self.training and (self.game_counter > self.pre_training_games):

            batch_third = self.batch_size // 3
            train_batch = self.replay_buffer_win.sample(batch_third)
            train_batch.extend(self.replay_buffer_loss.sample(batch_third))
            train_batch.extend(self.replay_buffer_draw.sample(batch_third))
            train_batch = np.array(train_batch)

            #
            # Let's compute the target q values for all non terminal move
            # We extract the resulting state, run it through the target net work and
            # get the maximum q value (of all valid moves)
            next_states = [s[2] for s in train_batch if s[2] is not None]
            target_qs = []

            if len(next_states) > 0:
                probs, qvals = self.get_valid_probs([self.board_state_to_nn_input(s) for s in next_states],
                                                    self.target_net, [Board(s) for s in next_states])

                i = 0
                for t in train_batch:
                    if t[2] is not None:
                        max_move = np.argmax(probs[i])
                        max_qval = qvals[i][max_move]
                        target_qs.append(max_qval * self.reward_discount)
                        i += 1
                    else:
                        target_qs.append(t[3])

                if i != len(next_states):
                    print("Something wrong here!!!")
            else:
                target_qs.extend(train_batch[:, 3])

            # We convert the input states we have recorded to feature vectors to feed into the training.
            nn_input = [self.board_state_to_nn_input(x[0]) for x in train_batch]
            actions = train_batch[:, 1]
            # We run the training step with the recorded inputs and new Q value targets.
            TFSN.get_session().run([self.q_net.train_step],
                                   feed_dict={self.q_net.input_positions: nn_input,
                                              self.q_net.target_q: target_qs,
                                              self.q_net.actions: actions})

            TFSN.get_session().run(self.graph_copy_op)

            self.random_move_prob *= self.random_move_decrease
