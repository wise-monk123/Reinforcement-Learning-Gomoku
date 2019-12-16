## Project Title: Gomoku: an AI human vs computer game

## Team Member: 
Jia Ma,
Ying (Laura) Liu,
Yuhua He,
Yuanzhe Li,
Samuel Yang 

This project is prepared for CMPE 297-99 instructed by Professor Jahan Ghofraniha.

## Table of Content:
- Executive Summary
- Background and Introduction
- Problem Statement
- Purpose/Motivation
- Differentiator/Contribution
- Methodology
- Implementation and Results
- Conclusions
- Appendix
- Notebook link
- References

## Executive Summary:
This project is an implementation of four models to enable computer to combat a human player in the Gomoku game (as known as five in a row). Key reinforcement learning concepts implemented in this project:
- Max Min algorithm
- Q-Learning
- Neural network
- Markov decision process

We first studied how max min algorithm can be applied in the Gomuku games by drawing out the branches and leaves of all possible moves, and used greedy policy of selecting best move to implete this model. We evaluated the model results by playing 1000 games, and the result is satisfying as a non-artificial intelligence application. 
Since mix max algorithm cannot handle players with other gaming strategies, we then applied Q-learning principle and built another model by saving the Q-learning score for all possible moves in the table with initial score assignment of 0.6. We evaluated the model by plotting multiple graphs with various order permutation and combination of random-move player, and min/max-move player combat against each other , and these graphs show Q-learning is an improved model. 
However, given the 8x8 board Gomoku game’s large potential score computation (1.2688693e+89 combination of moves, and 3.4336838e+30 number of total states), Q learning’s scrore table is not efficient even though we used computer cache to save the score for each state/move pair so that the computation happens only once. To resolve this inefficiency issue, we developed our next neural network model by feeding a one dimension vector data into tensor flow. This significantly improves the training time since Tensor topology doesn’t require saving the table data like Q-learning model. After evaluating the neural network model version #1, it seems the training outcome is not great compared to the pure min-max algorithm model and Q learning model. 
Finally, we improved the neural network model version #1 by using 2 dimensional pane as input data and adding a greedy decentent since in real life the board game environment is a flat 2 dimensional space, and a player is not 100% of the time making best possible move. We used ReLu as activation function, initiated weight by variance scaling initializer, and softmax in the output layer.  This neural network model #2 finally achieved both goals of efficiency and accuracy in the Computer versus Human gomoku game. 

![model screen](https://github.com/wise-monk123/Reinforcement-Learning-Gomoku/blob/master/Executive.png)

## Background and Introduction
Gomoku is a popular game played in many countries. The player wins by having their pieces forming a straight-line connection with the length of 5 in an 8x8 board. The connection can be either horizontal, vertical or diagonal. The player wins by having their symbol forming a connection with the length of 3. The connection can be either horizontal, vertical or diagonal. 

![model screen](https://github.com/wise-monk123/Reinforcement-Learning-Gomoku/blob/master/img/background.png)

## Problem Statement
There are two primary gaming models existing: AlphaGo and Tic-Tac-Toe related. AlphaGo model is too complex for Gomoku, taking prolonged time to train. We faced several problems with this model, such as running GPU using pytorch causing memory issues. Besides having a physical GPU, setting up proper environments for pytorch GPU is a challenge for us. 
The tic-tac-toe related models had a lot of memory and efficiency related issue if being applied to Gomoku, since tic-tac-toe is a 3x3 dimension based board, while Gomoku is a 8x8 dimension based board. The computation needs for Gomoku is exponentially higher than tic-tac-toe.To be specific, Gomuku game has 1.2688693e+89 combination of moves, and 3.4336838e+30 number of total states versus tic-tac-toe’s 362,800 combination of moves, and 19,683  number of total states.

## Purpose and Motivation
We want to develop an application so that users can play with machines in their leisure time, and find the opponent (computer) challenging. This application should be able to make a wise  move by learning from the opponent player’s prior strategies. In addition, the application should be able to run efficiently. 
Differentriator and Contribution
Current board game models mainly use Monte Carlo Tree Search principle, this project used max min algorithm, Q learning, and neural network (2 versions in Tensorflow) and built 4 models. The max min algorithm model is fast to run and simply uses loops to return the next move. The Q-learning model provides more flexibility by learning the opponent 's strategies, thus improved accuracy. The neural network Tensorflow model resolves Q-learning model’s table data storage inefficiency issue. The final version of neural network model improves both the efficiency and accuracy of  existing Gomoku applications. 

## Methodology
We used Python, Tensorflow as main tools, implemented max-min algorithm model, tabular Q-learning model, neural network model, and improved neural network model. These four models are compared again each other by plotting thousand-number-of games graphs. 

## Implementation and Results
Firstly, in the research and reinforcement learning definition stage, we initiated the board and defined the state of each play. There are four possible states of game move: draw, not finished, won, lost. Assuming each player always makes best-possible move, the game result is 100% draw. The other assumption is that, if each player makes random move, the chance of one player winning is 58%, losing is 29%, and drawing is 13% from 1000 games playing statistics. The reward for each player is defined as 1 for a win,  -1 for a loss, and a 0 for a draw. For setting up the board, there are possible 8x8=64 squares to occupy for both players, which essentially 64! = 1.2688693e+89 combination of moves, such as first move has 64 squares to choose from, second move has 63 squares to choose from etc, till the last move has 1 square to land. To simplify this process, we assume that to reach game end, not all the 64 board squares need to be occupied. Since each square will have only 3 states (win, loss, draw), the number of total states is 3 to the 64 power (3.4336838e+30). Given this large number of permutation combination, we planned to save the state computation in the computer’s cache so that each move’s reward computation is only calculated one time. 
 
Secondly, in the model building stage, we have built various models by using different reinforcement learning algorithms. The Min-Max algorithm states that the two players always make perfect best move for themselves, meaning every player always chooses one move among all the possible moves, which provide the maximum game value for each player. Since the opponent player always makes the move that has maximum value for them, which is essentially minimum value for us. In details, the board is 8x8, which has 64! possible move combinations. A tree graph would be best to illustrate the min-max principle. It would have all the possible moves for both players as branches and leaves at the game end, where we assign the reward number (-1,0,1) to the leave. If the leave is the final move for us, we backpropagate the maximum reward state to the previous branch, and backpropage the minimum reward value to the branch prior since it’s the opponent player’s term. Similarly, the reward value is assigned all the way up to the root.  Now, we need two code files to deal with a decision on what move to choose from if multiple moves have same reward score. MinMaxAlgorithm.py is for a deterministic decision, where the player always choose the same move in this situation. While MinMaxRandomAlgorithm.py is for a random decision, where the player chooses a random move in this situation.  We run 1000 game loops to let MinMax player and MinMax Random player to go against each other, and found that Min Max player always has significantly more winning games than MinMax Random player; Min Max player always has 100% draws against itself in the 1000 games. 

<img src="https://github.com/wise-monk123/Reinforcement-Learning-Gomoku/blob/master/img/Maxmin.png" width="200">


Mix-Max algorithm’s baseline result leads us to use Quality learning (Q learning) to build an artificial intelligent model. Compare to Mix-Max algorithm’s reward assignment on “move” only, Q-learning assigns a reward value to a pair of move and state. The backpropagation in Q-learning has a discount factor, unlike Mix-Max algorithm’s full reward score backpropagation.

<img src="https://github.com/wise-monk123/Reinforcement-Learning-Gomoku/blob/master/img/Qlearning.png" width="400">

To initiate the initial Q-learning score, we tried several values and plotted below graphs: 
Initial value = 1.0
<img src="https://github.com/wise-monk123/Reinforcement-Learning-Gomoku/blob/master/img/1.png" width="400">

Initial value = 0.001 

<img src="https://github.com/wise-monk123/Reinforcement-Learning-Gomoku/blob/master/img/0.001.png" width="400">

As the above two graph shows, when a player is too optimistic (value = 1) about the first move and considers that every move will lead to a win, it takes quite a longer time (about 2000 games) to complete learning and reach the drawing stage. However, when a player is too pessimistic (value = 0.001), the player quickly learns and settles about 1000 games. Even though this small initial value might seem better, the drawback is that the pessimistic player will be satisfied with any move and learn that move when it is better than “loss” only, thus always learned the “draw” move, instead of the “winning” move. We decided to choose initial value as 0.6 since intuitively this value balance out both situations. 
 
In our Q-learning model, we choose the two discounted factors alpha and gamma as 0.9, and 0.95 respectively. We don’t want to choose 1 to complete replace the new state with the best outcome from old states, and hope to have faster convergence, thus the two discounted factor values are decided. 
 
Both Mix-Max algorithm model and Q learning let computer play Gomoku games with human players. However, Mix-max algorithm is not an artificial intelligence model, and assume the player always choose the best possible move. If that is not the real-life situation, mix-max algorithm is not flexible to handle the game well. Q learning, as a reinforcement learning model, does learn the players pattern in the training process, it has better game results overall. 
 
To differentiate this project from typical game projects, we used neural network with Q functions. The above Q function model needs computer to save the Q function score for each state/move pair into a table. With our large potential score computation (1.2688693e+89 combination of moves, and 3.4336838e+30 number of total states), saving the scores in a table is not an efficient process. Deep neural network resolves this problem. We used tensorflow as backend and implemented the model in colab with GPU & TPU, and local computer with a GPU. In this neural network model, we inputted the binary array of scores into tensorflow and building a tensorflow graph. Then we used ReLu as activation function with one hidden layer, and a softmax in the output layer.  Here is a summary of our initial Q learning neural network: 
- Activation function: ReLu
- Loss function: mean square error
- Gradient descent: gradient descent optimizer
- Output layer: softmax
- Input node: 1 dimensional vector
- Hidden layer: one 
- Weight initialization: variance scaling initializer
- Neural Network model version #1: First graph is to go first again random player; the second graph is to go second. 

 
However, this neural network model didn’t return convincing result. After investigating various potential erroneous places, such as tweeking hyperparameters, we realized that the above hyperparameters value is not the primary cause of inaccuracy of the model. The main changes in the improved neural network version #2 is that the input is a 2 dimensional panes instead of 1 dimensional vector, and we added a greedy descent that decreases overtime since a player is not always having the best  move in real life. Here is a summary of the second neural network model built: 
 
- Activation function: ReLu
- Loss function: mean square error
- Gradient descent: gradient descent optimizer
- Output layer: softmax
- Input node: 2 dimensional pane
- Greedy descent: 𝜖  = 0.99
- Hidden layer: one 
- Weight initialization: variance scaling initializer
 
Neural Network Model Version #2: 
First graph is to go first again random player; the second graph is to go second. 



