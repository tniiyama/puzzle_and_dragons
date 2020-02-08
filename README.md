# puzzle_and_dragons

Puzzle and Dragons is a popular Japanese mobile game. The player chooses an orb to move around the game board, available actions being moving the orb up, down, left, or right. When the player releases the orb or runs out of moves, all matches of three or more connected orbs of the same type are erased, and counted as a “combo”. Generally, the goal of the game is to make as many combos as possible, so this project aims to maximize the number of combos achieved given a set number of moves. 

![description](https://imgur.com/a/qMuZl5L)

The highlighted red orb is moved along the black path to its final position in the second picture. All rows or columns with three or more connected orbs are erased, and the third picture is the final board state after one “game”. This game achieved 4 combos (highlighted in red): the row of 4 green orbs at the top, the row of 3 yellows right below it, the row of 3 greens near the bottom, and the column of 3 reds created once the 3 greens are erased. 

Methodology

	I chose to use reinforcement learning to learn how to play the game. Since there is no dataset of board positions and corresponding optimal paths, supervised learning can’t be used to learn the mapping between initial board positions and desired end positions. Reinforcement learning solves this problem by having an agent take an action, observe the reward, and store these state-action-reward pairs to produce its own training data, and using those to learn a policy to maximize reward. 
Q-learning is a type of reinforcement learning where each state-action pair has a corresponding Q-value that represents the estimated value of taking that action in that state. The optimal Q-values are modeled by the Bellman equation Q(s_t,a_t )= r_t+γ  max┬a⁡Q(s_(t+1),a), where s_t and a_t are the state and actions taken at time t, r_t is the reward given for that state-action pair, and γ is the “discount factor”, a number between 0 and 1 that determines how much to value future rewards. A discount factor of 0 means that the agent acts only according to current reward, and a discount factor closer to 1 values future rewards more and more. The true reward values are assigned at a terminal state, and over enough iterations, will backpropagate through the previous Q-values until all the Q-values converge. 
In normal Q-learning, a table stores all of the Q-values corresponding to each state-action pair. However, our game has 30 tiles, with each tile containing one of 6 possible orb colors, and 30 possible positions for the cursor. This results in our game having 30•630 possible states, a number far too large to store a table in memory. Since we can’t iterate through every possible game state-action pair to find the Q-values, we have to estimate the Q-values using a neural network. A convolutional neural network is used, with the input being a game state, with the output being a Q-value for each possible action the agent can take. 
One of the problems with reinforcement learning is the exploration-exploitation problem. Is it more valuable to keep following a policy that giving good rewards, or to take sub-optimal actions with the hope that they’ll lead to bigger rewards in the long run? To address this, DQN uses an epsilon-greedy policy. For each action, the agent has an epsilon% chance of taking a random action, otherwise it will act optimally according to its current policy. Epsilon starts at a value of 1 and decays over the course of training. This way, the game agent explores mostly at the beginning until it finds a good policy, then settles into exploiting the policy it found. 
Two optimizations are used to stabilize learning: a target network and experience replay. Instead of updating the same network you use to generate experiences, a target network is used to generate experiences. The target network’s weights are frozen while training is done on the other network. Periodically, the target network is updated with the weights of the training network. This is used because training on the same network used to generate experiences results in unstable training. Using a target network reduces large fluctuations in weights. 
Since experiences in reinforcement learning are highly correlated, experience replay is used so that the experiences used to train the network aren’t closely related. An experience buffer stores all state-action-reward pairs produced by the game agent. Once the buffer reaches a specified length, a minibatch of random samples is chosen to perform training on. The random sampling of experiences breaks up consecutive experiences so training isn’t dominated by repeated state-action-reward pairs, reducing the likelihood of getting stuck in feedback loops. 
	
The state representation of the game board is a one-hot encoding of 7 5x6 boards: one 5x6 board for each of the 6 orb colors and one for the cursor position. Since the player can only move the orb selected by the cursor, it’s important to include the cursor position in the state representation. For each 5x6 board encoding, there is a 1 in the spot where its corresponding orb color is and a 0 in all other spots. This input layer is fed into a 2D convolutional layer using 32 3x3 filters with stride 1 and uses the ReLU activation function. The first hidden layer uses 64 3x3 filters with stride 1 and a ReLU activation function. This layer is flattened and fed into a fully connected hidden layer with 64 nodes. The output layer is a fully connected layer with an output for each possible action (up, down, left, right, stop). 

Experimental Results

	Experimental training was done over 500,000 game frames, with the experience replay buffer being 100,000 frames large, and the target network updating every 10,000 frames. Experience replay was done with minibatches of 64 experiences. Observed reward at each state is measured by the number of combos created by the current board state. Epsilon was determined by the equation 0.1+0.9e^(-x/25,000), where x is the current frame number. 
 
Graph of epsilon vs. frames

All experiments were performed on the initial board state shown below. The initial cursor position was always set in the top-left corner. 
 
	The results of the DQN, random, and human performance are shown below. The DQN achieved a maximum of 3 combos, which is better than choosing actions randomly, which failed to achieve any combos. However, the DQN still performed considerably worse than a human player (me), which achieved 8 combos on the given board. 

       
		DQN				Random			Human

	Measuring the performance of the model during training can be hard because since epsilon is never less than 0.1, the game agent almost never performs optimally according to its policy during training. Shown below is the graph of the reward at the end of each game during the training data. Since the game engine is programmed to automatically stop after 20 or more moves are made, the 500,000 frames resulted in about 33,000 completed games. The reward function mapped is the number of combos * 10 – moves / 5 (e.g. Making 4 combos in 13 moves gives a reward of 37.4). Acting randomly at the beginning of training when epsilon is high occasionally gives very good results, but the network seems to converge to a sub-optimal policy as training progresses. 
 
Conclusion

	DQN has been proven as a powerful tool that’s been shown to learn how to play Atari games from just the screen pixel data. However, more work is needed on this project to get the DQN agent to perform anywhere close to human levels. I’ve tuned hyperparameters like the learning rate, epsilon, reward function, Bellman equation’s gamma value, and target network update frequency, but I haven’t really experimented with changing the network architecture. The converged result of the network also seems to be very reliant on the initial conditions, so I could experiment with different weight initializers to get better results. I could also experiment with optimizations like prioritized experience replay or double DQN to stabilize training. 

References

https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
https://greentec.github.io/reinforcement-learning-third-en/#weight-initialization
https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287
https://www.freecodecamp.org/news/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682/
http://outlace.com/rlpart3.html
http://cs231n.github.io/neural-networks-3/
http://neuralnetworksanddeeplearning.com/


pad_game.py is a simulation of the mobile game Puzzle and Dragons game engine using Pygame (see https://www.youtube.com/watch?v=zCX-Tz8KpXE for the actual game)

pad_dqn.py uses a Deep-Q Network (DQN) to maximize the number of combos made. Uses experience replay, 
and trains action values using the Bellman equation

orbs.png and background.png are sprite files
