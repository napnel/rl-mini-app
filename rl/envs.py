from enum import Enum, IntEnum

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Mark(IntEnum):
    EMPTY = 0
    PLAYER_1 = 1
    PLAYER_2 = 2


class TicTacToeEnv(gym.Env):
    def __init__(self, env_config=None):
        super(TicTacToeEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0, high=2, shape=(9,), dtype=int)

        # Initialize state
        self.state = [Mark.EMPTY for _ in range(9)]

        self.current_player = Mark.PLAYER_1

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        self.state = [Mark.EMPTY for _ in range(9)]
        self.current_player = Mark.PLAYER_1
        return self.state, {}

    def step(self, action):
        truncated = False
        # Execute one time step within the environment
        if self.state[action] == Mark.EMPTY:
            self.state[action] = self.current_player

            terminated = self.check_game_status()

            if terminated:
                reward = 1 if self.current_player == Mark.PLAYER_1 else -1
            else:
                reward = 0
                # Switch the current player
                self.current_player = (
                    Mark.PLAYER_1
                    if self.current_player == Mark.PLAYER_2
                    else Mark.PLAYER_2
                )
        else:
            terminated = True
            reward = -1

        return self.state, reward, terminated, truncated, {}

    def check_game_status(self):
        # Check the game status and return the game is finished or not
        lines = [
            self.state[0:3],
            self.state[3:6],
            self.state[6:9],  # horizontal lines
            self.state[0:7:3],
            self.state[1:8:3],
            self.state[2:9:3],  # vertical lines
            self.state[0:9:4],
            self.state[2:7:2],  # diagonal lines
        ]
        if [1, 1, 1] in lines or [2, 2, 2] in lines:
            return True  # a player wins
        elif 0 not in self.state:
            return True  # draw
        else:
            return False  # game continues


# class OthelloEnv(gym.Env):
#     def __init__(self):
#         super(OthelloEnv, self).__init__()

#         # Define action and observation space
#         self.action_space = spaces.Discrete(64)  # 8x8 board
#         self.observation_space = spaces.Box(
#             low=0,
#             high=2,
#             shape=(
#                 8,
#                 8,
#             ),
#             dtype=int,
#         )

#         # Initialize state
#         self.state = np.zeros((8, 8), dtype=int)
#         self.state[3, 3] = self.state[4, 4] = 1  # Player 1 initial stones
#         self.state[3, 4] = self.state[4, 3] = 2  # Player 2 initial stones

#         # Track whose turn it is, 1 for player 1, 2 for player 2
#         self.current_player = 1

#     def step(self, action):
#         truncated = False
#         # Execute one time step within the environment
#         if self.is_valid_action(action):
#             self.state = self.get_next_state(self.state, action, self.current_player)
#             terminated = self.is_game_over()
#             if terminated:
#                 # Game is over, player with more stones wins
#                 reward = 1 if np.sum(self.state == self.current_player) > 32 else -1
#             else:
#                 # Game continues
#                 reward = 0
#                 self.current_player = 1 if self.current_player == 2 else 2
#         else:
#             # Invalid action, game is over and the player loses
#             terminated = True
#             reward = -1

#         return self.state, reward, terminated, truncated, {}

#     def reset(self):
#         # Reset the state of the environment to an initial state
#         self.state = np.zeros((8, 8), dtype=int)
#         self.state[3, 3] = self.state[4, 4] = 1
#         self.state[3, 4] = self.state[4, 3] = 2
#         self.current_player = 1
#         return self.state, {}

#     def is_valid_action(self, action):
#         # Check if an action is valid or not for the current state and player
#         # You will need to implement the rules of Othello here
#         pass

#     def get_next_state(self, state, action, player):
#         # Get the next state by applying the action for the player
#         # You will need to implement the rules of Othello here
#         pass

#     def is_game_over(self):
#         # Check if the game is over or not
#         # Game is over if the board is full, or if a player has no valid actions
#         pass
