from enum import IntEnum

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
        self.train = env_config.get("train", True)

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        self.state = [Mark.EMPTY for _ in range(9)]
        # Todo: Randomly select the first player

        self.current_player = Mark.PLAYER_1
        if self.train:
            action = np.random.choice(np.where(np.array(self.state) == Mark.EMPTY)[0])
            self.step(action)

        return self.state, {}

    def step(self, action):
        truncated = False
        # Execute one time step within the environment
        if self.state[action] == Mark.EMPTY:
            self.state[action] = self.current_player

            terminated = self.check_game_status()
            if terminated == Mark.PLAYER_1:  # Player
                reward = -1
            elif terminated == Mark.PLAYER_2:  # Agent
                reward = 1
            elif terminated == -1:  # Draw
                reward = 0
            else:
                reward = 0

                # Switch the current player
                self.current_player = (
                    Mark.PLAYER_1
                    if self.current_player == Mark.PLAYER_2
                    else Mark.PLAYER_2
                )

            terminated = bool(abs(terminated) > 0)

        else:
            terminated = True
            reward = -1

        if self.train and self.current_player == Mark.PLAYER_2 and not terminated:
            action = np.random.choice(np.where(np.array(self.state) == Mark.EMPTY)[0])
            return self.step(action)

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
        if [Mark.PLAYER_1, Mark.PLAYER_1, Mark.PLAYER_1] in lines:
            return Mark.PLAYER_1

        elif [Mark.PLAYER_2, Mark.PLAYER_2, Mark.PLAYER_2] in lines:
            return Mark.PLAYER_2

        elif Mark.EMPTY not in self.state:
            return -1

        else:
            return 0
