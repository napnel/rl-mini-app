from enum import IntEnum

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Optional
from ray.rllib.env.multi_agent_env import MultiAgentEnv


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


class OthelloEnv(MultiAgentEnv):
    def __init__(self, config: Dict[str, Any] = None):
        super(OthelloEnv, self).__init__()

        self.agents = {"agent_1": 1, "agent_2": -1}
        self._agent_ids = ["agent_1", "agent_2"]
        self.action_space = gym.spaces.Discrete(8 * 8 + 1)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(1, 8, 8), dtype=np.int8
        )

        self.current_player = "agent_1"
        self.pass_count = {agent: 0 for agent in self.agents.keys()}
        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        # Reset the board
        self.board = np.zeros((8, 8), dtype=np.int8)
        # Initial position
        self.board[3][3] = 1
        self.board[3][4] = -1
        self.board[4][3] = -1
        self.board[4][4] = 1
        self.current_player = "agent_1"
        self.pass_count = {agent: 0 for agent in self.agents.keys()}
        info = {}
        return {self.current_player: self.board.copy().reshape(1, 8, 8)}, info

    def step(self, action_dict: Dict[str, int]):
        obs, reward, terminated, truncated, info = {}, {}, {}, {}, {}
        # assert (
        #     len(action_dict) == 1
        # ), f"Only one agent can take action at a time. {action_dict}"
        for name, action in action_dict.items():
            row, col = divmod(action, 8)
            agent_id = self.agents[name]
            if action == 64:  # pass
                self.pass_count[name] = 1
                terminated = {self.current_player: False, "__all__": False}
                truncated = {self.current_player: False, "__all__": False}

            elif self._is_valid_move(row, col, agent_id):
                self._put_piece(row, col, agent_id)
                terminated = {self.current_player: False, "__all__": False}
                truncated = {self.current_player: False, "__all__": False}

            else:
                terminated = {self.current_player: True, "__all__": True}
                truncated = {self.current_player: True, "__all__": True}

        if np.sum(self.board == 0) == 0:
            terminated = {self.current_player: True, "__all__": True}

        # Calculate the reward as the difference in the number of pieces
        # reward_agent_1 = np.sum(self.board == 1) - np.sum(self.board == -1)
        # reward_agent_2 = np.sum(self.board == -1) - np.sum(self.board == 1)
        if truncated.get("__all__", False):
            reward = {self.current_player: -1}
        else:
            reward = (np.sum(self.board == 1) - np.sum(self.board == -1)) / 64
            reward = reward if self.current_player == "agent_1" else -reward
            reward = {self.current_player: reward}

        # reward = {"agent_1": reward_agent_1, "agent_2": reward_agent_2}

        # No additional info to supply

        self.current_player = (
            "agent_1" if self.current_player == "agent_2" else "agent_2"
        )

        obs = {self.current_player: self.board.copy().reshape(1, 8, 8)}
        info = {self.current_player: {}}

        return obs, reward, terminated, truncated, info

    def _is_valid_move(self, row: int, col: int, agent_id: int) -> bool:
        if row < 0 or row > 7 or col < 0 or col > 7 or self.board[row][col] != 0:
            return False

        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

        for d_row, d_col in directions:
            r, c = row + d_row, col + d_col
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == -agent_id:
                r += d_row
                c += d_col
                while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == -agent_id:
                    r += d_row
                    c += d_col
                if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == agent_id:
                    return True

        return False

    def _put_piece(self, row: int, col: int, agent_id: int) -> None:
        self.board[row][col] = agent_id

        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
        for d_row, d_col in directions:
            r, c = row + d_row, col + d_col
            to_flip = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == -agent_id:
                to_flip.append((r, c))
                r += d_row
                c += d_col

            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == agent_id:
                for r_flip, c_flip in to_flip:
                    self.board[r_flip][c_flip] = agent_id

    def get_valid_moves(self, agent_name: str) -> List[int]:
        agent_id = self.agents[agent_name]
        valid_moves = []
        for row in range(8):
            for col in range(8):
                if self._is_valid_move(row, col, agent_id):
                    valid_moves.append(row * 8 + col)
        return valid_moves

    def render(self):
        # Define the mapping from numbers to characters
        piece_dict = {0: ".", 1: "O", -1: "X"}

        # Create an empty string to store the game board
        board_str = ""

        # Iterate over the rows of the board
        for row in self.board:
            # Convert the row to characters and join them with '|'
            row_str = "|".join(piece_dict[i] for i in row)
            # Add the row string to the game board string
            board_str += row_str + "\n"

        # Print the game board string
        print(board_str)
