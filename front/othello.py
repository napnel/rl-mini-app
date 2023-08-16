import gymnasium as gym
import numpy as np
from typing import Any, Dict, List, Optional


class OthelloEnv(gym.Env):
    def __init__(self, config: Dict[str, Any]):
        super(OthelloEnv, self).__init__()

        self.agents = {"agent_1": 1, "agent_2": -1}
        self._agent_ids = ["agent_1", "agent_2"]
        self.action_space = gym.spaces.Discrete(8 * 8 + 1)
        self.obs_shape = config.get("obs_shape", (64,))
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=self.obs_shape, dtype=np.int8
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
        return {self.current_player: self.board.copy().reshape(*self.obs_shape)}, info

    def step(self, action_dict: Dict[str, int]):
        obs, reward, terminated, truncated, info = {}, {}, {}, {}, {}
        reward = 0
        terminated, truncated = False, False
        for name, action in action_dict.items():
            row, col = divmod(action, 8)
            if action == 64:  # pass
                if self.pass_count[name] == 1:
                    truncated = True
                else:
                    self.pass_count[name] = 1

            elif self._is_valid_move(row, col, name):
                self._put_piece(row, col, name)
                self.pass_count[name] = 0

            else:
                truncated = True

        if np.sum(self.board == 0) == 0:
            terminated = True

        terminated = {self.current_player: terminated, "__all__": terminated}
        truncated = {self.current_player: truncated, "__all__": truncated}

        reward = self._calculate_reward(
            action_dict.get(self.current_player, 64), terminated, truncated
        )
        reward = {self.current_player: reward}

        self.current_player = (
            "agent_1" if self.current_player == "agent_2" else "agent_2"
        )

        obs = {self.current_player: self.board.copy().reshape(*self.obs_shape)}
        info = {self.current_player: {}}

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, action: int, terminated: bool, truncated: bool):
        # Reward for the game rule
        win_reward = 0
        if np.sum(self.board == 0) == 0 or terminated:
            player_id = self.agents[self.current_player]
            opponent_id = self.agents[self.opponent]
            win_reward = (
                10
                if np.sum(self.board == player_id) > np.sum(self.board == opponent_id)
                else -10
            )

        # if true, then the taken invalid action or two consecutive passes
        rule_break_penalty = 0 if not truncated else -10

        reward = win_reward + rule_break_penalty

        # Reward against the agent's action
        if not terminated and not truncated and action != 64:
            corner_reward = 0
            row, col = divmod(action, 8)
            if (row == 0 or row == 7) and (col == 0 or col == 7):
                corner_reward = 5

            edge_reward = 0
            if row == 0 or row == 7 or col == 0 or col == 7:
                edge_reward = 1

            restricting_opponent_actions_reward = (
                len(self.get_valid_moves(self.opponent)) * 0.2
            )

            reward += corner_reward + edge_reward + restricting_opponent_actions_reward

        return reward

    def _is_valid_move(self, row: int, col: int, agent_name: str) -> bool:
        if row < 0 or row > 7 or col < 0 or col > 7 or self.board[row][col] != 0:
            return False

        agent_id = self.agents[agent_name]
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

    def _put_piece(self, row: int, col: int, agent_name: str) -> None:
        agent_id = self.agents[agent_name]
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
        valid_moves = []
        for row in range(8):
            for col in range(8):
                if self._is_valid_move(row, col, agent_name):
                    valid_moves.append(row * 8 + col)
        return valid_moves

    @property
    def opponent(self) -> str:
        return "agent_2" if self.current_player == "agent_1" else "agent_1"

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "agent_1": np.sum(self.board == 1),
            "agent_2": np.sum(self.board == -1),
        }

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
