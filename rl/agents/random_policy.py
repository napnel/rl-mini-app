from typing import List
import random

import numpy as np
from ray.rllib.utils.annotations import override
from ray.rllib.policy.policy import Policy
from ray.rllib.examples.policy.random_policy import RandomPolicy


class OthelloRandomPolicy(RandomPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_id = self.config.get("__policy_id")

    @override(Policy)
    def compute_actions(self, obs_batch, state_batches=None, **kwargs):
        actions: List[int] = [self._valid_random_action(obs) for obs in obs_batch]
        return actions, [], {}

    def _valid_random_action(self, obs: np.ndarray) -> int:
        if obs.ndim != 2:
            obs = obs.reshape(8, 8)

        valid_actions: List[int] = []
        for row in range(obs.shape[0]):
            for col in range(obs.shape[1]):
                if self._is_valid_action(obs, row, col):
                    valid_actions.append(row * obs.shape[0] + col)

        return random.choice(valid_actions) if len(valid_actions) > 0 else 64

    def _is_valid_action(self, obs: np.ndarray, row: int, col: int) -> bool:
        if row < 0 or row > 7 or col < 0 or col > 7 or obs[row][col] != 0:
            return False

        marks = {"agent_1": 1, "agent_2": -1}
        agent_id = marks[self.policy_id]
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
            if 0 <= r < 8 and 0 <= c < 8 and obs[r][c] == -agent_id:
                r += d_row
                c += d_col
                while 0 <= r < 8 and 0 <= c < 8 and obs[r][c] == -agent_id:
                    r += d_row
                    c += d_col
                if 0 <= r < 8 and 0 <= c < 8 and obs[r][c] == agent_id:
                    return True

        return False
