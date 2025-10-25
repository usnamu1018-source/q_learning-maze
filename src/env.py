# src/env.py
import numpy as np

class GridWorld:
    """
    단순한 Grid 환경.
    0: 빈칸, 1: 장애물, G: 목표(골), S: 시작
    상태는 (row, col) 형태.
    행동: 0=up,1=right,2=down,3=left
    """
    def __init__(self, grid, start, goal, step_reward=-0.04, goal_reward=1.0, obstacle_reward=-1.0):
        self.grid = np.array(grid)
        self.start = start
        self.goal = goal
        self.pos = start
        self.n_rows, self.n_cols = self.grid.shape
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.obstacle_reward = obstacle_reward

    def reset(self):
        self.pos = self.start
        return self._state_to_idx(self.pos)

    def _in_bounds(self, r, c):
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def _state_to_idx(self, pos):
        r, c = pos
        return r * self.n_cols + c

    def _idx_to_state(self, idx):
        r = idx // self.n_cols
        c = idx % self.n_cols
        return (r, c)

    def step(self, action):
        r, c = self.pos
        if action == 0:
            nr, nc = r - 1, c
        elif action == 1:
            nr, nc = r, c + 1
        elif action == 2:
            nr, nc = r + 1, c
        elif action == 3:
            nr, nc = r, c - 1
        else:
            raise ValueError("Invalid action")

        if not self._in_bounds(nr, nc) or self.grid[nr, nc] == 1:
            # 장애물 또는 범위 밖: 벌점, 위치는 변하지 않음
            reward = self.obstacle_reward
            done = False
            next_state = self._state_to_idx(self.pos)
        else:
            self.pos = (nr, nc)
            if self.pos == self.goal:
                reward = self.goal_reward
                done = True
            else:
                reward = self.step_reward
                done = False
            next_state = self._state_to_idx(self.pos)
        return next_state, reward, done

    @property
    def n_states(self):
        return self.n_rows * self.n_cols

    @property
    def n_actions(self):
        return 4
