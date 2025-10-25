# src/visualize.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.patches as patches

from env import GridWorld

ACTION_SYMBOL = {0: '^', 1: '>', 2: 'v', 3: '<'}
MOVE = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

def load_q_table(path):
    return pd.read_csv(path).values

def best_action(q_table, state):
    return int(np.argmax(q_table[state]))

def draw_policy(grid, q_table, start, goal, save_path):
    n_rows = len(grid)
    n_cols = len(grid[0])
    fig, ax = plt.subplots(figsize=(n_cols, n_rows))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(0, n_cols+1, 1))
    ax.set_yticks(np.arange(0, n_rows+1, 1))
    ax.grid(True)
    ax.invert_yaxis()

    # 그리드 시각화
    for r in range(n_rows):
        for c in range(n_cols):
            if grid[r][c] == 1:
                ax.add_patch(patches.Rectangle((c, r), 1, 1, color='black'))
            elif (r, c) == start:
                ax.add_patch(patches.Rectangle((c, r), 1, 1, color='lightblue'))
                ax.text(c+0.5, r+0.5, "S", ha='center', va='center', fontsize=16, weight='bold')
            elif (r, c) == goal:
                ax.add_patch(patches.Rectangle((c, r), 1, 1, color='lightgreen'))
                ax.text(c+0.5, r+0.5, "G", ha='center', va='center', fontsize=16, weight='bold')
            else:
                s = r * n_cols + c
                a = best_action(q_table, s)
                ax.text(c+0.5, r+0.5, ACTION_SYMBOL[a], ha='center', va='center', fontsize=14)

    plt.title("Learned Policy (Q-Learning)")
    plt.savefig(save_path)
    plt.close()

def visualize_path(grid, q_table, start, goal, save_path):
    """최적 경로를 따라 이동하는 모습을 화살표로 그리기"""
    n_rows = len(grid)
    n_cols = len(grid[0])
    path = [start]
    pos = start
    visited = set()

    for _ in range(100):
        s = pos[0] * n_cols + pos[1]
        a = best_action(q_table, s)
        dr, dc = MOVE[a]
        next_pos = (pos[0] + dr, pos[1] + dc)
        if next_pos == goal:
            path.append(goal)
            break
        if next_pos in visited or not (0 <= next_pos[0] < n_rows and 0 <= next_pos[1] < n_cols) or grid[next_pos[0]][next_pos[1]] == 1:
            break
        path.append(next_pos)
        visited.add(pos)
        pos = next_pos

    # 시각화
    fig, ax = plt.subplots(figsize=(n_cols, n_rows))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(0, n_cols+1, 1))
    ax.set_yticks(np.arange(0, n_rows+1, 1))
    ax.grid(True)
    ax.invert_yaxis()

    # 장애물 및 목표 표시
    for r in range(n_rows):
        for c in range(n_cols):
            if grid[r][c] == 1:
                ax.add_patch(patches.Rectangle((c, r), 1, 1, color='black'))
            elif (r, c) == start:
                ax.add_patch(patches.Rectangle((c, r), 1, 1, color='lightblue'))
            elif (r, c) == goal:
                ax.add_patch(patches.Rectangle((c, r), 1, 1, color='lightgreen'))

    # 경로 시각화
    for i in range(len(path)-1):
        y, x = path[i]
        ny, nx = path[i+1]
        ax.arrow(x+0.5, y+0.5, (nx-x)*0.8, (ny-y)*0.8,
                 head_width=0.2, length_includes_head=True, color='red')

    plt.title("Optimal Path After Learning")
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    grid = [
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,1,0],
        [0,1,0,0,0],
        [0,0,0,1,0],
    ]
    start, goal = (0,0), (4,4)
    q = load_q_table("results/q_table.csv")
    os.makedirs("results", exist_ok=True)
    draw_policy(grid, q, start, goal, "results/policy_visual.png")
    visualize_path(grid, q, start, goal, "results/path_visual.png")

