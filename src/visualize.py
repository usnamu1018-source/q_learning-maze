# src/visualize.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from env import GridWorld

ACTION_SYMBOL = {0: '^', 1: '>', 2: 'v', 3: '<'}

def load_q_table(path):
    return pd.read_csv(path).values

def draw_policy(grid, q_table, start, goal, save_path):
    n_rows = len(grid)
    n_cols = len(grid[0])
    policy = [[' ']*n_cols for _ in range(n_rows)]
    for r in range(n_rows):
        for c in range(n_cols):
            if grid[r][c] == 1:
                policy[r][c] = '■'  # obstacle
            elif (r,c) == goal:
                policy[r][c] = 'G'
            elif (r,c) == start:
                policy[r][c] = 'S'
            else:
                idx = r * n_cols + c
                action = int(np.argmax(q_table[idx]))
                policy[r][c] = ACTION_SYMBOL.get(action, '?')

    # 텍스트 출력 파일 저장
    with open(save_path, 'w', encoding='utf-8') as f:
        for row in policy:
            f.write(' '.join(row) + '\n')

    # 간단한 이미지로도 저장
    plt.figure(figsize=(n_cols, n_rows))
    plt.axis('off')
    tb = plt.table(cellText=policy, loc='center', cellLoc='center')
    tb.scale(1, 2)
    plt.savefig(save_path + ".png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 기본 maze와 함께 시도
    grid, start, goal = [
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,1,0],
        [0,1,0,0,0],
        [0,0,0,1,0],
    ], (0,0), (4,4)
    q = load_q_table("results/q_table.csv")
    os.makedirs("results", exist_ok=True)
    draw_policy(grid, q, start, goal, "results/policy.txt")
