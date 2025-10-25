# src/compare_baseline.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env import GridWorld

def default_maze():
    grid = [
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,1,0],
        [0,1,0,0,0],
        [0,0,0,1,0],
    ]
    start, goal = (0,0), (4,4)
    return grid, start, goal

def random_baseline(results_dir="results", episodes=300, max_steps=100):
    os.makedirs(results_dir, exist_ok=True)
    grid, start, goal = default_maze()
    env = GridWorld(grid, start, goal)
    n_actions = env.n_actions

    rewards = []
    success_history = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        success = 0
        for step in range(max_steps):
            action = np.random.randint(n_actions)  # 무작위 행동
            s_next, reward, done = env.step(action)
            total_reward += reward
            if done:
                success = 1
                break
        rewards.append(total_reward)
        success_history.append(success)

    # 그래프 저장
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("에피소드")
    plt.ylabel("총 보상")
    plt.title("무학습(랜덤 탐색) 보상 변화")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "baseline_reward.png"))
    plt.close()

    success_rate = np.cumsum(success_history) / np.arange(1, len(success_history)+1)
    plt.figure()
    plt.plot(success_rate)
    plt.xlabel("에피소드")
    plt.ylabel("성공률")
    plt.title("무학습(랜덤 탐색) 성공률 변화")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "baseline_success.png"))
    plt.close()

    # 요약
    with open(os.path.join(results_dir, "baseline_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"총 에피소드,{episodes}\n")
        f.write(f"성공 횟수,{sum(success_history)}\n")
        f.write(f"성공률,{sum(success_history)/episodes:.3f}\n")
        f.write(f"평균 보상,{np.mean(rewards):.3f}\n")

    print("[완료] 무학습 대조군 결과 생성 완료.")

if __name__ == "__main__":
    random_baseline()
