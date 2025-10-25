# src/train.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from env import GridWorld
from q_learning import QLearningAgent

def default_maze():
    grid = [
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,1,0],
        [0,1,0,0,0],
        [0,0,0,1,0],
    ]
    start = (0,0)
    goal = (4,4)
    return grid, start, goal

def train_and_save(results_dir="results",
                   episodes=2000,
                   max_steps=100,
                   lr=0.1, gamma=0.99,
                   epsilon=1.0, min_epsilon=0.01, decay=0.995):
    os.makedirs(results_dir, exist_ok=True)

    grid, start, goal = default_maze()
    env = GridWorld(grid=grid, start=start, goal=goal)
    agent = QLearningAgent(env.n_states, env.n_actions,
                           lr=lr, gamma=gamma,
                           epsilon=epsilon, min_epsilon=min_epsilon, decay=decay)

    rewards_per_episode = []
    success_history = []  # ✅ 성공 여부 기록용 리스트

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        success = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            s_next, reward, done = env.step(action)
            agent.learn(state, action, reward, s_next, done)
            state = s_next
            total_reward += reward

            if done:
                success = 1
                break

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        success_history.append(success)

        if (ep + 1) % 200 == 0:
            print(f"Episode {ep+1}/{episodes} | Reward={total_reward:.2f} | Epsilon={agent.epsilon:.3f}")

    # --- 결과 저장 ---
    q_df = pd.DataFrame(agent.Q)
    q_df.to_csv(os.path.join(results_dir, "q_table.csv"), index=False)

    # ✅ 보상 그래프
    plt.figure()
    plt.plot(rewards_per_episode)
    plt.xlabel("에피소드")
    plt.ylabel("총 보상")
    plt.title("에피소드별 학습 보상 변화")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "reward_curve.png"))
    plt.close()

    # ✅ 성공률 그래프
    success_rate = np.cumsum(success_history) / np.arange(1, len(success_history)+1)
    plt.figure()
    plt.plot(success_rate)
    plt.xlabel("에피소드")
    plt.ylabel("성공률")
    plt.title("시간에 따른 성공률 변화")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "success_rate.png"))
    plt.close()

    # ✅ 요약 정보
    total_success = sum(success_history)
    with open(os.path.join(results_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"총 에피소드,{episodes}\n")
        f.write(f"성공 횟수,{total_success}\n")
        f.write(f"성공률,{total_success/episodes:.3f}\n")

    print(f"[완료] 학습이 끝났습니다. 결과가 '{results_dir}' 폴더에 저장되었습니다.")
    return agent, rewards_per_episode

if __name__ == "__main__":
    train_and_save()

