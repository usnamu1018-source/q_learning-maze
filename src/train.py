# src/train.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from env import GridWorld
from q_learning import QLearningAgent

def default_maze():
    # 0 empty, 1 obstacle
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
    success_count = 0

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        for step in range(max_steps):
            action = agent.choose_action(state)
            s_next, reward, done = env.step(action)
            agent.learn(state, action, reward, s_next, done)
            state = s_next
            total_reward += reward
            if done:
                success_count += 1
                break
        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        if (ep+1) % 200 == 0:
            print(f"Episode {ep+1}/{episodes}, total_reward={total_reward:.2f}, epsilon={agent.epsilon:.3f}")

    # 저장: Q-table CSV
    q_df = pd.DataFrame(agent.Q)
    q_df.to_csv(os.path.join(results_dir, "q_table.csv"), index=False)

    # 보상 그래프 저장
    plt.figure()
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward per Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "reward_curve.png"))
    plt.close()

    # 성공률 저장
    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write(f"episodes,{episodes}\n")
        f.write(f"successful_episodes,{success_count}\n")
        f.write(f"success_rate,{success_count/episodes:.4f}\n")

    print("Training finished. Results saved to", results_dir)
    return agent, rewards_per_episode

if __name__ == "__main__":
    train_and_save()
