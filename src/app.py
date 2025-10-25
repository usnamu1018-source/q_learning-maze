# src/app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from env import GridWorld
from q_learning import QLearningAgent

st.set_page_config(page_title="AI 길찾기 교실", page_icon="🧠", layout="wide")

st.title("🧠 Q-러닝 기반 AI 길찾기 교실")
st.write("""
이 앱은 **Q-러닝(Q-learning)** 알고리즘을 이용해 에이전트가 미로 속에서 스스로 **최적 경로를 학습**하는 과정을 시각화합니다.  
실험 파라미터를 바꾸며 직접 학습 효과를 관찰해보세요!
""")

# ----- Sidebar -----
st.sidebar.header("⚙️ 실험 설정")

episodes = st.sidebar.slider("학습 에피소드 수", 100, 3000, 1000, step=100)
lr = st.sidebar.slider("학습률 (Learning Rate)", 0.01, 1.0, 0.1, step=0.01)
gamma = st.sidebar.slider("감가율 (Discount Factor)", 0.5, 0.99, 0.9, step=0.01)
epsilon = st.sidebar.slider("탐험 확률 초기값 (Epsilon)", 0.0, 1.0, 1.0, step=0.05)
decay = st.sidebar.slider("Epsilon 감소율", 0.90, 0.999, 0.995, step=0.001)

# ----- Maze 설정 -----
st.sidebar.subheader("🏁 미로 선택")
maze_type = st.sidebar.selectbox("환경 선택", ["기본 미로", "좁은 통로형", "장애물 밀집형"])

if maze_type == "기본 미로":
    grid = [
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,1,0],
        [0,1,0,0,0],
        [0,0,0,1,0],
    ]
elif maze_type == "좁은 통로형":
    grid = [
        [0,0,1,0,0],
        [1,0,1,0,1],
        [0,0,0,0,0],
        [1,0,1,0,1],
        [0,0,1,0,0],
    ]
else:
    grid = [
        [0,1,1,1,0],
        [0,1,0,1,0],
        [0,0,0,0,0],
        [0,1,0,1,0],
        [0,1,1,1,0],
    ]

start, goal = (0, 0), (4, 4)

st.sidebar.markdown("---")
run_training = st.sidebar.button("🚀 학습 시작")

# ----- 학습 및 시각화 -----
def visualize_policy(agent, grid):
    n_rows, n_cols = len(grid), len(grid[0])
    fig, ax = plt.subplots()
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(0, n_cols+1))
    ax.set_yticks(np.arange(0, n_rows+1))
    ax.grid(True)
    ax.invert_yaxis()

    ACTION_SYMBOL = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    for r in range(n_rows):
        for c in range(n_cols):
            if grid[r][c] == 1:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color='black'))
            elif (r,c) == start:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color='lightblue'))
                ax.text(c+0.5, r+0.5, "S", ha='center', va='center', fontsize=16)
            elif (r,c) == goal:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color='lightgreen'))
                ax.text(c+0.5, r+0.5, "G", ha='center', va='center', fontsize=16)
            else:
                idx = r * n_cols + c
                a = int(np.argmax(agent.Q[idx]))
                ax.text(c+0.5, r+0.5, ACTION_SYMBOL[a], ha='center', va='center', fontsize=14)
    st.pyplot(fig)

if run_training:
    st.subheader("📚 학습 진행 중...")
    env = GridWorld(grid, start, goal)
    agent = QLearningAgent(env.n_states, env.n_actions, lr=lr, gamma=gamma,
                           epsilon=epsilon, decay=decay)
    rewards = []
    success = 0
    progress = st.progress(0)
    chart = st.empty()

    for ep in range(episodes):
        s = env.reset()
        total_reward = 0
        for _ in range(100):
            a = agent.choose_action(s)
            s_next, r, done = env.step(a)
            agent.learn(s, a, r, s_next, done)
            s = s_next
            total_reward += r
            if done:
                success += 1
                break
        agent.decay_epsilon()
        rewards.append(total_reward)
        if ep % (episodes//100) == 0:
            progress.progress(ep / episodes)
    progress.progress(1.0)

    # 결과 시각화
    st.success("✅ 학습 완료!")
    avg_reward = np.mean(rewards[-100:])
    success_rate = success / episodes

    st.metric("최근 100회 평균 보상", f"{avg_reward:.3f}")
    st.metric("전체 성공률", f"{success_rate:.2%}")

    st.subheader("📈 보상 변화 그래프")
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_xlabel("에피소드")
    ax.set_ylabel("총 보상")
    ax.set_title("에피소드별 보상 변화")
    st.pyplot(fig)

    st.subheader("🧭 학습된 정책 시각화")
    visualize_policy(agent, grid)

    st.balloons()
else:
    st.info("왼쪽 설정을 조정하고 **[🚀 학습 시작]** 버튼을 눌러보세요!")

