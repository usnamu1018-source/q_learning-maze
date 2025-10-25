# src/q_learning.py
import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=1.0, min_epsilon=0.01, decay=0.995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            return int(np.argmax(self.Q[state]))

    def learn(self, s, a, r, s_next, done):
        target = r
        if not done:
            target = r + self.gamma * np.max(self.Q[s_next])
        td_error = target - self.Q[s, a]
        self.Q[s, a] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
