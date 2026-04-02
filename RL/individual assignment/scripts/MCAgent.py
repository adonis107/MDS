import os, sys
import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle
from pathlib import Path

import text_flappy_bird_gym

class GLIEMCAgent:
    """GLIE Monte-Carlo Control agent for Text Flappy Bird."""

    def __init__(self, n_actions: int = 2, min_epsilon: float = 0.01, max_epsilon: float = 1.0, decay_step: float = 0.0001, lambda_: float = 0.0001):
        self.n_actions = n_actions
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay_step = decay_step    # for linear decay
        self.lambda_ = lambda_          # for exponential decay
        # Q-values and visit counts stored as dicts for sparse state space
        self.Q = defaultdict(float)         # (state, action) -> value
        self.N = defaultdict(int)           # (state, action) -> visit count
        self.epsilon = 1.0
        self.episode_count = 0

    # policy
    def get_action(self, state: tuple) -> int:
        """Select an action using the epsilon-greedy policy w.r.t. current Q."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        # greedy: pick action with highest Q (break ties randomly)
        q_values = [self.Q[(state, a)] for a in range(self.n_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return np.random.choice(best_actions)

    def greedy_action(self, state: tuple) -> int:
        """Purely greedy action (for evaluation)."""
        q_values = [self.Q[(state, a)] for a in range(self.n_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return np.random.choice(best_actions)

    # learning
    def update(self, episode: list[tuple], gamma: float = 1.0):
        """
        Update Q from a complete episode using first-visit MC.

        Parameters
        ----------
        episode : list of (state, action, reward) tuples
        gamma   : discount factor (1.0 for undiscounted)
        """
        G = 0.0
        visited = set()
        # Walk backwards through the episode
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward

            sa = (state, action)
            # First-visit: only update the first time we see (s, a)
            if sa not in visited:
                visited.add(sa)
                self.N[sa] += 1
                # Incremental mean update: Q ← Q + (1/N)(G - Q)
                self.Q[sa] += (1.0 / self.N[sa]) * (G - self.Q[sa])

    def decay_epsilon(self, n_episodes: int, decay_type: str = "linear"):
        """GLIE epsilon decay: linear decay over all training episodes."""
        self.episode_count += 1
        if decay_type == "inverse":
            self.epsilon = max(self.min_epsilon, 1.0 / self.episode_count)
        elif decay_type == "linear":
            self.epsilon = max(self.min_epsilon, self.max_epsilon - self.episode_count * self.decay_step)
        elif decay_type == "exponential":
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(- self.lambda_ * self.episode_count)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / 'results' / 'MC'
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / 'mc_agent.pkl'

    # Single environment: simple-state variant also supports render()
    env = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)
    
    # Create agent
    agent = GLIEMCAgent(n_actions=env.action_space.n, min_epsilon=0.01)
    
    if save_path.exists():
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
            agent.Q = defaultdict(float, data['Q'])
            agent.N = defaultdict(int, data['N'])
            agent.epsilon = data['epsilon']
            agent.episode_count = data['episode_count']
        print(f"Agent loaded from {save_path}")
    else:
        print(f"No saved agent found at {save_path}. Please train the agent first.")
        sys.exit(1)

    obs, _ = env.reset()

    done = False
    total_reward = 0

    while not done:
        state = tuple(obs)
        action = agent.greedy_action(state)

        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        os.system("cls" if os.name == "nt" else "clear")
        sys.stdout.write(env.render())
        time.sleep(0.15)

    print(f"\nGame Over - Total Reward: {total_reward}")
    env.close()