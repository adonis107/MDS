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

    def __init__(self, n_actions: int = 2, min_epsilon: float = 0.01):
        self.n_actions = n_actions
        self.min_epsilon = min_epsilon
        # Q-values and visit counts stored as dicts for sparse state space
        self.Q = defaultdict(float)        # (state, action) -> value
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

    def decay_epsilon(self):
        """GLIE epsilon decay: epsilon = 1/k, floored at min_epsilon."""
        self.episode_count += 1
        self.epsilon = max(self.min_epsilon, 1.0 / self.episode_count)

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / 'results' / 'MC'
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / 'mc_agent.pkl'

    # Play one episode with the screen renderer
    demo_env = gym.make('TextFlappyBird-screen-v0', height=15, width=20, pipe_gap=4)
    # We also need the simple-state env running in parallel to get (x, y) for our Q-table
    state_env = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)
    
    agent = GLIEMCAgent(n_actions=state_env.action_space.n, min_epsilon=0.01)
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

    obs_screen, _ = demo_env.reset()
    obs_state, _ = state_env.reset()

    done = False
    total_reward = 0

    while not done:
        state = tuple(obs_state)
        action = agent.greedy_action(state)

        obs_screen, reward, done, _, info = demo_env.step(action)
        obs_state, _, _, _, _ = state_env.step(action)
        total_reward += reward

        os.system("cls" if os.name == "nt" else "clear")
        sys.stdout.write(demo_env.render())
        time.sleep(0.15)

    print(f"\nGame Over - Total Reward: {total_reward}")
    demo_env.close()
    state_env.close()