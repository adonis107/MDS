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


class SarsaLambdaAgent:
    """SARSA(lambda) agent with accumulating eligibility traces for Text Flappy Bird."""

    def __init__(
        self,
        n_actions: int = 2,
        alpha: float = 0.1,
        gamma: float = 1.0,
        lambd: float = 0.8,
        min_epsilon: float = 0.01,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd
        self.min_epsilon = min_epsilon

        self.Q = defaultdict(float)   # (state, action) -> value
        self.epsilon = 1.0
        self.episode_count = 0

    # policy
    def get_action(self, state: tuple) -> int:
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = [self.Q[(state, a)] for a in range(self.n_actions)]
        max_q = max(q_values)
        best = [a for a, q in enumerate(q_values) if q == max_q]
        return np.random.choice(best)

    def greedy_action(self, state: tuple) -> int:
        """Purely greedy action (for evaluation)."""
        q_values = [self.Q[(state, a)] for a in range(self.n_actions)]
        max_q = max(q_values)
        best = [a for a, q in enumerate(q_values) if q == max_q]
        return np.random.choice(best)

    # learning (one full episode)
    def run_episode(self, env) -> float:
        """
        Play one episode with SARSA(lambda) updates (accumulating traces).

        Returns the total undiscounted reward collected in the episode.
        """
        # Reset eligibility traces at the start of every episode
        E = defaultdict(float)

        obs, _ = env.reset()
        S = tuple(obs)
        A = self.get_action(S)
        total_reward = 0.0

        done = False
        while not done:
            obs_next, reward, done, _, info = env.step(A)
            total_reward += reward
            S_next = tuple(obs_next)

            if done:
                # Terminal step: Q(terminal, ·) = 0, so TD target is just R
                delta = reward - self.Q[(S, A)]
            else:
                A_next = self.get_action(S_next)
                delta = reward + self.gamma * self.Q[(S_next, A_next)] - self.Q[(S, A)]

            # Accumulating trace for the current (S, A)
            E[(S, A)] += 1.0

            # Update all Q-values that have non-zero traces
            for sa in list(E.keys()):
                self.Q[sa] += self.alpha * delta * E[sa]
                E[sa] *= self.gamma * self.lambd
                # Remove negligible traces to keep the dict small
                if E[sa] < 1e-6:
                    del E[sa]

            if not done:
                S = S_next
                A = A_next

        return total_reward

    # epsilon decay
    def decay_epsilon(self):
        """GLIE epsilon decay: epsilon = 1/k, floored at min_epsilon."""
        self.episode_count += 1
        self.epsilon = max(self.min_epsilon, 1.0 / self.episode_count)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / 'results' / 'SARSA'
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / 'sarsa_lambda_agent.pkl'

    # Play one episode with the screen renderer
    demo_env = gym.make('TextFlappyBird-screen-v0', height=15, width=20, pipe_gap=4)
    state_env = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)

    # Create agent
    agent = SarsaLambdaAgent(n_actions=state_env.action_space.n)

    if save_path.exists():
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
            agent.Q = defaultdict(float, data['Q'])
            agent.epsilon = data['epsilon']
            agent.episode_count = data['episode_count']
            agent.alpha = data['alpha']
            agent.gamma = data['gamma']
            agent.lambd = data['lambd']
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