from dataclasses import dataclass

import gymnasium as gym

from rl_snake.agents import BaseAgent
from rl_snake.rewards import BaseReward
from rl_snake.states import BaseStateEncoder


@dataclass
class EpisodeStats:
    raw_return: float
    shaped_return: float
    steps: int
    max_length: int
    food_collected: int
    terminated: bool
    truncated: bool
    iteration_limit_reached: bool = False


class SnakeEnv:
    """Thin environment wrapper to centralize Snake-v1 creation and lifecycle."""

    def __init__(self, render_mode: str | None = None, **options):
        self.env = gym.make("Snake-v1", render_mode=render_mode, **options)

    def close(self) -> None:
        self.env.close()

    def _snake_length(self) -> int:
        snake = self.env.unwrapped.snake
        return 1 + len(snake.body)

    def _food_collected(self) -> int:
        snake = self.env.unwrapped.snake
        return int(snake.score)

    def run_episode(
        self,
        agent: BaseAgent,
        encoder: BaseStateEncoder,
        reward_wrapper: BaseReward | None = None,
        seed: int | None = None,
        *,
        train: bool = True,
    ) -> "EpisodeStats":
        obs, info = self.env.reset(seed=seed)
        if reward_wrapper is not None:
            reward_wrapper.reset(info)
        state = encoder.encode(obs, info)

        raw_return = 0.0
        shaped_return = 0.0
        steps = 0
        max_length = self._snake_length()
        food_collected = self._food_collected()
        terminated = False
        truncated = False

        done = False
        iteration_limit_reached = False
        while not done:
            action = agent.choose_action(state)
            next_obs, raw_reward, terminated, truncated, next_info = self.env.step(action)
            next_state = encoder.encode(next_obs, next_info)
            # Raw reward measures true environment performance.
            # Shaped reward is only the optimization signal.
            if reward_wrapper is None:
                shaped_reward = float(raw_reward)
            else:
                shaped_reward = reward_wrapper.compute(
                    state=state,
                    action=action,
                    next_state=next_state,
                    raw_reward=raw_reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=next_info,
                )

            done = terminated or truncated

            if train:
                agent.update(
                    state=state,
                    action=action,
                    reward=shaped_reward,
                    next_state=next_state,
                    done=done,
                )

            state = next_state
            raw_return += float(raw_reward)
            shaped_return += float(shaped_reward)
            steps += 1
            max_length = max(max_length, self._snake_length())
            food_collected = self._food_collected()

        if not done:
            iteration_limit_reached = True

        return EpisodeStats(
            raw_return=raw_return,
            shaped_return=shaped_return,
            steps=steps,
            max_length=max_length,
            food_collected=food_collected,
            terminated=terminated,
            truncated=truncated,
            iteration_limit_reached=iteration_limit_reached,
        )
