import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from rl_snake.agents import BaseAgent
from rl_snake.env import SnakeEnv
from rl_snake.states import BaseStateEncoder


@hydra.main(version_base=None, config_path="../conf", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    env = SnakeEnv(render_mode=cfg.get("render_mode", None), **cfg.env_params)
    agent = BaseAgent.load(cfg.agent_path)
    encoder: BaseStateEncoder = instantiate(cfg.encoder)

    food_collected = []
    survival_steps = []

    for episode in range(cfg.episodes):
        episode_seed = None if cfg.seed is None else cfg.seed + episode
        stats = env.run_episode(
            agent=agent,
            encoder=encoder,
            seed=episode_seed,
            train=False,
        )

        food_collected.append(stats.food_collected)
        survival_steps.append(stats.steps)

        print(
            "Episode "
            f"{episode + 1:4d}/{cfg.episodes} | "
            f"Food Collected: {stats.food_collected:4d} | "
            f"Survival Steps: {stats.steps:6d} | "
            f"Terminated: {stats.terminated!s:>5s} | "
            f"Truncated: {stats.truncated!s:>5s} | "
            f"Iteration Limit Reached: {stats.iteration_limit_reached!s:>5s}",
        )

    if food_collected:
        print(
            "Evaluation complete | "
            f"Episodes: {len(food_collected)} | "
            f"Avg Food Collected: {sum(food_collected) / len(food_collected):6.2f} | "
            f"Avg Survival Steps: {sum(survival_steps) / len(survival_steps):6.2f}",
        )

    env.close()


if __name__ == "__main__":
    main()
