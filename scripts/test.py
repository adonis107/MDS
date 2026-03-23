import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from rl_snake.agents import BaseAgent
from rl_snake.env import SnakeEnv
from rl_snake.rewards import BaseReward
from rl_snake.states import BaseStateEncoder


@hydra.main(version_base=None, config_path="../conf", config_name="test")
def main(cfg: DictConfig) -> None:
    env = SnakeEnv(render_mode="human", **cfg.env_params)

    agent: BaseAgent = instantiate(cfg.agent)
    encoder: BaseStateEncoder = instantiate(cfg.encoder)
    reward_wrapper: BaseReward = instantiate(cfg.reward)
    stats = env.run_episode(
        agent=agent,
        encoder=encoder,
        reward_wrapper=reward_wrapper,
        seed=cfg.get("seed", None),
        train=False,
    )

    print(
        f"Test episode complete | "
        f"Raw Reward: {stats.raw_return:8.3f} | "
        f"Shaped Reward: {stats.shaped_return:8.3f} | "
        f"Steps: {stats.steps} | "
        f"Iteration Limit Reached: {stats.iteration_limit_reached}",
    )
    env.close()


if __name__ == "__main__":
    main()
