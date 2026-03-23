from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm.auto import tqdm

import wandb
from rl_snake.agents import BaseAgent
from rl_snake.env import SnakeEnv
from rl_snake.rewards import BaseReward
from rl_snake.states import BaseStateEncoder


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    env = SnakeEnv(**cfg.env_params)

    agent: BaseAgent = instantiate(cfg.agent)
    encoder: BaseStateEncoder = instantiate(cfg.encoder)
    reward_wrapper: BaseReward = instantiate(cfg.reward)
    wandb_run = None
    if cfg.wandb:
        wandb_run = wandb.init(
            project="rl-snake",
            name=cfg.run_name,
            config={
                "agent": cfg.agent,
                "encoder": cfg.encoder,
                "reward": cfg.reward,
                "env_params": cfg.env_params,
                "episodes": cfg.episodes,
                "seed": cfg.seed,
            },
        )

    shaped_episode_rewards = []
    raw_episode_rewards = []
    food_collected = []
    survival_steps = []

    progress = tqdm(range(cfg.episodes), desc="Training", unit="episode")
    for episode in progress:
        episode_seed = None if cfg.seed is None else cfg.seed + episode
        stats = env.run_episode(
            agent=agent,
            encoder=encoder,
            reward_wrapper=reward_wrapper,
            seed=episode_seed,
            train=True,
        )

        shaped_episode_rewards.append(stats.shaped_return)
        raw_episode_rewards.append(stats.raw_return)
        food_collected.append(stats.food_collected)
        survival_steps.append(stats.steps)

        metrics = {
            "train/episode": episode + 1,
            "train/raw_reward": stats.raw_return,
            "train/shaped_reward": stats.shaped_return,
            "train/food_collected": stats.food_collected,
            "train/survival_steps": stats.steps,
        }

        if wandb_run is not None:
            wandb_run.log(metrics)

    if cfg.save.enabled:
        save_path = Path("models") / Path(cfg.save.path).with_suffix(".pkl")
        agent.save(save_path)
        progress.write(f"Agent saved to {save_path}")

    progress.close()
    if wandb_run is not None:
        wandb_run.finish()

    env.close()


if __name__ == "__main__":
    main()
