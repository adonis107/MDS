import pickle

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from rl_snake.agents import BaseAgent
from rl_snake.env import SnakeEnv
from rl_snake.rewards import BaseReward
from rl_snake.states import BaseStateEncoder


@hydra.main(version_base=None, config_path="../conf", config_name="test")
def main(cfg: DictConfig) -> None:
    checkpoint_metadata = OmegaConf.create({})
    if cfg.agent_path is not None:
        agent = BaseAgent.load(cfg.agent_path)
        with open(cfg.agent_path, "rb") as f:
            checkpoint_payload = pickle.load(f)
        checkpoint_metadata = OmegaConf.create(checkpoint_payload.get("metadata", {}))
    else:
        agent: BaseAgent = instantiate(cfg.agent)

    env_params = cfg.env_params
    if cfg.use_checkpoint_env and checkpoint_metadata.get("env_params") is not None:
        env_params = checkpoint_metadata.env_params

    encoder_cfg = cfg.encoder
    if cfg.use_checkpoint_encoder and checkpoint_metadata.get("encoder") is not None:
        encoder_cfg = checkpoint_metadata.encoder

    reward_cfg = cfg.reward
    if cfg.use_checkpoint_reward and checkpoint_metadata.get("reward") is not None:
        reward_cfg = checkpoint_metadata.reward

    env = SnakeEnv(render_mode=cfg.render_mode, **env_params)
    encoder: BaseStateEncoder = instantiate(encoder_cfg)
    reward_wrapper: BaseReward = instantiate(reward_cfg)
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
