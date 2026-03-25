"""Train a DQN agent on the Snake environment.

Run via:
    python -m rl_snake.train [args]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import wandb
from rl_snake.agent import CNNDQNAgent, DQNAgent, FrameStack, get_grid_state, get_state
from rl_snake.env import SnakeEnv


def _build_run_name(args: argparse.Namespace) -> str:
    """Auto-generate a descriptive run name from the config when none is given."""
    parts = [args.agent_type]
    if args.double_dqn:
        parts.append("double")
    if args.dueling:
        parts.append("dueling")
    if args.n_frames > 1:
        parts.append(f"fs{args.n_frames}")
    if args.n_dynamic_obstacles > 0 or args.n_silver > 0 or args.n_poison > 0:
        parts.append("hard")
    return "_".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN on Snake")

    # Environment — grid
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=500)

    # Environment — rewards
    p.add_argument("--gold-reward", type=float, default=10.0)
    p.add_argument("--silver-reward", type=float, default=5.0)
    p.add_argument("--poison-reward", type=float, default=0.0)
    p.add_argument("--death-reward", type=float, default=-10.0)
    p.add_argument("--step-reward", type=float, default=-0.01,
                   help="Per-step reward (negative penalises stalling)")
    p.add_argument(
        "--distance-reward-scale",
        type=float,
        default=0.0,
        help="Scale for potential-based Manhattan distance reward shaping (0 = disabled)",
    )

    # Environment — food
    p.add_argument("--n-gold", type=int, default=1)
    p.add_argument("--n-silver", type=int, default=0)
    p.add_argument("--n-poison", type=int, default=0)
    p.add_argument("--poison-shrink", type=int, default=2, help="Body segments removed when poison is eaten")

    # Environment — obstacles
    p.add_argument(
        "--n-dynamic-obstacles",
        type=int,
        default=0,
        help="Number of moving wall segments (size 3)",
    )

    # Agent
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.01)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-capacity", type=int, default=100_000)
    p.add_argument("--target-update", type=int, default=1_000)
    p.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
        help="Max gradient norm for clipping (0 = disabled)",
    )
    p.add_argument(
        "--target-tau",
        type=float,
        default=None,
        help="Soft target update rate τ (None = hard update every --target-update steps; "
             "recommended: 0.005 for soft updates)",
    )

    # Architecture
    p.add_argument(
        "--agent-type",
        type=str,
        default="mlp",
        choices=["mlp", "cnn"],
        help="mlp: 15-feature DQN;  cnn: full-grid CNN-DQN",
    )
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "rmsprop"])
    p.add_argument("--n-frames", type=int, default=1, help="Frames to stack as input (1 = no stacking)")
    p.add_argument("--double-dqn", action="store_true", default=False, help="Enable Double DQN")
    p.add_argument(
        "--dueling",
        action="store_true",
        default=False,
        help="Enable dueling network architecture",
    )

    # Training
    p.add_argument("--episodes", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument(
        "--save-video",
        action="store_true",
        default=False,
        help="Save a GIF of the agent playing alongside each checkpoint",
    )

    # W&B
    p.add_argument("--wandb-project", type=str, default="rl-snake")
    p.add_argument("--run-name", type=str, default=None, help="W&B run name (auto-generated if not set)")

    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    run_name = args.run_name or _build_run_name(args)

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
    )

    env = SnakeEnv(
        height=args.height,
        width=args.width,
        seed=args.seed,
        gold_reward=args.gold_reward,
        silver_reward=args.silver_reward,
        poison_reward=args.poison_reward,
        death_reward=args.death_reward,
        step_reward=args.step_reward,
        distance_reward_scale=args.distance_reward_scale,
        max_steps=args.max_steps,
        n_gold=args.n_gold,
        n_silver=args.n_silver,
        n_poison=args.n_poison,
        poison_shrink=args.poison_shrink,
        n_dynamic_obstacles=args.n_dynamic_obstacles,
    )

    if args.agent_type == "cnn":
        agent = CNNDQNAgent(
            height=args.height,
            width=args.width,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            batch_size=args.batch_size,
            buffer_capacity=args.buffer_capacity,
            target_update_freq=args.target_update,
            optimizer_name=args.optimizer,
            n_frames=args.n_frames,
            double_dqn=args.double_dqn,
            dueling=args.dueling,
            grad_clip=args.grad_clip or None,
            target_tau=args.target_tau,
        )
        extract_state = get_grid_state
    else:
        agent = DQNAgent(
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            batch_size=args.batch_size,
            buffer_capacity=args.buffer_capacity,
            target_update_freq=args.target_update,
            optimizer_name=args.optimizer,
            n_frames=args.n_frames,
            double_dqn=args.double_dqn,
            dueling=args.dueling,
            grad_clip=args.grad_clip or None,
            target_tau=args.target_tau,
        )
        extract_state = get_state

    frame_stack = FrameStack(args.n_frames, extract_state)

    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(1, args.episodes + 1):
        env.reset()
        state = frame_stack.reset(env)
        total_reward = 0.0
        episode_losses: list[float] = []

        while not env.done:
            action = agent.select_action(state)
            result = env.step(action)
            next_state = frame_stack.step(env)

            agent.store_transition(state, action, result.reward, next_state, result.done)
            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += result.reward

        wandb.log(
            {
                "episode": episode,
                "length": env.length,
                "total_reward": total_reward,
                "steps": env.steps,
                "epsilon": agent.epsilon,
                "loss": float(np.mean(episode_losses)) if episode_losses else None,
            },
        )

        if episode % args.save_every == 0:
            ckpt_path = save_dir / f"ep{episode:05d}.pt"
            agent.save(str(ckpt_path))
            wandb.save(str(ckpt_path))
            print(
                f"[{episode:>{len(str(args.episodes))}}/{args.episodes}]  "
                f"length={env.length}  ε={agent.epsilon:.3f}  saved → {ckpt_path}",
            )

            if args.save_video:
                _save_checkpoint_video(
                    save_dir / f"ep{episode:05d}.gif",
                    env,
                    agent,
                    extract_state,
                    args.n_frames,
                )

    final_path = save_dir / "final.pt"
    agent.save(str(final_path))
    wandb.save(str(final_path))

    if args.save_video:
        _save_checkpoint_video(
            save_dir / "final.gif",
            env,
            agent,
            extract_state,
            args.n_frames,
        )

    wandb.finish()
    print(f"Training complete. Final checkpoint → {final_path}")


def _save_checkpoint_video(
    path: Path,
    env: SnakeEnv,
    agent,
    extract_state,
    n_frames: int,
) -> None:
    try:
        from rl_snake.visuals import save_video

        save_video(str(path), env, agent, extract_state, n_frames=n_frames)
        print(f"  video → {path}")
    except Exception as exc:
        print(f"  video save skipped: {exc}")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
