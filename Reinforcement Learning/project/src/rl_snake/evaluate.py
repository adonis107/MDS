from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np

import wandb
from rl_snake.agent import CNNDQNAgent, DQNAgent, FrameStack, get_grid_state, get_state, get_window_state
from rl_snake.env import SnakeEnv

# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------


def get_test_configs(height: int, width: int) -> list[dict]:
    base = dict(height=height, width=width, max_steps=500)
    return [
        dict(
            name="baseline",
            description="1 gold, no obstacles",
            env_kwargs={**base, "n_gold": 1},
        ),
        dict(
            name="more_food",
            description="2 gold + 1 silver",
            env_kwargs={**base, "n_gold": 2, "n_silver": 1},
        ),
        dict(
            name="poison",
            description="1 gold + 1 poison",
            env_kwargs={**base, "n_gold": 1, "n_poison": 1},
        ),
        dict(
            name="rand_obstacles",
            description="1 gold + 3 rand obstacles",
            env_kwargs={**base, "n_gold": 1, "n_rand_obstacles": 3},
        ),
        dict(
            name="moving_obstacles",
            description="1 gold + 2 dynamic obstacles",
            env_kwargs={**base, "n_gold": 1, "n_dynamic_obstacles": 2},
        ),
        dict(
            name="full_hard",
            description="2 gold + 1 silver + 1 poison + 3 rand obs + 2 dynamic obs",
            env_kwargs={
                **base,
                "n_gold": 2,
                "n_silver": 1,
                "n_poison": 1,
                "n_rand_obstacles": 3,
                "n_dynamic_obstacles": 2,
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------


def load_agent(
    checkpoint: str,
    agent_type: str,
    height: int,
    width: int,
    window_size: int = 11,
    device: str | None = None,
) -> tuple[DQNAgent | CNNDQNAgent, object]:
    """Load a trained agent. Returns (agent, state_fn)."""
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu")

    if agent_type == "mlp":
        state_dim = ckpt["q_net"]["trunk.0.weight"].shape[1]
        n_frames = state_dim // DQNAgent.BASE_STATE_DIM
        agent = DQNAgent(
            n_frames=n_frames,
            epsilon_start=0.0,
            epsilon_end=0.0,
            device=device,
        )
        state_fn = get_state

    elif agent_type == "cnn":
        in_channels = ckpt["q_net"]["conv.0.weight"].shape[1]
        n_frames = in_channels // CNNDQNAgent.BASE_CHANNELS
        agent = CNNDQNAgent(
            height=height,
            width=width,
            n_frames=n_frames,
            epsilon_start=0.0,
            epsilon_end=0.0,
            device=device,
        )
        state_fn = get_grid_state

    elif agent_type == "window-cnn":
        in_channels = ckpt["q_net"]["conv.0.weight"].shape[1]
        n_frames = in_channels // CNNDQNAgent.BASE_CHANNELS
        half_size = window_size // 2
        agent = CNNDQNAgent(
            height=window_size,
            width=window_size,
            n_frames=n_frames,
            epsilon_start=0.0,
            epsilon_end=0.0,
            device=device,
        )
        state_fn = partial(get_window_state, half_size=half_size)

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent.load(checkpoint)
    agent.epsilon = 0.0  # greedy evaluation
    return agent, state_fn


def _infer_n_frames(agent: DQNAgent | CNNDQNAgent) -> int:
    if isinstance(agent, DQNAgent):
        return agent.q_net.trunk[0].in_features // DQNAgent.BASE_STATE_DIM
    return agent.q_net.conv[0].in_channels // CNNDQNAgent.BASE_CHANNELS


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def run_config(
    config: dict,
    agent: DQNAgent | CNNDQNAgent,
    state_fn,
    n_frames: int,
    n_episodes: int,
    seed: int,
) -> list[dict]:
    results = []
    env = SnakeEnv(seed=seed, **config["env_kwargs"])
    frame_stack = FrameStack(n_frames, state_fn)
    max_steps = config["env_kwargs"].get("max_steps", 500)

    for _ in range(n_episodes):
        env.reset()
        state = frame_stack.reset(env)
        init_length = env.length
        total_reward = 0.0

        while not env.done:
            action = agent.select_action(state)
            result = env.step(action)
            state = frame_stack.step(env)
            total_reward += result.reward

        results.append(
            dict(
                length=env.length,
                total_reward=total_reward,
                steps=env.steps,
                foods_eaten=env.length - init_length,
                death=env.steps < max_steps,  # True = died before timeout
            ),
        )

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(config: dict, results: list[dict]) -> dict:
    lengths = [r["length"] for r in results]
    rewards = [r["total_reward"] for r in results]
    steps = [r["steps"] for r in results]
    foods = [r["foods_eaten"] for r in results]
    deaths = [r["death"] for r in results]
    return dict(
        name=config["name"],
        description=config["description"],
        mean_length=float(np.mean(lengths)),
        std_length=float(np.std(lengths)),
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_steps=float(np.mean(steps)),
        mean_foods=float(np.mean(foods)),
        death_rate=float(np.mean(deaths)),
        n_episodes=len(results),
    )


def compute_adaptability_score(all_metrics: list[dict]) -> float:
    """Mean normalized reward across non-baseline configs relative to baseline.

    Score = 1.0 → same performance everywhere.
    Score < 1.0 → degrades on harder envs.
    """
    baseline = next(m for m in all_metrics if m["name"] == "baseline")
    baseline_reward = baseline["mean_reward"]
    non_baseline = [m for m in all_metrics if m["name"] != "baseline"]
    if abs(baseline_reward) < 1e-8:
        return float(np.mean([m["mean_reward"] for m in non_baseline]))
    return float(np.mean([m["mean_reward"] / baseline_reward for m in non_baseline]))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_summary(all_metrics: list[dict], adaptability_score: float) -> None:
    print("\n" + "=" * 76)
    print("ADAPTABILITY EVALUATION RESULTS")
    print("=" * 76)
    print(f"{'Config':<22} {'MeanLen':>8} {'MeanRew':>10} {'DeathRate':>10} {'Foods':>7}")
    print("-" * 76)
    for m in all_metrics:
        print(
            f"{m['name']:<22} {m['mean_length']:>8.2f} {m['mean_reward']:>10.3f}"
            f" {m['death_rate']:>10.2%} {m['mean_foods']:>7.2f}",
        )
    print("=" * 76)
    print(f"Adaptability Score: {adaptability_score:.4f}")
    print("=" * 76 + "\n")


def log_to_wandb(
    all_metrics: list[dict],
    adaptability_score: float,
    checkpoint: str,
    wandb_project: str,
    run_name: str | None,
) -> None:
    wandb.init(
        project=wandb_project,
        name=run_name or f"eval_{Path(checkpoint).stem}",
        job_type="evaluation",
    )
    wandb.summary["adaptability_score"] = adaptability_score
    wandb.summary["checkpoint"] = checkpoint

    columns = [
        "config",
        "description",
        "mean_length",
        "std_length",
        "mean_reward",
        "std_reward",
        "mean_steps",
        "mean_foods",
        "death_rate",
        "n_episodes",
    ]
    rows = [
        [
            m["name"],
            m["description"],
            m["mean_length"],
            m["std_length"],
            m["mean_reward"],
            m["std_reward"],
            m["mean_steps"],
            m["mean_foods"],
            m["death_rate"],
            m["n_episodes"],
        ]
        for m in all_metrics
    ]
    table = wandb.Table(columns=columns, data=rows)
    wandb.log({"adaptability_results": table, "adaptability_score": adaptability_score})
    wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate adaptability of a trained Snake agent")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument(
        "--agent-type",
        type=str,
        default="mlp",
        choices=["mlp", "cnn", "window-cnn"],
        help="Agent architecture (must match training)",
    )
    p.add_argument("--episodes-per-config", type=int, default=100)
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--window-size", type=int, default=11, help="Window side length for window-cnn")
    p.add_argument("--wandb-project", type=str, default="rl-snake")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--output-json", type=str, default=None, help="Save results to JSON file")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    agent, state_fn = load_agent(
        args.checkpoint,
        args.agent_type,
        args.height,
        args.width,
        window_size=args.window_size,
        device=args.device,
    )
    n_frames = _infer_n_frames(agent)

    configs = get_test_configs(args.height, args.width)
    all_metrics: list[dict] = []

    for config in configs:
        print(f"Running '{config['name']}': {config['description']} ...")
        results = run_config(config, agent, state_fn, n_frames, args.episodes_per_config, args.seed)
        metrics = aggregate(config, results)
        all_metrics.append(metrics)
        print(
            f"  mean_reward={metrics['mean_reward']:.3f}  "
            f"death_rate={metrics['death_rate']:.1%}  "
            f"mean_length={metrics['mean_length']:.1f}",
        )

    adaptability_score = compute_adaptability_score(all_metrics)
    print_summary(all_metrics, adaptability_score)

    log_to_wandb(all_metrics, adaptability_score, args.checkpoint, args.wandb_project, args.run_name)

    if args.output_json:
        out = dict(
            checkpoint=args.checkpoint,
            agent_type=args.agent_type,
            adaptability_score=adaptability_score,
            configs=all_metrics,
        )
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.output_json}")


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
