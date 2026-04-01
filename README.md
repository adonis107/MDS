# Sneakie : Learning to play the snake game in all its forms

## Overview

This project provides the implemention of a fully custom Snake game environment (`SnakeEnv`) and trains Deep Q-Network (DQN) agents to master it. Three neural architectures are compared across increasingly complex game configurations, from a simple single-food grid to environments with multiple food types, poison, and dynamic moving obstacles.

## Key Features

- **Custom Snake environment** — configurable grid size, multiple food types (gold, silver, poison), static and dynamic obstacles, and a rich reward-shaping API
- **Three agent architectures** — `mlp` (15-feature vector), `cnn` (full-grid 6-channel), and `window-cnn` (ego-centric local crop)
- **DQN variants** — Double DQN, dueling networks, frame stacking, soft/hard target updates, gradient clipping

## Project Structure

```
src/rl_snake/
├── env.py        # SnakeEnv — custom game environment
├── agent.py      # DQNAgent, CNNDQNAgent, state extraction functions
├── train.py      # Training loop + CLI
├── evaluate.py   # Cross-config evaluation + CLI
└── visuals.py    # render_observation(), make_animation(), save_video()

scripts/          # Experiment shell scripts (exp1–exp9)
notebooks/        # Tutorial.ipynb — interactive walkthrough
deliverables/     # Poster, proposal, and report (LaTeX sources)
```

## Getting Started

### Requirements

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd MDS-RL_Snake_Agent

# Install with uv (creates a virtual environment automatically)
uv sync

# Or with pip
pip install -e .
```
### Quick Example
**Train an agent:**

```bash
# Basic MLP agent on a 16×16 grid
uv run src/rl_snake/train.py \
  --agent-type mlp --height 16 --width 16 \
  --episodes 5000 --wandb-project rl-snake

# Full-grid CNN with body-proximity shaping
uv run src/rl_snake/train.py \
  --agent-type cnn --height 10 --width 10 \
  --episodes 5000 --body-proximity-scale 0.1 \
  --cnn-hidden 128 --wandb-project rl-snake
```

Run `uv run src/rl_snake/train.py --help` for the full argument list. Pre-configured experiment scripts are available in `scripts/`.

**Evaluate a trained agent:**

```bash
uv run src/rl_snake/evaluate.py \
  --checkpoint checkpoints/exp1_mlp_basic/ep05000.pt \
  --agent-type mlp --episodes-per-config 200 \
  --output-json logs/results.json --wandb-project rl-snake
```

`notebooks/Tutorial.ipynb` provides an interactive walkthrough of the environment, agent architectures, and training process.

## Authors
- **Adonis Jamal** - CentraleSupélec
- **Fotios Kapotos** - CentraleSupélec
- **Jean-Vincent Martini** - CentraleSupélec