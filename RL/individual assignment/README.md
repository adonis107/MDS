# Text Flappy Bird - Tabular RL Agents

> GLIE Monte-Carlo Control and SARSA(λ) with eligibility traces, trained and compared on the [Text Flappy Bird](https://gitlab-research.centralesupelec.fr/stergios.christodoulidis/text-flappy-bird-gym) Gymnasium environment.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Gymnasium](https://img.shields.io/badge/gymnasium-compatible-green)

## Overview

This project implements and compares two tabular reinforcement learning agents that learn to play **Text Flappy Bird** - a text-rendered Gymnasium environment where an agent controls a bird navigating through pipe gaps by choosing to **flap** or **idle**.

| Agent | Method | Key idea |
|---|---|---|
| **GLIE MC** | First-visit Monte-Carlo control | Updates Q-values from full episode returns; supports linear, inverse, and exponential ε-decay |
| **SARSA(λ)** | SARSA with accumulating eligibility traces | Online TD updates that propagate credit backward through recent state-action pairs |

Both agents use ε-greedy exploration with GLIE-compatible decay schedules, and hyperparameters are selected via parallel grid search.

### Key Results

- **SARSA(λ)** achieves perfect greedy play (hits the 5,000-step evaluation cap every time)
- **GLIE MC** learns a reasonable but higher-variance policy (mean reward ~1,860)
- Neither agent generalises well to harder (unseen) environment configurations - a fundamental limitation of tabular methods under state-space shift

## Project Structure

```
individual assignment/
├── notebooks/
│   ├── 1. MC Agent.ipynb              # MC agent: training, grid search, evaluation
│   ├── 2. Sarsa Agent.ipynb           # SARSA(λ) agent: training, grid search, evaluation
│   ├── 3. Configuration Analysis.ipynb # Cross-configuration generalization study
│   └── RL_TextFlappyBird_JAMAL.ipynb  # Combined end-to-end notebook (all experiments)
├── scripts/
│   ├── MCAgent.py                     # Standalone MC agent with terminal rendering
│   └── SARSAAgent.py                  # Standalone SARSA(λ) agent with terminal rendering
├── results/
│   ├── MC/                            # MC trained model (.pkl), plots
│   ├── SARSA/                         # SARSA trained model (.pkl), plots
│   └── Configurations/               # Generalization analysis outputs
├── report/
│   ├── report.tex                     # LaTeX report (LNCS format)
│   └── report.pdf                     # Compiled PDF
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git (for installing packages from remote repositories)

### Installation

1. **Clone the repository** and navigate to the project root:

   ```bash
   git clone https://github.com/adonis107/MDS-RL.git
   cd MDS-RL
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   This installs NumPy, Pandas, Matplotlib, Seaborn, tqdm, and the Text Flappy Bird Gymnasium environment.

### Training Agents

Open any of the Jupyter notebooks in `individual assignment/notebooks/` and set the training flags to `True`:

```python
TRAIN_AGENT = True
SAVE_AGENT = True
```

Then run all cells. Each notebook runs a grid search over hyperparameters, trains the best configuration for 50,000 episodes, and produces evaluation plots.

**Grid search configurations:**

| Agent | Parameters swept | Configs |
|---|---|---|
| MC | decay type (linear/inverse/exponential), decay rate, ε_min | 10 |
| SARSA(λ) | α ∈ {0.05, 0.1, 0.2}, λ ∈ {0.6, 0.8, 0.9}, ε_min | 18 |

### Watching a Trained Agent Play

Pre-trained agents are saved as `.pkl` files in `results/`. Run the standalone scripts to watch them play in the terminal:

```bash
cd "individual assignment"
python scripts/MCAgent.py
python scripts/SARSAAgent.py
```

The game renders in real time as ASCII art with a 150ms frame delay.

### Running the Notebooks

The recommended order is:

1. **`1. MC Agent.ipynb`** - Train/evaluate the Monte-Carlo agent
2. **`2. Sarsa Agent.ipynb`** - Train/evaluate the SARSA(λ) agent
3. **`3. Configuration Analysis.ipynb`** - Compare both agents across different environment configurations (requires trained models from steps 1–2)

Alternatively, **`RL_TextFlappyBird_JAMAL.ipynb`** contains all experiments in a single notebook.

## Environment

The project uses `TextFlappyBird-v0` (simple state variant):

| Property | Value |
|---|---|
| **State** | `(x_dist, y_dist)` - distance to nearest pipe gap centre |
| **Actions** | `0` = idle, `1` = flap |
| **Reward** | +1 per survived step |
| **Training config** | `height=15, width=20, pipe_gap=4` |
| **State space** | ~300 reachable states (tabular methods are directly applicable) |

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical computation |
| `pandas` | Data analysis and grid search results |
| `matplotlib` / `seaborn` | Visualization |
| `gymnasium` | RL environment interface |
| `text-flappy-bird-gym` | Text Flappy Bird environment |
| `tqdm` | Progress bars |
| `joblib` | Parallel grid search |
| `pickle` | Model serialization |

## Documentation

- **[Report (PDF)](report/report.pdf)** - Full technical write-up in LNCS format covering methods, results, and discussion
- **Notebooks** - Detailed, self-contained experiments with inline explanations and visualizations

## Author

**Adonis JAMAL**
- CentraleSupélec, MDS-MMF
- École Normale Supérieure Paris-Saclay, MVA
