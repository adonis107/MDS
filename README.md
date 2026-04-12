# Probabilistic Detection of Market Manipulation

![Status](https://img.shields.io/badge/status-research%20prototype-0A7E8C)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows-5A5A5A)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)

This project implements a full pipeline for anomaly detection in Limit Order Book (LOB) data, with a focus on market-manipulation patterns such as spoofing-like behavior.

It combines:
- rich microstructure feature engineering,
- probabilistic/deep models,
- adaptive thresholding,
- and post-hoc analysis for interpretability and clustering.

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Why This Project Is Useful](#why-this-project-is-useful)
- [Repository Structure](#repository-structure)
- [How to Get Started](#how-to-get-started)
- [Usage Examples](#usage-examples)
- [Where to Get Help](#where-to-get-help)
- [Maintainers and Contributing](#maintainers-and-contributing)

## What This Project Does

The repository provides end-to-end tooling to:
1. preprocess raw daily LOB files into feature-rich parquet datasets,
2. train three anomaly-detection model families,
3. evaluate on held-out, distal, and out-of-sample periods,
4. run threshold sweeps, anomaly clustering, and diagnostics.

### Implemented model families

- **Transformer + OC-SVM (`transformer_ocsvm`)**
  - Learns latent representations of normal market behavior and detects outliers in latent space.
- **Probabilistic Neural Network (`pnn`)**
  - Predicts skew-normal distribution parameters and uses spoofing-gain logic for anomaly signaling.
- **Probabilistic Robust Autoencoder (`prae`)**
  - Uses stochastic sample gating to improve robustness to contamination in training data.

## Why This Project Is Useful

- **Multi-view detection**: combines representation learning, probabilistic forecasting, and robust reconstruction.
- **Domain-aware features**: includes imbalance, dynamics, volatility, event-flow, Hawkes-style memory, and OFI features.
- **Adaptive thresholding**: includes POT/SPOT/DSPOT/RFDR methods for changing score distributions.
- **Operationally ready for HPC**: includes SLURM scripts for train/test/post-hoc orchestration.
- **Research workflow support**: includes notebooks, figure-generation scripts, and deliverables used in project reporting.

## Repository Structure

```text
.
├── config/
│   └── default.yaml                 # Pipeline configuration defaults
├── detection/                       # Core package (data/features/models/thresholds/trainers)
├── scripts/
│   ├── preprocess.py                # Raw LOB -> processed parquet + engineered features
│   ├── train.py                     # Sequential day-by-day model training
│   ├── test.py                      # Evaluation on held-out and OOS periods
│   ├── anomaly_clustering.py        # Cluster detected anomalies
│   ├── test_inference_cache.py      # Cache inference outputs
│   ├── test_threshold_sweep.py      # Compare threshold methods from cached scores
│   ├── run.py                       # Config-driven single-entry pipeline runner
│   ├── setup_env.sh                 # One-time Ruche setup helper
│   └── slurm/                       # HPC job scripts (train/test/posthoc/submit_all)
├── notebooks/                       # Exploratory and analysis notebooks
├── data/
│   ├── raw/
│   └── processed/
├── results/                         # Trained artifacts and evaluation outputs
└── deliverables/                    # Report and presentation sources
```

## How to Get Started

### 1. Clone and create an environment

```bash
git clone <your-repo-url>
cd MDS-Market_Manipulation

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Notes:
- `requirements.txt` installs PyTorch from the CUDA 12.1 index by default.
- If you are CPU-only, adjust the PyTorch install line accordingly.

### 2. Prepare input data

Expected raw input folder:
- `data/raw/TOTF.PA-book/`

Run preprocessing:

```bash
python scripts/preprocess.py
```

Output:
- processed parquet files under `data/processed/TOTF.PA-book/`

### 3. Train models

```bash
# Default runs all model types for year from env (default set in script)
python scripts/train.py

# Example: train only PNN for 2015
# Linux/macOS:
MDS_MODELS=pnn MDS_YEAR=2015 python scripts/train.py
# Windows PowerShell:
# $env:MDS_MODELS='pnn'; $env:MDS_YEAR='2015'; python scripts/train.py
```

### 4. Evaluate models

```bash
# Example: evaluate models trained for 2015
# Linux/macOS:
MDS_YEAR=2015 python scripts/test.py
# Windows PowerShell:
# $env:MDS_YEAR='2015'; python scripts/test.py
```

Primary outputs are written to:
- `results/<YEAR>/test_output/`

### 5. Optional: config-driven run entrypoint

You can run the generic pipeline entrypoint:

```bash
python scripts/run.py -c config/default.yaml
```

Common overrides:

```bash
python scripts/run.py --model pnn --epochs 100 --device cpu --threshold dspot
```

## Usage Examples

### SLURM workflow (Ruche/HPC)

One-time environment bootstrap:

```bash
bash scripts/setup_env.sh
```

Submit full dependency graph (train -> test -> posthoc -> cleanup):

```bash
bash scripts/slurm/submit_all.sh
```

Manual per-stage examples:

```bash
# Train one model-year pair
MDS_MODEL=transformer_ocsvm MDS_YEAR=2015 sbatch scripts/slurm/train.sh

# Test a year
MDS_YEAR=2015 sbatch scripts/slurm/test.sh

# Post-hoc analysis
sbatch scripts/slurm/posthoc.sh
```

### Verification scripts

For quick sanity/integration checks:

```bash
python scripts/verify_dissimilarity.py
python scripts/verify_layer2.py
python scripts/verify_layer3.py
```

## Where to Get Help

If you are onboarding to this repository, the most practical first path is:
1. run preprocessing on a small subset,
2. run one model training (`pnn`) end-to-end,
3. run `scripts/test.py` for one training year.

## Maintainers and Contributing

### Maintainers

Current project maintainers:
- Adonis Jamal
- Jean Martini

Academic supervision:
- Damien Challet (Supervisor)
- Lionel Gabet (Tutor)

### Contributing

Contributions are welcome through issues and pull requests.

Recommended lightweight process:
1. Open an issue describing the bug/feature or experimental change.
2. Create a focused branch.
3. Add or update validation scripts/notebooks when relevant.
4. Submit a pull request with clear reproduction steps and expected outputs.

For substantial methodological changes, include:
- affected scripts/modules,
- parameter changes,
- and a short summary of result impact.

---

If you use this repository in academic work, cite the corresponding report/presentation in [deliverables](deliverables).
