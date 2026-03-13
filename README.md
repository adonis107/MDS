# Probabilistic Detection of Market Manipulation

> Learning the Spoofability of Limit Order Books With Interpretable Probabilistic Neural Networks

A machine learning pipeline for detecting spoofing, layering, and quote stuffing in Limit Order Book (LOB) data using probabilistic and interpretable deep learning models, backed by Extreme Value Theory for adaptive thresholding. Developed by Adonis Jamal and Jean-Vincent Martini for the *Mathematics and Data Science* major at CentraleSupélec under the supervision of Professor Damien Challet.

## Key Features

- **Three complementary detection models**
  - **Transformer + One-Class SVM** — Autoencoder learns a latent representation of normal order flow; OC-SVM flags outliers in latent space.
  - **Probabilistic Neural Network (PNN)** — Predicts price movements as a skewed Gaussian distribution; high negative log-likelihood signals anomalies.
  - **Probabilistic Robust Autoencoder (PRAE)** — Transformer autoencoder with learnable per-sample gates that separate normal from anomalous training data.

- **Rich feature engineering** — Imbalance, dynamics, elasticity, volatility, weighted imbalance (Tao et al.), event flow / rapidity, Hawkes self-exciting order flows, and multi-level Order Flow Imbalance (OFI).

- **Adaptive thresholding** — Peak-Over-Threshold (POT), Streaming POT (SPOT), Drift-aware Streaming POT (DSPOT), and Rolling False Discovery Rate (RFDR) via Extreme Value Theory and Benjamini-Hochberg correction.

- **Interpretability** — Integrated Gradients and Grouped Occlusion sensitivity analysis to attribute anomaly scores to individual features and order-book levels.

- **Spoofing gain estimation** — Economic model (Fabre & Challet) that quantifies the expected profitability of a spoofing strategy given PNN predictions.

## Project Structure

```
├── run.py                  # CLI entry point – runs the full pipeline
├── preprocess.py           # Raw CSV → cleaned Parquet with engineered features
├── config/
│   └── default.yaml        # All pipeline hyperparameters
├── detection/
│   ├── pipeline.py         # Orchestrates data → features → model → threshold → evaluation
│   ├── base.py             # Abstract base classes (BaseDeepModel, BaseDetector, BaseThreshold)
│   ├── data/               # Data loading, sequence creation, scalers
│   ├── features/           # Feature engineering modules (dynamics, hawkes, ofi, …)
│   ├── models/             # Transformer, PNN, PRAE, OC-SVM, Hybrid
│   ├── sensitivity/        # Integrated Gradients, Grouped Occlusion
│   ├── spoofing/           # Spoofing gain calculator
│   ├── thresholds/         # POT, SPOT, DSPOT, RFDR
│   └── trainers/           # Training loop & EarlyStopping callback
└── notebooks/              # Step-by-step analysis (training, model diagnostics, testing, …)
```

## Getting Started

### Prerequisites

- Python ≥ 3.9
- CUDA-capable GPU recommended (the pipeline auto-detects and falls back to CPU)

### Installation

```bash
git clone https://github.com/adonis107/MDS-PDMM.git
cd MDS-PDMM

python -m venv .venv
source .venv/bin/activate  

pip install -r requirements.txt
```

> **Note:** PyTorch is installed with CUDA 12.1 support by default (see the `--extra-index-url` in [requirements.txt](requirements.txt)). Adjust for your CUDA version or remove the flag for CPU-only.

### Data Preparation

Place raw daily LOB CSV files under `data/raw/TOTF.PA-book/`, then run the preprocessing script:

```bash
python preprocess.py
```

This produces cleaned Parquet files in `data/processed/TOTF.PA-book/` with all engineered features.

### Running the Pipeline

```bash
# Run with default configuration
python run.py

# Use a custom config
python run.py -c config/default.yaml

# Override model type and training epochs
python run.py --model pnn --epochs 100

# Change device and threshold method
python run.py --device cpu --threshold dspot
```

All pipeline settings (data paths, feature sets, model architecture, training, thresholding, evaluation) are controlled via [config/default.yaml](config/default.yaml) and can be overridden from the CLI.

### Configuration

The YAML config is organized into sections:

| Section          | Key options                                                       |
|------------------|-------------------------------------------------------------------|
| `data`           | File path, row limit, market hours                                |
| `features`       | Feature sets to compute (`base`, `tao`, `poutre`, `hawkes`, `ofi`), window size |
| `preprocessing`  | Scaler (`minmax` / `standard` / `quantile`), sequence length, train/val split |
| `model`          | Model type (`transformer_ocsvm` / `pnn` / `prae`) and architecture hyperparameters |
| `training`       | Epochs, learning rate, batch size, early-stopping patience, device |
| `threshold`      | Method (`pot` / `spot` / `dspot` / `rfdr`), risk level, window sizes |
| `evaluation`     | F-beta parameter for scoring                                      |

## Notebooks

Interactive notebooks provide step-by-step analysis and visualization:

| # | Notebook | Description |
|---|----------|-------------|
| 1 | Training | Sequential day-by-day training across 24 daily snapshots |
| 2a–c | Model Analysis | Diagnostics for Transformer+OC-SVM, PNN, and PRAE |
| 3 | Testing | Evaluation on held-out data with anomaly classification |
| 4 | Sensitivity Analysis | Feature attribution via Integrated Gradients and Grouped Occlusion |
| 5 | Threshold Analysis | Comparison of POT, DSPOT, RFDR and percentile-based methods |
| 6 | Anomaly Clustering | Clustering of detected anomalies to identify patterns and potential spoofing events |

## Models at a Glance

| Model | Approach | Anomaly Score |
|-------|----------|---------------|
| **Transformer + OC-SVM** | Bottleneck autoencoder → latent vectors → One-Class SVM with Nyström-approximated RBF kernel | Negated OC-SVM decision function |
| **PNN** | Single hidden layer → skewed Gaussian parameters (μ, σ, α) | Negative log-likelihood |
| **PRAE** | Transformer AE with learnable stochastic gates per training sample | Reconstruction error |

## Authors

- **Adonis Jamal** — CentraleSupélec
- **Jean-Vincent Martini** — CentraleSupélec