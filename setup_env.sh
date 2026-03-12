#!/bin/bash
# ============================================================
#  One-time environment setup on Ruche
#  Run interactively on the login node (or in an interactive
#  GPU job for a quick smoke-test).
#
#  Usage:
#    ssh username@ruche.mesocentre.universite-paris-saclay.fr
#    bash setup_env.sh
# ============================================================

set -euo pipefail

CONDA_ENV="mds_market"

# -- 1. Load Miniconda ----------------------------------------
module load miniconda3/25.5.1/none-none

# -- 2. Move .conda to $WORKDIR to avoid home quota (50 GB) --
#       Large environments (PyTorch, etc.) can easily exceed it.
if [ ! -L "$HOME/.conda" ]; then
    echo "Moving .conda to WORKDIR to avoid home quota..."
    mkdir -p "$WORKDIR/.conda"
    if [ -d "$HOME/.conda" ]; then
        cp -r "$HOME/.conda/." "$WORKDIR/.conda/"
        rm -rf "$HOME/.conda"
    fi
    ln -s "$WORKDIR/.conda" "$HOME/.conda"
    echo ".conda -> $WORKDIR/.conda (symlink created)"
fi

# -- 3. Create the conda environment (Python 3.10) ------------
if conda env list | grep -q "^${CONDA_ENV}\s"; then
    echo "Environment '${CONDA_ENV}' already exists — skipping creation."
else
    echo "Creating conda environment '${CONDA_ENV}'..."
    conda create -y -n "$CONDA_ENV" python=3.10
fi

source activate "$CONDA_ENV"

# -- 4. Install dependencies ----------------------------------
# PyTorch with CUDA 12.1 (matches Ruche's Nvidia driver ≥ 525)
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Remaining requirements (skip the torch lines already installed)
pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib pyyaml

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source activate ${CONDA_ENV}"
