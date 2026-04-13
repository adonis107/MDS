#!/bin/bash

set -euo pipefail

CONDA_ENV="mds_market"

module load miniconda3/25.5.1/none-none

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

if conda env list | grep -q "^${CONDA_ENV}\s"; then
    echo "Environment '${CONDA_ENV}' already exists â€” skipping creation."
else
    echo "Creating conda environment '${CONDA_ENV}'..."
    conda create -y -n "$CONDA_ENV" python=3.10
fi

source activate "$CONDA_ENV"

pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib pyyaml

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source activate ${CONDA_ENV}"
