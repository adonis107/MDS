#!/bin/bash




set -euo pipefail

PROJECT_ROOT="$HOME/MDS-PDMM"
CONDA_ENV="mds_market"

module purge
module load miniconda3/25.5.1/none-none
source activate "$CONDA_ENV"

echo "=== Python: $(which python) ==="
echo "=== PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)') ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

RUN_TAG="sweep_$(date +%Y%m%d_%H%M%S)"

python scripts/test_threshold_sweep.py --methods pot dspot rfdr --streams all --run-tag "$RUN_TAG"

echo "=== Threshold sweep finished ==="
