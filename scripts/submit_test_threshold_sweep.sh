#!/bin/bash
# ============================================================
#  Slurm job script — Threshold Sweep From Cached Inference
#
#  Submit with: sbatch scripts/submit_test_threshold_sweep.sh
# ============================================================

#SBATCH --job-name=mds_threshold_sweep
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adonis.jamal@student-cs.fr

#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=08:00:00

#SBATCH --export=NONE
#SBATCH --propagate=NONE

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

# Train year can be edited directly in test_threshold_sweep.py (TRAIN_YEAR).
# Optional CLI override example:
# python scripts/test_threshold_sweep.py --train-year 2017 --methods pot dspot rfdr --streams all --run-tag "$RUN_TAG"
python scripts/test_threshold_sweep.py --methods pot dspot rfdr --streams all --run-tag "$RUN_TAG"

echo "=== Threshold sweep finished ==="
