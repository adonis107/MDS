#!/bin/bash
# ============================================================
#  Slurm job script — Test Inference Cache Builder
#
#  Submit with: sbatch submit_test_inference_cache.sh
# ============================================================

#SBATCH --job-name=mds_test_cache
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adonis.jamal@student-cs.fr

#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00

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

# Train year can be edited directly in test_inference_cache.py (TRAIN_YEAR).
# Optional CLI override example:
# python test_inference_cache.py --train-year 2017
python test_inference_cache.py

echo "=== Inference cache run finished ==="
