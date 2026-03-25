#!/bin/bash
# ============================================================
#  Slurm job script — Anomaly Clustering
#
#  Submit with: sbatch scripts/submit_anomaly_clustering.sh
# ============================================================

#SBATCH --job-name=mds_anom_clust
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
#SBATCH --time=06:00:00

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

python scripts/anomaly_clustering.py --train-years 2015 2017

echo "=== Anomaly clustering finished ==="
