#!/bin/bash
# ============================================================
#  Slurm job script — Market Manipulation Testing
#  Ruche cluster (CentraleSupélec / Paris-Saclay mesocentre)
#
#  Submit with:  sbatch submit_testing.sh
# ============================================================

#SBATCH --job-name=mds_testing
#SBATCH --output=%x.o%j          # stdout  →  mds_testing.o<jobid>
#SBATCH --error=%x.e%j           # stderr  →  mds_testing.e<jobid>
#SBATCH --mail-type=ALL           # email on BEGIN / END / FAIL
#SBATCH --mail-user=adonis.jamal@student-cs.fr

# ── Resources ────────────────────────────────────────────────
#SBATCH --partition=gpua100       # A100-40 GB (fastest; 24 h max)
                                  # fall back to: gpu  (V100-32 GB)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # data-loader workers + overhead
#SBATCH --gres=gpu:1             # one GPU for inference
#SBATCH --mem=40G                # matches ~1× A100 node RAM budget
#SBATCH --time=12:00:00          # testing is faster than training
# ─────────────────────────────────────────────────────────────

# -- do NOT inherit the submission environment --
#SBATCH --export=NONE
#SBATCH --propagate=NONE

set -euo pipefail

# == 1. Project root on the cluster ===========================
PROJECT_ROOT="$HOME/MDS-PDMM"

# == 2. Conda environment name ================================
CONDA_ENV="mds_market"

# == 3. Load modules ==========================================
module purge
module load miniconda3/25.5.1/none-none   # conda on Ruche

# == 4. Activate your environment =============================
# Use "source activate" (not "conda activate") — see Ruche docs.
source activate "$CONDA_ENV"

# == 5. Confirm device ========================================
echo "=== Python: $(which python) ==="
echo "=== PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# == 6. Run testing ==========================================
# test.py uses relative paths from the project root
# (data/processed/... and results/<TRAIN_YEAR>/test_output/)
# Set TRAIN_YEAR in test.py to "2015" or "2017" before submitting.
cd "$PROJECT_ROOT"

python test.py

echo "=== Testing finished ==="
