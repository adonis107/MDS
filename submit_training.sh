#!/bin/bash
# ============================================================
#  Slurm job script — Market Manipulation Training
#  Ruche cluster (CentraleSupélec / Paris-Saclay mesocentre)
#
#  Submit with:  sbatch submit_training.sh
# ============================================================

#SBATCH --job-name=mds_training
#SBATCH --output=%x.o%j          # stdout  →  mds_training.o<jobid>
#SBATCH --error=%x.e%j           # stderr  →  mds_training.e<jobid>
#SBATCH --mail-type=ALL           # email on BEGIN / END / FAIL
#SBATCH --mail-user=jean-vincent.martini@student-cs.fr   # <-- replace with your address

# ── Resources ────────────────────────────────────────────────
#SBATCH --partition=gpua100       # A100-40 GB (fastest; 24 h max)
                                  # fall back to: gpu  (V100-32 GB)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # data-loader workers + overhead
#SBATCH --gres=gpu:1             # one GPU is enough (single-node)
#SBATCH --mem=40G                # matches ~1× A100 node RAM budget
#SBATCH --time=24:00:00          # max walltime on gpua100/gpu partitions
# ─────────────────────────────────────────────────────────────

# -- do NOT inherit the submission environment --
#SBATCH --export=NONE
#SBATCH --propagate=NONE

set -euo pipefail

# == 1. Project root on the cluster ===========================
# Adjust this path after uploading your project to $WORKDIR.
PROJECT_ROOT="$WORKDIR/MDS-Market_Manipulation"

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

# == 6. Run training ==========================================
# training.py uses relative paths from the project root
# (data/processed/... and results/)
# It now checkpoints progress per day into results/resume_state,
# so rerunning this same script continues automatically after timeout.
cd "$PROJECT_ROOT"

python train.py

echo "=== Training finished ==="
