#!/bin/bash
# ============================================================
#  SLURM job — Training (parameterized by MDS_MODEL, MDS_YEAR)
#
#  Submit via submit_all.sh or manually:
#    MDS_MODEL=pnn MDS_YEAR=2015 sbatch scripts/slurm/train.sh
#
#  Supports checkpointing: re-submit to resume after timeout.
# ============================================================

#SBATCH --job-name=mds_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adonis.jamal@student-cs.fr

#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00

#SBATCH --export=ALL
#SBATCH --propagate=NONE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

: "${MDS_MODEL:?Set MDS_MODEL (transformer_ocsvm|pnn|prae)}"
: "${MDS_YEAR:?Set MDS_YEAR (2015|2017)}"

export MDS_MODELS="$MDS_MODEL"

echo "=== Training model=$MDS_MODEL year=$MDS_YEAR ==="
python scripts/train.py

echo "=== Training finished: model=$MDS_MODEL year=$MDS_YEAR ==="
