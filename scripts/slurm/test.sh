#!/bin/bash
# ============================================================
#  SLURM job — Testing (parameterized by MDS_YEAR)
#
#  Loads ALL three trained models for the given year and runs
#  full evaluation (scores, predictions, consensus, proximity).
#
#  Submit via submit_all.sh or manually:
#    MDS_YEAR=2015 sbatch scripts/slurm/test.sh
# ============================================================

#SBATCH --job-name=mds_test
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
#SBATCH --time=12:00:00

#SBATCH --export=ALL
#SBATCH --propagate=NONE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

: "${MDS_YEAR:?Set MDS_YEAR (2015|2017)}"

echo "=== Testing year=$MDS_YEAR ==="
python scripts/test.py

echo "=== Testing finished: year=$MDS_YEAR ==="
