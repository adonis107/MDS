#!/bin/bash
# ============================================================
#  SLURM job — Merge chunked test outputs
#
#  Run after all chunk jobs complete:
#    MDS_YEAR=2015 MDS_NUM_CHUNKS=4 sbatch scripts/slurm/test_merge.sh
# ============================================================

#SBATCH --job-name=mds_test_merge
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adonis.jamal@student-cs.fr

#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

#SBATCH --export=ALL
#SBATCH --propagate=NONE

source "$HOME/MDS-PDMM/scripts/slurm/env.sh"

: "${MDS_YEAR:?Set MDS_YEAR (2015|2017)}"
: "${MDS_NUM_CHUNKS:?Set MDS_NUM_CHUNKS (e.g. 4)}"

echo "=== Merging $MDS_NUM_CHUNKS test chunks for year=$MDS_YEAR ==="
python scripts/test_merge.py

echo "=== Merge finished: year=$MDS_YEAR ==="
