#!/bin/bash
# ============================================================
#  SLURM job — Testing (parameterized by MDS_YEAR)
#
#  Loads ALL three trained models for the given year and runs
#  full evaluation (scores, predictions, consensus, proximity).
#
#  Single job:
#    MDS_YEAR=2015 sbatch scripts/slurm/test.sh
#
#  Chunked (4 parallel jobs):
#    for i in 0 1 2 3; do
#      MDS_YEAR=2015 MDS_NUM_CHUNKS=4 MDS_CHUNK=$i sbatch scripts/slurm/test.sh
#    done
#    # then: MDS_YEAR=2015 MDS_NUM_CHUNKS=4 sbatch scripts/slurm/test_merge.sh
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
#SBATCH --mem=60G
#SBATCH --time=12:00:00

#SBATCH --export=ALL
#SBATCH --propagate=NONE

source "$HOME/MDS-PDMM/scripts/slurm/env.sh"

: "${MDS_YEAR:?Set MDS_YEAR (2015|2017)}"

export MDS_NUM_CHUNKS=${MDS_NUM_CHUNKS:-1}
export MDS_CHUNK=${MDS_CHUNK:-0}

echo "=== Testing year=$MDS_YEAR chunk=$MDS_CHUNK/$MDS_NUM_CHUNKS ==="
python scripts/test.py

echo "=== Testing finished: year=$MDS_YEAR chunk=$MDS_CHUNK/$MDS_NUM_CHUNKS ==="
