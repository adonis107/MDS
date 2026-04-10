#!/bin/bash
# ============================================================
#  SLURM job — Cleanup intermediate files to reduce disk usage
#
#  Removes resume_state checkpoints (no longer needed after
#  training completes) and other temporary artefacts.
#
#  Submit via submit_all.sh or manually:
#    sbatch scripts/slurm/cleanup.sh
# ============================================================

#SBATCH --job-name=mds_cleanup
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adonis.jamal@student-cs.fr

#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

#SBATCH --export=NONE
#SBATCH --propagate=NONE

source "$HOME/MDS-PDMM/scripts/slurm/env.sh"

echo "=== Cleaning up intermediate files ==="

for year in 2015 2017; do
    resume_dir="results/${year}/resume_state"
    if [ -d "$resume_dir" ]; then
        echo "Removing $resume_dir"
        rm -rf "$resume_dir"
    fi
done

# Remove stale checkpoint files from project root (created by EarlyStopping)
for f in *_checkpoint.pth; do
    [ -e "$f" ] && echo "Removing $f" && rm -f "$f"
done

# Prune threshold sweep runs older than 30 days
for YEAR in 2015 2017; do
    sweep_dir="results/${YEAR}/threshold_runs"
    if [ -d "$sweep_dir" ]; then
        echo "Pruning sweep runs older than 30 days in $sweep_dir"
        find "$sweep_dir" -maxdepth 1 -mindepth 1 -type d -mtime +30 -exec rm -rf {} +
    fi
done

# Keep only the 50 most recent SLURM log files
if [ -d "logs" ]; then
    echo "Pruning old SLURM logs (keeping 50 newest)"
    ls -t logs/*.out logs/*.err 2>/dev/null | tail -n +51 | xargs -r rm -f
fi

echo "=== Cleanup finished ==="
