#!/bin/bash




source "$HOME/MDS-PDMM/scripts/slurm/env.sh"

echo "=== Cleaning up intermediate files ==="

for year in 2015 2017; do
    resume_dir="results/${year}/resume_state"
    if [ -d "$resume_dir" ]; then
        echo "Removing $resume_dir"
        rm -rf "$resume_dir"
    fi
done

for f in *_checkpoint.pth; do
    [ -e "$f" ] && echo "Removing $f" && rm -f "$f"
done

for YEAR in 2015 2017; do
    sweep_dir="results/${YEAR}/threshold_runs"
    if [ -d "$sweep_dir" ]; then
        echo "Pruning sweep runs older than 30 days in $sweep_dir"
        find "$sweep_dir" -maxdepth 1 -mindepth 1 -type d -mtime +30 -exec rm -rf {} +
    fi
done

if [ -d "logs" ]; then
    echo "Pruning old SLURM logs (keeping 50 newest)"
    ls -t logs/*.out logs/*.err 2>/dev/null | tail -n +51 | xargs -r rm -f
fi

echo "=== Cleanup finished ==="
