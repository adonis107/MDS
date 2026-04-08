#!/bin/bash
# Queue several dependent runs of submit_training.sh.
# Each run resumes from results/resume_state in case of timeout.
#
# Usage:
#   bash queue_resume_chain.sh            # queue 4 runs by default
#   bash queue_resume_chain.sh 6          # queue 6 runs
#   bash queue_resume_chain.sh 6 submit_training.sh

set -euo pipefail

N_RUNS="${1:-4}"
JOB_SCRIPT="${2:-scripts/slurm/train.sh}"

if [ ! -f "$JOB_SCRIPT" ]; then
    echo "Job script not found: $JOB_SCRIPT"
    exit 1
fi

prev_jobid=""

for i in $(seq 1 "$N_RUNS"); do
    if [ -z "$prev_jobid" ]; then
        out="$(sbatch "$JOB_SCRIPT")"
    else
        out="$(sbatch --dependency=afterany:${prev_jobid} "$JOB_SCRIPT")"
    fi

    jobid="$(echo "$out" | awk '{print $4}')"
    echo "Queued run ${i}/${N_RUNS}: jobid=${jobid}"
    prev_jobid="$jobid"
done

echo "Last queued jobid: ${prev_jobid}"
