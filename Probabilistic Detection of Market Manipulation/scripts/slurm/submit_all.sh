#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train.sh"
TEST_SCRIPT="$SCRIPT_DIR/test.sh"
POSTHOC_SCRIPT="$SCRIPT_DIR/posthoc.sh"
CLEANUP_SCRIPT="$SCRIPT_DIR/cleanup.sh"

N_RESUME="${1:-1}"
if [[ "$1" == "--resume" ]]; then
    N_RESUME="${2:-3}"
fi

PROJECT_ROOT="$HOME/MDS-PDMM"
mkdir -p "$PROJECT_ROOT/logs"

MODELS=(transformer_ocsvm pnn prae)
YEARS=(2015 2017)

echo "=== Submitting full pipeline (${#MODELS[@]} models x ${#YEARS[@]} years, $N_RESUME resume chain(s)) ==="

declare -A LAST_TRAIN_JOBID

for year in "${YEARS[@]}"; do
    for model in "${MODELS[@]}"; do
        prev_jobid=""
        for run in $(seq 1 "$N_RESUME"); do
            if [ -z "$prev_jobid" ]; then
                out=$(sbatch --export="ALL,MDS_MODEL=$model,MDS_YEAR=$year" \
                             --job-name="train_${model}_${year}" \
                             "$TRAIN_SCRIPT")
            else
                out=$(sbatch --export="ALL,MDS_MODEL=$model,MDS_YEAR=$year" \
                             --job-name="train_${model}_${year}" \
                             --dependency="afterany:${prev_jobid}" \
                             "$TRAIN_SCRIPT")
            fi
            jobid=$(echo "$out" | awk '{print $4}')
            echo "  train ${model} ${year} [run $run/$N_RESUME]: jobid=$jobid"
            prev_jobid="$jobid"
        done
        LAST_TRAIN_JOBID["${model}_${year}"]="$prev_jobid"
    done
done

declare -A TEST_JOBID

for year in "${YEARS[@]}"; do
    job_ids=""
    for model in "${MODELS[@]}"; do
        tid="${LAST_TRAIN_JOBID["${model}_${year}"]}"
        job_ids="${job_ids:+${job_ids}:}${tid}"
    done
    deps="afterok:${job_ids}"

    out=$(sbatch --export="ALL,MDS_YEAR=$year" \
                 --job-name="test_${year}" \
                 --dependency="$deps" \
                 "$TEST_SCRIPT")
    jobid=$(echo "$out" | awk '{print $4}')
    echo "  test ${year}: jobid=$jobid (depends on training)"
    TEST_JOBID["$year"]="$jobid"
done

test_job_ids=""
for year in "${YEARS[@]}"; do
    tid="${TEST_JOBID["$year"]}"
    test_job_ids="${test_job_ids:+${test_job_ids}:}${tid}"
done
test_deps="afterok:${test_job_ids}"

out=$(sbatch --dependency="$test_deps" "$POSTHOC_SCRIPT")
posthoc_jobid=$(echo "$out" | awk '{print $4}')
echo "  posthoc: jobid=$posthoc_jobid (depends on all testing)"

out=$(sbatch --dependency="afterok:${posthoc_jobid}" "$CLEANUP_SCRIPT")
cleanup_jobid=$(echo "$out" | awk '{print $4}')
echo "  cleanup: jobid=$cleanup_jobid (depends on posthoc)"

echo ""
echo "=== Pipeline submitted ==="
echo "Monitor with:  squeue -u \$USER"
echo "Cancel all:    scancel -u \$USER"
