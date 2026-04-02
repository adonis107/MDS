#!/usr/bin/env bash
set -euo pipefail

JOB_NAME=""

# Optional: --name my_job
if [[ "${1:-}" == "--name" ]]; then
  JOB_NAME="$2"
  shift 2
fi

if [ "$#" -lt 2 ]; then
  echo "Usage:"
  echo "  $0 [--name job_name] <prod10|prod40|prod80> <command> [args...]"
  exit 1
fi

PARTITION="$1"
shift
CMD=("$@")

case "$PARTITION" in
  prod10|prod40|prod80) ;;
  *)
    echo "Error: invalid partition '$PARTITION'"
    echo "Valid partitions: prod10, prod40, prod80"
    exit 1
    ;;
esac

# Default job name = executable basename
if [ -z "$JOB_NAME" ]; then
  JOB_NAME=$(basename "${CMD[0]}")
fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Safely quote command for bash -lc
CMD_STR=$(printf '%q ' "${CMD[@]}")

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=08:00:00
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.out

set -euo pipefail
echo "Running on partition: ${PARTITION}"
echo "Job name: ${JOB_NAME}"
echo "Command: ${CMD_STR}"

bash -lc "${CMD_STR}"
EOF