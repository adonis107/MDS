#!/bin/bash
# ============================================================
#  SLURM job — Post-hoc analysis
#
#  Runs inference caching, threshold sweep, and anomaly
#  clustering after all testing jobs complete.
#
#  Submit via submit_all.sh or manually:
#    sbatch scripts/slurm/posthoc.sh
# ============================================================

#SBATCH --job-name=mds_posthoc
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
#SBATCH --time=20:00:00

#SBATCH --export=ALL
#SBATCH --propagate=NONE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "=== Phase 3a: Inference cache ==="
for YEAR in 2015 2017; do
    echo "=== Inference cache: ${YEAR} ==="
    python scripts/test_inference_cache.py --train-year "$YEAR"
done

echo "=== Phase 3b: Threshold sweep ==="
RUN_TAG="sweep_$(date +%Y%m%d_%H%M%S)"
for YEAR in 2015 2017; do
    echo "=== Threshold sweep: ${YEAR} ==="
    python scripts/test_threshold_sweep.py \
        --train-year "$YEAR" \
        --methods pot dspot rfdr \
        --streams all \
        --run-tag "${RUN_TAG}_${YEAR}"
done

echo "=== Phase 3c: Anomaly clustering ==="
python scripts/anomaly_clustering.py --train-years 2015 2017

echo "=== Post-hoc analysis finished ==="
