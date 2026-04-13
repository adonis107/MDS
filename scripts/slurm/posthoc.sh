#!/bin/bash




source "$HOME/MDS-PDMM/scripts/slurm/env.sh"

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
