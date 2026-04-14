#!/bin/bash




source "$HOME/MDS-PDMM/scripts/slurm/env.sh"

: "${MDS_YEAR:?Set MDS_YEAR (2015|2017)}"
: "${MDS_NUM_CHUNKS:?Set MDS_NUM_CHUNKS (e.g. 4)}"

echo "=== Merging $MDS_NUM_CHUNKS test chunks for year=$MDS_YEAR ==="
python scripts/test_merge.py

echo "=== Merge finished: year=$MDS_YEAR ==="
