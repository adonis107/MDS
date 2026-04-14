#!/bin/bash




source "$HOME/MDS-PDMM/scripts/slurm/env.sh"

: "${MDS_YEAR:?Set MDS_YEAR (2015|2017)}"

export MDS_NUM_CHUNKS=${MDS_NUM_CHUNKS:-1}
export MDS_CHUNK=${MDS_CHUNK:-0}

echo "=== Testing year=$MDS_YEAR chunk=$MDS_CHUNK/$MDS_NUM_CHUNKS ==="
python scripts/test.py

echo "=== Testing finished: year=$MDS_YEAR chunk=$MDS_CHUNK/$MDS_NUM_CHUNKS ==="
